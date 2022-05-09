
import mne
from pathlib import Path


data_path = Path(mne.datasets.sample.data_path(verbose=False))
sample_dir = data_path / 'MEG' / 'sample'
subjects_dir = data_path / 'subjects'



raw_path = sample_dir / 'sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)



ica = mne.preprocessing.ICA(
    n_components=5,  # fit 5 ICA components
    fit_params=dict(tol=0.01)  # assume very early on that ICA has converged
)

ica.fit(inst=raw)

# create epochs based on EOG events, find EOG artifacts in the data via pattern
# matching, and exclude the EOG-related ICA components
eog_epochs = mne.preprocessing.create_eog_epochs(raw=raw)
eog_components, eog_scores = ica.find_bads_eog(
    inst=eog_epochs,
    ch_name='EEG 001',  # a channel close to the eye
    threshold=1  # lower than the default threshold
)
ica.exclude = eog_components

report = mne.Report(title='ICA example')
report.add_ica(
    ica=ica,
    title='ICA cleaning',
    picks=[0, 1],  # only plot the first two components
    inst=raw,
    eog_evoked=eog_epochs.average(),
    eog_scores=eog_scores,
    n_jobs=1  # could be increased!
)
report.save('report_ica.html', overwrite=True)