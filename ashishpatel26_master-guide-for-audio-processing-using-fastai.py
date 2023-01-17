from functools import partial

from pathlib import Path

from multiprocessing import Pool

import os

import shutil

import numpy as np

import pandas as pd

from scipy.io import wavfile

from tqdm import tqdm_notebook as tqdm

import torch.nn.functional as F

from fastai.basic_data import DatasetType

from itertools import islice

import librosa

import librosa.display

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.fftpack import fft

from scipy.signal import get_window

from IPython.display import Audio
def read_file(filename, path='', sample_rate=None, trim=False):

    ''' Reads in a wav file and returns it as an np.float32 array in the range [-1,1] '''

    filename = Path(path) / filename

    file_sr, data = wavfile.read(filename)

    if data.dtype == np.int16:

        data = np.float32(data) / np.iinfo(np.int16).max

    elif data.dtype != np.float32:

        raise OSError('Encounted unexpected dtype: {}'.format(data.dtype))

    if sample_rate is not None and sample_rate != file_sr:

        if len(data) > 0:

            data = librosa.core.resample(data, file_sr, sample_rate, res_type='kaiser_fast')

        file_sr = sample_rate

    if trim and len(data) > 1:

        data = librosa.effects.trim(data, top_db=40)[0]

    return data, file_sr





def write_file(data, filename, path='', sample_rate=44100):

    ''' Writes a wav file to disk stored as int16 '''

    filename = Path(path) / filename

    if data.dtype == np.int16:

        int_data = data

    elif data.dtype == np.float32:

        int_data = np.int16(data * np.iinfo(np.int16).max)

    else:

        raise OSError('Input datatype {} not supported, use np.float32'.format(data.dtype))

    wavfile.write(filename, sample_rate, int_data)





def load_audio_files(path, filenames=None, sample_rate=None, trim=False):

    '''

    Loads in audio files and resamples if necessary.

    

    Args:

        path (str or PosixPath): directory where the audio files are located

        filenames (list of str): list of filenames to load. if not provided, load all 

                                 files in path

        sampling_rate (int): if provided, audio will be resampled to this rate

        trim (bool): 

    

    Returns:

        list of audio files as numpy arrays, dtype np.float32 between [-1, 1]

    '''

    path = Path(path)

    if filenames is None:

        filenames = sorted(list(f.name for f in path.iterdir()))

    files = []

    for filename in tqdm(filenames, unit='files'):

        data, file_sr = read_file(filename, path, sample_rate=sample_rate, trim=trim)

        files.append(data)

    return files

    

        

def _resample(filename, src_path, dst_path, sample_rate=16000, trim=True):

    data, sr = read_file(filename, path=src_path, sample_rate=sample_rate, trim=trim)

    write_file(data, filename, path=dst_path, sample_rate=sample_rate)

    



def resample_path(src_path, dst_path, **kwargs):

    transform_path(src_path, dst_path, _resample, **kwargs)    

    



def _to_mono(filename, dst_path):

    data, sr = read_file(filename)

    if len(data.shape) > 1:

        data = librosa.core.to_mono(data.T) # expects 2,n.. read_file returns n,2

    write_file(data, dst_path/filename.name, sample_rate=sr)





def convert_to_mono(src_path, dst_path, processes=None):

    src_path, dst_path = Path(src_path), Path(dst_path)

    os.makedirs(dst_path, exist_ok=True)

    filenames = list(src_path.iterdir())

    convert_fn = partial(_to_mono, dst_path=dst_path)

    with Pool(processes=processes) as pool:

        with tqdm(total=len(filenames), unit='files') as pbar:

            for _ in pool.imap_unordered(convert_fn, filenames):

                pbar.update()

                

                

def transform_path(src_path, dst_path, transform_fn, fnames=None, processes=None, delete=False, **kwargs):

    src_path, dst_path = Path(src_path), Path(dst_path)

    if dst_path.exists() and delete:

        shutil.rmtree(dst_path)

    os.makedirs(dst_path, exist_ok=True)

    

    _transformer = partial(transform_fn, src_path=src_path, dst_path=dst_path, **kwargs)

    if fnames is None:

        fnames = [f.name for f in src_path.iterdir()]

    with Pool(processes=processes) as pool:

        with tqdm(total=len(fnames), unit='files') as pbar:

            for _ in pool.imap_unordered(_transformer, fnames):

                pbar.update()





class RandomPitchShift():

    def __init__(self, sample_rate=22050, max_steps=3):

        self.sample_rate = sample_rate

        self.max_steps = max_steps

    def __call__(self, x):

        n_steps = np.random.uniform(-self.max_steps, self.max_steps)

        x = librosa.effects.pitch_shift(x, sr=self.sample_rate, n_steps=n_steps)

        return x





def _make_transforms(filename, src_path, dst_path, tfm_fn, sample_rate=22050, n_tfms=5):

    data, sr = read_file(filename, path=src_path)

    fn = Path(filename)

    # copy original file 

    new_fn = fn.stem + '_00.wav'

    write_file(data, new_fn, path=dst_path, sample_rate=sample_rate)

    # make n_tfms modified files

    for i in range(n_tfms):

        new_fn = fn.stem + '_{:02d}'.format(i+1) + '.wav'

        if not (dst_path/new_fn).exists():

            x = tfm_fn(data)

            write_file(x, new_fn, path=dst_path, sample_rate=sample_rate)





def pitch_shift_path(src_path, dst_path, max_steps, sample_rate, n_tfms=5):

    pitch_shifter = RandomPitchShift(sample_rate=sample_rate, max_steps=max_steps)

    transform_path(src_path, dst_path, _make_transforms, 

                   tfm_fn=pitch_shifter, sample_rate=sample_rate, n_tfms=n_tfms)

    

    

def rand_pad_crop(signal, pad_start_pct=0.1, crop_end_pct=0.5):

    r_pad, r_crop = np.random.rand(2)

    pad_start = int(pad_start_pct * r_pad * signal.shape[0])

    crop_end  = int(crop_end_pct * r_crop * signal.shape[0]) + 1

    return F.pad(signal[:-crop_end], (pad_start, 0), mode='constant')





def get_transforms(min_len=2048):

    def _train_tfm(x):

        x = rand_pad_crop(x)

        if x.shape[0] < min_len:

            x = F.pad(x, (0, min_len - x.shape[0]), mode='constant')

        return x

    

    def _valid_tfm(x):

        if x.shape[0] < min_len:

            x = F.pad(x, (0, min_len - x.shape[0]), mode='constant')

        return x

  

    return [_train_tfm],[_valid_tfm]





def save_submission(learn, filename, tta=False):

    fnames = [Path(f).name for f in learn.data.test_ds.x.items]

    get_predsfn = learn.TTA if tta else learn.get_preds

    preds = get_predsfn(ds_type=DatasetType.Test)[0]

    top_3 = np.array(learn.data.classes)[np.argsort(-preds, axis=1)[:, :3]]

    labels = [' '.join(list(x)) for x in top_3]

    df = pd.DataFrame({'fname': fnames, 'label': labels})

    df.to_csv(filename, index=False)

    return df





def precision(y_pred, y_true, thresh:float=0.2, eps:float=1e-9, sigmoid:bool=True):

    "Computes the f_beta between preds and targets"

    if sigmoid: y_pred = y_pred.sigmoid()

    y_pred = (y_pred>thresh).float()

    y_true = y_true.float()

    TP = (y_pred*y_true).sum(dim=1)

    prec = TP/(y_pred.sum(dim=1)+eps)

    return prec.mean()





def recall(y_pred, y_true, thresh:float=0.2, eps:float=1e-9, sigmoid:bool=True):

    "Computes the f_beta between preds and targets"

    if sigmoid: y_pred = y_pred.sigmoid()

    y_pred = (y_pred>thresh).float()

    y_true = y_true.float()

    TP = (y_pred*y_true).sum(dim=1)

    rec = TP/(y_true.sum(dim=1)+eps)

    return rec.mean()
plt.rcParams['figure.figsize'] = (12, 3)
DATA = Path('../input/')

AUDIO = DATA/'audio_train/audio_train/'

CSV = DATA/'train.csv'



df = pd.read_csv(CSV)

df.head(3)
row = df.iloc[1] # saxophone clip

filename = AUDIO / row.fname



# open the audio file

clip, sample_rate = librosa.load(filename, sr=None)



print('Sample Rate   {} Hz'.format(sample_rate))

print('Clip Length   {:3.2f} seconds'.format(len(clip)/sample_rate))
three_seconds = sample_rate * 3

clip = clip[:three_seconds]
timesteps = np.arange(len(clip)) / sample_rate  # in seconds



fig, ax = plt.subplots(2, figsize=(12, 5))

fig.subplots_adjust(hspace=0.5)



# plot the entire clip 

ax[0].plot(timesteps, clip)

ax[0].set_xlabel('Time (s)')

ax[0].set_ylabel('Amplitude')

ax[0].set_title('Raw Audio: {} ({} samples)'.format(row.label, len(clip)))





n_fft = 1024 # frame length 

start = 45000 # start at a part of the sound thats not silence.. 

x = clip[start:start+n_fft]



# mark location of frame in the entire signal

ax[0].axvline(start/sample_rate, c='r') 

ax[0].axvline((start+n_fft)/sample_rate, c='r')



# plot N samples 

ax[1].plot(x)

ax[1].set_xlabel('Samples')

ax[1].set_ylabel('Amplitude')

ax[1].set_title('Raw Audio: {} ({} samples)'.format(row.label, len(x)));
Audio(clip, rate=sample_rate)
window = get_window('hann', n_fft)

wx = x * window



fig, ax = plt.subplots(1, 2, figsize=(16, 2))

ax[0].plot(window)

ax[1].plot(wx);
# Compute (real) FFT on window

X = fft(x, n_fft)

X.shape, X.dtype
# We only use the first (n_fft/2)+1 numbers of the output, as the second half if redundant

X = X[:n_fft//2+1]



# Convert from rectangular to polar, usually only care about magnitude

X_magnitude, X_phase = librosa.magphase(X)



plt.plot(X_magnitude);



X_magnitude.shape, X_magnitude.dtype
# we hear loudness in decibels (on a log scale of amplitude)

X_magnitude_db = librosa.amplitude_to_db(X_magnitude)



plt.plot(X_magnitude_db);
hop_length = 512

stft = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)

stft_magnitude, stft_phase = librosa.magphase(stft)

stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)



plt.figure(figsize=(12, 6))

librosa.display.specshow(stft_magnitude_db, x_axis='time', y_axis='linear', 

                         sr=sample_rate, hop_length=hop_length)



title = 'n_fft={},  hop_length={},  time_steps={},  fft_bins={}  (2D resulting shape: {})'

plt.title(title.format(n_fft, hop_length, 

                       stft_magnitude_db.shape[1], 

                       stft_magnitude_db.shape[0], 

                       stft_magnitude_db.shape));
# number of mel frequency bands 

n_mels = 64



fig, ax = plt.subplots(1, 2, figsize=(12, 5))



fmin = 0            

fmax = 22050 # sample_rate/2

mel_spec = librosa.feature.melspectrogram(clip, n_fft=n_fft, hop_length=hop_length,

                                          n_mels=n_mels, sr=sample_rate, power=1.0,

                                          fmin=fmin, fmax=fmax)

mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

librosa.display.specshow(mel_spec_db, x_axis='time',  y_axis='mel', 

                         sr=sample_rate, hop_length=hop_length, 

                         fmin=fmin, fmax=fmax, ax=ax[0])

ax[0].set_title('n_mels=64, fmin=0, fmax=22050')



fmin = 20           

fmax = 8000

mel_spec = librosa.feature.melspectrogram(clip, n_fft=n_fft, hop_length=hop_length,

                                          n_mels=n_mels, sr=sample_rate, power=1.0, 

                                          fmin=fmin, fmax=fmax)

mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

librosa.display.specshow(mel_spec_db, x_axis='time',  y_axis='mel', 

                         sr=sample_rate, hop_length=hop_length, 

                         fmin=fmin, fmax=fmax, ax=ax[1])

ax[1].set_title('n_mels=64, fmin=20, fmax=8000')



plt.show()