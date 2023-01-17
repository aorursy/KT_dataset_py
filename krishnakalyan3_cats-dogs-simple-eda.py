!pip install fastai==2.0.9

!pip install python_speech_features
import random, os

import numpy as np

import torch

from fastai.vision.all import *

import matplotlib.pyplot as plt

import librosa.display

import librosa

import IPython.display as ipd

import python_speech_features as psf
def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything()
def plot_sound(path):

    plt.figure(figsize=(14, 5))

    x, sr = librosa.load(path)

    print("length {}, sample-rate {}".format(x.shape, sr))

    librosa.display.waveplot(x, sr=sr)

    return x
path = Path('/kaggle/input/audio-cats-and-dogs/')

train = path / 'cats_dogs' / 'train'

cat0 = (train / 'cat').ls()[0]

dog0 = (train / 'dog').ls()[0]
cat_audio = plot_sound(cat0)
ipd.Audio(cat0)
_ = plot_sound(dog0)
ipd.Audio(dog0)
def audio_to_melspectrogram(conf, audio):

    spectrogram = librosa.feature.melspectrogram(audio, 

                                                 sr=conf.sampling_rate,

                                                 n_mels=conf.n_mels,

                                                 hop_length=conf.hop_length,

                                                 n_fft=conf.n_fft,

                                                 fmin=conf.fmin,

                                                 fmax=conf.fmax)

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram.astype(np.float32)

    return spectrogram
def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):

    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 

                             sr=conf.sampling_rate, hop_length=conf.hop_length,

                            fmin=conf.fmin, fmax=conf.fmax)

    plt.colorbar(format='%+2.0f dB')

    plt.title(title)

    plt.show()
class conf:

    sampling_rate = 44100

    duration = 2

    hop_length = 347*duration

    fmin = 20

    fmax = sampling_rate // 2

    n_mels = 128

    n_fft = n_mels * 20

    samples = sampling_rate * duration
cat_spectogram = audio_to_melspectrogram(conf, cat_audio)
show_melspectrogram(conf, cat_spectogram)
def im_from_audio(fn, sample_rate=44100, window_length=0.05, window_step=0.0045, NFFT=2205):

  

  # Load the audio into an array (signal) at the specified sample rate

  signal, sr = librosa.load(fn, sr=sample_rate)



  # preemphasis

  signal = psf.sigproc.preemphasis(signal, coeff=0.95)



  # get specrogram

  # Get the frames

  frames = psf.sigproc.framesig(signal, 

                                  window_length*sample_rate, 

                                  window_step*sample_rate, 

                                  lambda x:np.ones((x,)))        # Window function 

    

  # magnitude Spectrogram

  spectrogram = np.rot90(psf.sigproc.magspec(frames, NFFT))

  

  # get rid of high frequencies

  spectrogram = spectrogram[512:,:]



  # normalize in [0, 1]

  spectrogram -= spectrogram.min(axis=None)

  spectrogram /= spectrogram.max(axis=None)        



  # Clip to max 512, 512

  spectrogram = spectrogram[:512, :512]

  

  return spectrogram 
cats = pd.DataFrame({'cats': (train/'cat').ls()})

cats['labels'] = 'cat'

dogs = pd.DataFrame({'cats': (train/'dog').ls()})

dogs['labels'] = 'dog'
train_df = pd.concat([cats, dogs], axis=0)

train_df.columns = ['fname', 'labels']

shuffle = train_df.sample(n=len(train_df), random_state=42)

shuffle
def get_x(fn):



  # Use our function from earlier

  spectrogram = im_from_audio(fn) # a 2D array



  # Pad to make sure it is 512 x 512

  w, h = spectrogram.shape

  spectrogram = np.pad(spectrogram, [(0, 512-w), (0, 512-h)])



   # Scale to (0, 255)

  spectrogram  -= spectrogram.min()

  spectrogram *= 255.0/spectrogram.max()



  # Make it uint8

  im_arr = np.array(spectrogram, np.uint8)



  # Make it rgb (hint - some fun tricks you can do here!)

  r = im_arr

  g = im_arr

  b = im_arr



  return np.stack([r, g, b], axis=-1)
def get_fns(_):

  return shuffle['fname']



def get_y(fname):

  return shuffle.loc[shuffle.fname == fname].labels.values[0]



dblock = DataBlock(

    blocks=(ImageBlock, CategoryBlock),    

    get_items=get_fns,

    get_x=get_x,

    get_y=get_y, 

    splitter=RandomSplitter(valid_pct=0.1),

)
dls = dblock.dataloaders(Path(''), bs=32)

dls.show_batch()
learn = cnn_learner(dls, resnet18, metrics=[accuracy])

learn.fine_tune(3)