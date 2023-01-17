import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os,glob,re,math

#from pathlib import Path



#from tqdm import tqdm_notebook as tqdm

import cv2

import librosa

#from itertools import islice

import matplotlib.pyplot as plt

#from multiprocessing.pool import Pool

#from sklearn.model_selection import StratifiedKFold

import tensorflow as tf

from scipy.signal import freqz

import warnings

warnings.filterwarnings('ignore')
from librosa.display import waveplot
from scipy.signal import butter, lfilter

from skimage.restoration import denoise_wavelet 

from scipy import signal
ROOT_DIR = '../input/birdsong-recognition/'

TRAIN_AUDIO = f'{ROOT_DIR}/train_audio'
CLASS = os.listdir('../input/birdsong-recognition/train_audio')
train_audio = glob.glob('../input/birdsong-recognition/train_audio/*/*.mp3')

train_audio_1 =glob.glob('../input/xeno-canto-bird-recordings-extended-a-m/A-M/*/*.mp3')

train_audio_2 = glob.glob('../input/xeno-canto-bird-recordings-extended-n-z/N-Z/*/*.mp3')
df = pd.read_csv('../input/birdsong-recognition/train.csv')
df = df[['ebird_code', 'filename', 'duration','author','country','rating']]

df1 = pd.read_csv('../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv')[['ebird_code', 'filename', 'duration','author','country','rating']]

df2 = pd.read_csv('../input/xeno-canto-bird-recordings-extended-n-z/train_extended.csv')[['ebird_code', 'filename', 'duration','author','country','rating']]
frames = pd.concat([df,df1,df2])

#frames = frames.loc[frames.duration <= 30]
path = train_audio + train_audio_1 + train_audio_2

path.remove('../input/birdsong-recognition/train_audio/lotduc/XC195038.mp3')

path.remove('../input/xeno-canto-bird-recordings-extended-a-m/A-M/houspa/XC555482.mp3')
from scipy.signal import butter, lfilter



#https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html



def butter_bandpass(lowcut, highcut, fs, order=5):

    nyq =  fs//2 # Nyquist sampling rate

    low = lowcut/ nyq

    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')

    return b, a





def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)

    y = lfilter(b, a, data)

    return y
def waveletDenoising(data):

        

    #Wavelet Denosing using scikit-image

    

    im_bayes = denoise_wavelet(data,)

    

    return im_bayes



def audio_norm(data):

    '''Normalization of audio'''

    max_data = np.max(data)

    min_data = np.min(data)

    data = (data - min_data)/(max_data - min_data + 1e-6)

    return data
class config:

    shape = (128,256)

    rate = 32000

    low_cut = 500.0 #low pass filter

    high_cut = 15000.0 #high pass filter

    order = 5

    duration = 30 #sec

    nq_rate = 0.2 * rate

    n_fft = 4096

    hop_len = 1024

    n_mels = 128

    fmin = 100.0

    fmax = 15000.0

    
def read_audio(audio):

    sig, rt = librosa.load(audio, duration = config.duration, mono = True,sr = config.rate,res_type='kaiser_fast')

    

    return sig,rt
def mel_spec(sig, preemphasis = True, normalize = True):

    

        #Split an audio signal into non-silent intervals,

        #from discussion https://www.kaggle.com/c/birdsong-recognition/discussion/167264

        sig = librosa.effects.trim(y = sig)

        

        if preemphasis:

            sig = librosa.effects.preemphasis(y=sig[0],) #coef = 0.95

        #Librosa mel-spectrum

        

        HOP_LEN = len(sig) // (config.shape[1] - 1)

        

        melspec = librosa.feature.melspectrogram(y=sig, sr=config.rate, 

                                       hop_length = HOP_LEN,

                                      n_mels = config.n_mels,

                                      fmax = config.fmax, 

                                      fmin = config.fmin,center = True,

                                      window = 'hamming')

        



        

        melspec = librosa.power_to_db(melspec,ref=np.max,) #using default top_db = 80 sometime works better

        

        #melspec = librosa.core.pcen(melspec,)

        

        #mfcc 

        #melspec = librosa.feature.mfcc(S=s,n_mfcc = config.n_mels)

               

        melspec =  melspec[::-1, ...] #flip lower frequency

        

        melspec = melspec[:config.shape[0], :config.shape[1]] #trim to desired shape

     

        # Normalize values between 0 and 1

        if normalize:

            melspec  = audio_norm(melspec)

    

        return melspec.astype('float32')
def filtered(audiof):

    sig = read_audio(audiof)[0] #read audio

    sig_fit = butter_bandpass_filter(sig,config.low_cut, config.high_cut, config.rate, config.order)

    return waveletDenoising(mel_spec(sig_fit))
sig,rt = read_audio('../input/birdsong-recognition/train_audio/bkbmag1/XC114081.mp3')



y = butter_bandpass_filter(sig,config.low_cut, config.high_cut, config.rate, config.order)



plt.figure()

plt.subplot(3,1,1)

waveplot(y=sig, sr = rt)

plt.title('original_signal')

plt.subplot(3,1,2)

waveplot(y=y, sr = rt)

plt.title('filtered_signal')
sig,rt = read_audio(path[6969])



y = waveletDenoising(butter_bandpass_filter(sig,config.low_cut, config.high_cut, config.rate, config.order))



plt.figure()

plt.subplot(3,1,1)

waveplot(y=sig, sr = rt)

plt.title('original_signal')

plt.subplot(3,1,2)

waveplot(y=y, sr = rt)

plt.title('filtered_signal')
# settings

h, w = 10, 10       

nrows, ncols = 8, 4  

figsize = [12, 12]  





xs = np.linspace(0, 2*np.pi, 60)  

ys = np.abs(np.sin(xs))           





fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)





for i, axi in enumerate(ax.flat):

    

    img = path[i]

    axi.imshow(filtered(img))    

    rowid = i // ncols

    colid = i % ncols

    

    axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))





ax[0][2].plot(xs, 3*ys, color='red', linewidth=3)

ax[4][3].plot(ys**2, xs, color='green', linewidth=3)



plt.tight_layout(True)

plt.show()
img = filtered(path[10])

#img = mono_to_color(img)

cv2.imwrite('melspec.png', img*255)
plt.imshow(cv2.imread('./melspec.png'))