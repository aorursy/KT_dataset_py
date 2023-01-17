import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")
train.head()
import IPython.display as ipd  # To play sound in the notebook
fname = '../input/audio_train/' + '001ca53d.wav'
ipd.Audio(fname)
from scipy.io import wavfile
rate, data = wavfile.read(fname)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
print(data)
data = np.array([(e/2**16.0)*2 for e in data]) #16 bits tracks, normalization
plt.plot(data)
from scipy import signal
#data
freqs, times, specs = signal.spectrogram(data,
                                         fs=rate,
                                         window="boxcar",
                                        nperseg=13230,
                                        noverlap=0,
                                        detrend=False,
                                        mode = 'complex')

plt.plot(freqs,np.absolute(specs[:,0]))
RATE = 44100                                                     #44.1 kHz
MAX_FRAME = int(RATE * 30)                                       #Max frame = 44.1 kHz * 30 seconds
MIN_FRAME = int(RATE * 0.3)                                      #Min frame = 44.1 kHz * 0.3 seconds
NORM_FACTOR = 1.0/2**16.0                                        # Used later to normalize audio signal
 
MAX_INPUT = int(MAX_FRAME / MIN_FRAME)                           #Size of the second dimension
FREQUENCY_BINS = int(MIN_FRAME / 2) + 1                          #Size of the first dimension

#Input of the NN
nn_input = np.zeros((FREQUENCY_BINS,
                    MAX_INPUT,
                    2))

freqs, times, specs = signal.spectrogram(data,                          #Signal               
                                         fs=RATE,                       #Sampling rate
                                         window="boxcar",               #Rectangular segments
                                         nperseg=MIN_FRAME,             #Number of frames per segments
                                         noverlap=0,                    #No overlap
                                         detrend=False,
                                         mode = 'complex')              #Retrieve complex numbers

#Fill the first component of the 3rd dimension with real part
nn_input[:,:specs.shape[1],0] = np.real(specs)
#Fill the first component of the 3rd dimension with imaginary part
nn_input[:,:specs.shape[1],1] = np.imag(specs)

#Display output for a small part of the tensor
nn_input[:3,:3,:]
from scipy.io import wavfile

RATE = 44100
 
MAX_INPUT = int(MAX_FRAME / MIN_FRAME)
FREQUENCY_BINS = int(MIN_FRAME / 2) + 1

MAX_FRAME = int(RATE * 30)
MIN_FRAME = int(RATE * 0.3)
NORM_FACTOR = 1.0/2**16.0

def make_tensor(fname):
    """
    Brief
    -----
    Creates a 3D tensor from an audio file
    
    Params
    ------
    fname: name of the file to pre-process
    
    Returns
    -------
    A 3D tensor of the audio file as an np.array
    """
    rate, data = wavfile.read(fname)
    data = np.array([(e*NORM_FACTOR)*2 for e in data])
    output = nn_input = np.zeros((FREQUENCY_BINS,
                                  MAX_INPUT,
                                  2))
    freqs, times, specs = signal.spectrogram(data,                                         
                                         fs=RATE,
                                         window="boxcar",
                                         nperseg=MIN_FRAME,
                                         noverlap=0,
                                         detrend=False,
                                         mode = 'complex')
    output[:,:specs.shape[1],0] = np.real(specs)
    output[:,:specs.shape[1],1] = np.imag(specs)
    return output
    
make_tensor(fname)[1:5,1:5,:]
    
    
import os

def make_input_data(audio_dir, fnames=None):
    """
    Brief
    -----
    Pre-process a list of file or a full directory.
    
    Params
    ------
    audio_dir: str
        Directory where files are stored
    fnames: str or None
        List of filenames to preprocess. If None: pre-process the full directory.
    
    Returns
    -------
    A 4D tensor (last dimension refers to observations) as an np.array
    """
    if fnames is None:
        fnames = os.listdir(AUDIO_DIR)
    else:
        fnames = [fname + '.wav' for fname in fnames]
    output = np.zeros((FREQUENCY_BINS,MAX_INPUT,2,len(fnames)))
    i = 0
    for fname in fnames:
        full_path = os.path.join(audio_dir,fname)
        
        output[:,:,:,i] = make_tensor(full_path)
        i+1
    return output


#Example
AUDIO_DIR = '../input/audio_train/'
fnames = ['00044347','001ca53d']
make_input_data(AUDIO_DIR,fnames)

#This takes too long to run
#make_input_data(AUDIO_DIR)


