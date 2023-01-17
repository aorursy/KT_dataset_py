import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, fftpack
from scipy.interpolate import griddata
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn, optim
import scipy
import scipy.io as sio
import scipy.io.wavfile
from scipy.signal import find_peaks
import copy
import pandas as pd
from pandas import Series
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

## Raw Label values for dataset, 'unmarked data' and 'extended palm' has been excluded from set of data 
labels = {0:'unmarked data', 1:'hand at rest', 2:'hand clenched in a fist',3:'wrist flexion', 
          4:'wrist extension', 5:'radial deviations', 6: 'ulnar deviations', 7:'extended palm'}

## Reassigned Label values for datasets for easy computation
labels = {0:'hand at rest', 1:'hand clenched in a fist',2:'wrist flexion', 
          3:'wrist extension', 4:'radial deviations', 5: 'ulnar deviations'}

heading = np.array(['time', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5',
                    'channel6', 'channel7', 'channel8', 'class'], dtype='<U8')

## Directory of datasets from working folder
DATA_FOLDER = "/kaggle/input/emg-data-for-gesturesuci/EMG_data_for_gestures-master/EMG_data_for_gestures-master/"
# import module we'll need to import our custom module
from shutil import copyfile

# copy loader.py file into the working directory
copyfile(src = "/kaggle/input/emghelper/loader.py", dst = "/kaggle/working/loader.py")

# import custom dataloader for EMG datasets
from loader import DataLoader

fs = 200 ##Resampling frequency
trainloader = DataLoader(DATA_FOLDER, resample_fs=fs, training_set=True)
testloader = DataLoader(DATA_FOLDER, resample_fs=None)
def do_nothing(data, label):
    """do_nothing callback function passed to DataLoader return exact datasets
    Attributes:
        data (arr): time series data for channels-8
        label (arr): gestures for time series data
    Returns:
        data, label
    """
    return data, label


def rms_transform(data, label):
    """rms_transform callback function passed to DataLoader returns root-mean-square of time-series data
    Attributes:
        data (array): time series data for channels-8
        label (array): gestures for time series data
    Returns:
        rms_norm (array(1,8)) : normalized root mean squared data for 8-channels
        groups (array(1,6)): gestures for time series data
    """
    
    df = pd.DataFrame(data=data, columns=heading[:-1], index=None)
    df['class'] = label
    grouped_df = df[df['class']<7].groupby('class')
    df_rms = grouped_df.apply(np.std).drop(['time','class'], axis=1)
    rms_norm = df_rms.div(df_rms.max(axis=1), axis=0)
    rms_norm.columns = [1,2,3,4,5,6,7,8]
    
    groups = [name for name,unused_df in grouped_df]
    
    return rms_norm, groups
def get_npeaks(data, n):
    """Get n number of top-most peaks from data
    """
    peaks, _ = find_peaks(data, height=0)
    #print('get_npeaks_peaks', peaks.shape)
    threshold = np.sort(data[peaks])[-n-1]
    ix = data[peaks]>threshold
    n_peaks = peaks[ix]
    
    output = list(zip(n_peaks, data[n_peaks]))
    sort_out = output.copy()
    sort_out.sort(key=lambda elem: elem[1])
    sort_out = sort_out[-n:]
    split = list(zip(*sort_out))
    
    return list(split[0]), list(split[1])
for idx, (data, label) in enumerate(trainloader.get_data(do_nothing)):
    """
    Get an overview of datasets along with some smoothing using moving average with window size 10
    Investigate fourier transform properties with topmost n peaks also for latter reference
    """
    label = label.reshape(-1)
    datax = data[label==0] #label is 1
    if(idx==2):
        for channel in np.arange(1,9):
            timevec = datax[:,0]
            amplx = datax[:,channel]-torch.mean(datax[:,channel])
            # convert data to pandas.Series for smoothing
            ampl = Series(amplx).rolling(window=10)
            ampl = ampl.mean()
            pow_sdx = np.abs(fftpack.fft(amplx.numpy()))**2
            pow_sd = np.abs(fftpack.fft(np.array(ampl.dropna())))**2
            fftfreqx = fftpack.fftfreq(len(pow_sdx))*fs #Hz
            fftfreq = fftpack.fftfreq(len(pow_sd))*fs #Hz
            ix_x = fftfreqx > 0
            ix = fftfreq > 0
            
            peaks, _ = get_npeaks(pow_sd[ix], 5)
            
            fig, ax = plt.subplots(1, 2, figsize=(16,4))
            ax[0].plot(timevec, amplx, label='raw data')
            ax[0].plot(timevec, ampl, label='smoothened data',color='r')
            ax[0].legend(loc='upper right')
            ax[0].set_xlabel('Time in ms')
            ax[0].set_ylabel('Signal amplitude in mV')
            ax[0].set_title('Amplitude plot for channel-{}'.format(channel))
            
            ax[1].plot(fftfreqx[ix_x], pow_sdx[ix_x], label='raw data')
            ax[1].plot(fftfreq[ix], pow_sd[ix], label='smoothened data',color='r')            
            ax[1].plot(fftfreq[ix][peaks], pow_sd[ix][peaks],'x',color='k')
            ax[1].grid()
            ax[1].legend(loc='upper right')
            ax[1].set_xlim([0,30])
            ax[1].set_xlabel('Frequency (Hz)')
            ax[1].set_ylabel('Power spectra density')
            ax[1].set_title('FFT plot for channel-{}'.format(channel))
    else: pass
class NetworkRMS(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(8, 12)
        self.fc2 = nn.Linear(12, 24)
        # Output layer, 6 units - one for each digit
        self.fc3 = nn.Linear(24, 6)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
model = NetworkRMS()

# if GPU is available, move the model to GPU
if train_on_gpu:
    model.cuda()
    
criterion = nn.NLLLoss() #Resampling- Cross entropy loss: 76.7%, NLLLoss: 80%, No sampling:78.8%, Sampling: 81.3%
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 40
steps = 0

train_losses, test_losses = [], []
train_len = len(list(trainloader.get_data(rms_transform)))
test_len = len(list(testloader.get_data(rms_transform)))

for e in range(epochs):
    running_loss = 0
    
    for data, label in trainloader.get_data(rms_transform):
        
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, label = data.cuda(), label.cuda()
            
        optimizer.zero_grad()
        
        train_data, train_label = data, label
        log_ps = model(train_data)
        #print(log_ps, label)
        loss = criterion(log_ps, train_label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
       
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for data, label in testloader.get_data(rms_transform):
                
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, label = data.cuda(), label.cuda()
                    
                ## Flatten 5peaks per channel to a vector of size 40
                test_data, test_label = data, label
                log_ps = model(test_data)
                test_loss += criterion(log_ps, test_label)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == test_label.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(running_loss/train_len)
        test_losses.append(test_loss/test_len)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/train_len),
              "Test Loss: {:.3f}.. ".format(test_loss/test_len),
              "Test Accuracy: {:.3f}".format(accuracy/test_len))
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)