import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import spectrogram
import pywt
import pywt.data

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import io
import cv2
from PIL import Image, ImageChops

_input_dir = '/kaggle/input/predict-volcanic-eruptions-ingv-oe/'
_train_dir = f'{_input_dir}train/'
_test_dir = f'{_input_dir}test/'

train = pd.read_csv(f'{_input_dir}train.csv')
test = pd.read_csv(f'{_input_dir}sample_submission.csv')

# Resample starting at 25th earthquake getting every 15th earthquake in order of increasing time to eruption
train_resample = train.sort_values(['time_to_eruption'])\
                      .reset_index(drop=True) \
                      .iloc[10::15, :]
train_resample.head(3)
def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")
    
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest(coeff[-level])

    # Calculate the univeral threshold
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def denoise_transform(signals):
    for i in range(1, 11):
        signal_col = f'sensor_{i}'
        if (signals[signal_col].isnull().sum() > 0) & (signals[signal_col].isnull().sum() < 20000):
            # Impute the missing values if there are gaps in our data
            imp = IterativeImputer(max_iter=10, random_state=42)
            signals[signal_col] = imp.fit_transform(signals[signal_col].to_numpy().reshape(-1, 1))[:,0]
        
        signals[signal_col] = denoise_signal(signals[signal_col], level=2)[:-1]
    
    return signals


def transform_signals(signals):
    for i in range(1, 11):
        signal_col = f'sensor_{i}'
        if (signals[signal_col].isnull().sum() > 0) & (signals[signal_col].isnull().sum() < 20000):
            imp = IterativeImputer(max_iter=10, random_state=42)
            signals[signal_col] = imp.fit_transform(signals[signal_col].to_numpy().reshape(-1, 1))[:,0]
        
        
        for l in range(1,3):
            signals[f'sensor_{i}_l{l}'] = denoise_signal(signals[signal_col], level=l)[:-1]
            signals[f'sensor_{i}_l{l}_sum'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).sum()
            signals[f'sensor_{i}_l{l}_mean'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).mean()
            signals[f'sensor_{i}_l{l}_std'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).std()
            signals[f'sensor_{i}_l{l}_var'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).var()
            signals[f'sensor_{i}_l{l}_max'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).max()
            signals[f'sensor_{i}_l{l}_min'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).min()
            signals[f'sensor_{i}_l{l}_median'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).median()
            signals[f'sensor_{i}_l{l}_skew'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).skew()
            signals[f'sensor_{i}_l{l}_kurt'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).kurt()
            signals[f'sensor_{i}_l{l}_q95'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).quantile(.95, interpolation='midpoint')
            signals[f'sensor_{i}_l{l}_q87'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).quantile(.87, interpolation='midpoint')
            signals[f'sensor_{i}_l{l}_q13'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).quantile(.13, interpolation='midpoint')
            signals[f'sensor_{i}_l{l}_q05'] = signals[f'sensor_{i}_l{l}'].rolling(50, min_periods=1).quantile(.05, interpolation='midpoint')
    
    return signals
fig, axes = plt.subplots(4, 4, figsize=(40, 20))

def plot_row(temp, sensor, ax1, ax2, ax3, ax4):
    ax1.plot(temp.index, temp[sensor], color='pink')
    ax1.set_title(f'{sensor}')
                  
    ax2.plot(temp.index, denoise_transform(temp.copy())[sensor], color='pink')
    ax2.set_title(f'{sensor} Denoised')
    
    # Spectrogram
    f, t, Sxx = spectrogram(temp[sensor])
    ax3.pcolormesh(t, f, np.log10(Sxx), shading='nearest')
    ax3.set_title(f'{sensor} Spectrogram')
    ax3.set_ylabel('f [Hz]')
    ax3.set_xlabel('t [sec]')
    ax3.set_yscale('symlog')
    
    # Denoise Spectrogram
    f, t, Sxx = spectrogram(denoise_transform(temp.copy())[sensor])
    ax4.pcolormesh(t, f, np.log10(Sxx), shading='nearest')
    ax4.set_title(f'{sensor} Denoised Spectrogram')
    ax4.set_ylabel('f [Hz]')
    ax4.set_xlabel('t [sec]')
    ax4.set_yscale('symlog')
    

temp = pd.read_csv(f'{_train_dir}1593620672.csv')
plot_row(temp, 'sensor_1', axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3])
plot_row(temp, 'sensor_2', axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3])
plot_row(temp, 'sensor_3', axes[2, 0], axes[2, 1], axes[2, 2], axes[2, 3])
plot_row(temp, 'sensor_4', axes[3, 0], axes[3, 1], axes[3, 2], axes[3, 3])

plt.show()
fig, axes = plt.subplots(3, 3, figsize=(40, 15))

def plot_denoise_spectrogram(temp, sensor, ax):
    # Denoise Spectrogram
    f, t, Sxx = spectrogram(denoise_transform(temp.copy())[sensor])
    ax.pcolormesh(t, f, np.log10(Sxx), shading='nearest')
    ax.set_title(f'{sensor} Denoised Spectrogram')
    ax.set_ylabel('f [Hz]')
    ax.set_xlabel('t [centi-seconds]')
    ax.set_yscale('symlog')
    
    return ax

    
temp = pd.read_csv(f'{_train_dir}1992733806.csv')
plot_denoise_spectrogram(temp, 'sensor_1', axes[0, 0])
plot_denoise_spectrogram(temp, 'sensor_2', axes[0, 1])
plot_denoise_spectrogram(temp, 'sensor_3', axes[0, 2])
plot_denoise_spectrogram(temp, 'sensor_4', axes[1, 0])
plot_denoise_spectrogram(temp, 'sensor_5', axes[1, 1])
plot_denoise_spectrogram(temp, 'sensor_6', axes[1, 2])
plot_denoise_spectrogram(temp, 'sensor_7', axes[2, 0])
plot_denoise_spectrogram(temp, 'sensor_8', axes[2, 1])
plot_denoise_spectrogram(temp, 'sensor_9', axes[2, 2])
fig, axes = plt.subplots(3, 3, figsize=(40, 15))
    
temp = pd.read_csv(f'{_train_dir}1826701813.csv')
plot_denoise_spectrogram(temp, 'sensor_1', axes[0, 0])
plot_denoise_spectrogram(temp, 'sensor_2', axes[0, 1])
plot_denoise_spectrogram(temp, 'sensor_3', axes[0, 2])
plot_denoise_spectrogram(temp, 'sensor_4', axes[1, 0])
plot_denoise_spectrogram(temp, 'sensor_5', axes[1, 1])
plot_denoise_spectrogram(temp, 'sensor_6', axes[1, 2])
plot_denoise_spectrogram(temp, 'sensor_7', axes[2, 0])
plot_denoise_spectrogram(temp, 'sensor_8', axes[2, 1])
plot_denoise_spectrogram(temp, 'sensor_9', axes[2, 2])
def spectrogram_data(temp, sensor):
    # Denoise Spectrogram
    fig = plt.figure(figsize=(40, 35))
    f, t, Sxx = spectrogram(denoise_transform(temp.copy())[sensor])
    plt.pcolormesh(t, f, np.log10(Sxx), shading='nearest')
    plt.title(f'{sensor} Denoised Spectrogram')
    plt.ylabel('f [Hz]')
    plt.xlabel('t [centi-seconds]')
    plt.yscale('symlog')
    plt.close()
    return fig


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

    
def get_img_from_fig(fig, dpi=0):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=None)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    return img[305:2200][:,360:2500]


def get_sensor_spectrograms(signals):
    spectrograms = []
    for i in range(1, 11):
        sensor_col = f'sensor_{i}'
        fig = spectrogram_data(signals, sensor_col)
        spectrogram = get_img_from_fig(fig)
        spectrograms.append(spectrogram)
        
    return np.array(spectrograms)
    
grams = get_sensor_spectrograms(temp)
fig, axes = plt.subplots(1, 3, figsize=(40, 5))

axes[0].imshow(grams[0], aspect='auto')
axes[1].imshow(grams[1], aspect='auto')
axes[2].imshow(grams[2], aspect='auto')
60000/10/10/10
class ConvNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(10, 15, kernel_size=4, padding=1)
        self.act1 =nn.Tanh()
        self.pool1 = nn.MaxPool1d(10, ceil_mode=True)
        self.conv2 = nn.Conv1d(15, 5, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool1d(10, ceil_mode=True)
        self.conv3 = nn.Conv1d(5, 1, kernel_size=3, padding=1)
        self.act3 = nn.Tanh()
        self.pool3 = nn.MaxPool1d(10, ceil_mode=True)
        
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.fc1 = nn.Linear(60, 32)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = self.pool3(self.act3(self.conv3(out)))
        out = self.flatten2(self.flatten1(out))
        #print(out.shape)
        out = self.act4(self.fc1(out))
        out = self.fc2(out)
        return out

model = ConvNN()
loss_fn = nn.L1Loss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
import torch.nn.functional as F

class TrainDataset(Dataset):
    
    def __init__(self, segment_ids):
        self.segment_ids = segment_ids
        
    def __getitem__(self, index):
        segment_id = self.segment_ids[index]
        signals = pd.read_csv(f'{_train_dir}{segment_id}.csv')
        grams = denoise_transform(signals).to_numpy()
        del signals
        for i in range(0, 10):
            grams[i] = grams[i] / 32767.0
        grams = torch.tensor(grams)
        grams = grams.permute(1, 0)
        ttf  = torch.tensor(train.set_index('segment_id').loc[segment_id]['time_to_eruption'])
        return (grams, ttf)
        
    def __len__(self):
        return len(self.segment_ids)
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    device = torch.device('cuda')
    model.to(device)
    
    for epoch in range(0, n_epochs):
        loss_train = 0.0
        
        i = 1
        for grams, ttf in train_loader:
            grams = grams.to(device).float()
            ttf = ttf.to(device).float()
            outputs = model(grams)
            loss = loss_fn(outputs, ttf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            del grams
            del ttf
            del outputs
            
            if i % 100 == 0:
                print(f'Epoch {epoch}, Training Loss {loss_train / i}')
            i += 1
            
        print(f'Epoch {epoch}, Training Loss {loss_train/i}')


trd = TrainDataset(list(train.segment_id[::-1]))
train_loader = DataLoader(
    trd,
    batch_size = 1,
    shuffle=True
)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    training_loop(15, optimizer, model, loss_fn, train_loader)
