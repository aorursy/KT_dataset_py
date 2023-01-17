import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

import numpy as np

import pandas as pd

from scipy import signal

import scipy.io.wavfile as wavfile



import torch

torch.set_printoptions(edgeitems=2, threshold=50, linewidth=75)
freq, waveform_arr = wavfile.read('/kaggle/input/1-100038-A-14.wav')

freq, waveform_arr
waveform = torch.from_numpy(waveform_arr).float()

waveform.shape
f_arr, t_arr, sp_arr = signal.spectrogram(waveform_arr, freq)



sp_mono = torch.from_numpy(sp_arr)

sp_mono.shape
sp_left = sp_right = sp_arr

sp_left_t = torch.from_numpy(sp_left)

sp_right_t = torch.from_numpy(sp_right)

sp_left_t.shape, sp_right_t.shape
sp_t = torch.stack((sp_left_t, sp_right_t), dim=0)

sp_t.shape