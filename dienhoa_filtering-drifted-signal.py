import numpy as np 

import pandas as pd 

from pathlib import Path

import matplotlib.pyplot as plt

from scipy import signal

from scipy.fft import fftshift
train_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

sample_sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
train_df.head()
plt.figure(figsize=(20,5)); res=1000

plt.plot(train_df.time[::res], train_df.signal[::res])

plt.ylabel('signal')

plt.xlabel('time')

plt.show()
sos = signal.butter(5, 10, 'hp', fs=10000, output='sos')

filtered = signal.sosfilt(sos, train_df.signal)
plt.figure(figsize=(20,5)); res=1000

plt.plot(train_df.time[::res], filtered[::res])