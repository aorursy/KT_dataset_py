import keras

import keras.backend as K

from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten

from keras.models import Sequential

import tensorflow as tf

import pandas as pd

# import pyarrow as pa

import pyarrow.parquet as pq

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

#Both numpy and scipy has utilities for FFT which is an endlessly useful algorithm

from numpy.fft import *

from scipy import fftpack
signals = pq.read_table('../input/train.parquet', columns=[str(i) for i in range(999)]).to_pandas()

print('signals shape is: ',signals.shape)

#Since data size is big we just load one third of it for now

signals = np.array(signals).T.reshape((999//3, 3, 800000))

print('signals shape after reshaping is: ', signals.shape)

train_df = pd.read_csv('../input/metadata_train.csv')

print('metadata shape is: ',train_df.shape)
fig, axs = plt.subplots(2, 2, constrained_layout=True,figsize=(15, 10))



axs[0,0].set_title('normal wires')

axs[0,0].plot(signals[0, 0, :], label='Phase 0')

axs[0,0].plot(signals[0, 1, :], label='Phase 1')

axs[0,0].plot(signals[0, 2, :], label='Phase 2')

axs[0,0].legend()



axs[1,0].set_title('damaged wires')

axs[1,0].plot(signals[1, 0, :], label='Phase 0')

axs[1,0].plot(signals[1, 1, :], label='Phase 1')

axs[1,0].plot(signals[1, 2, :], label='Phase 2')

axs[1,0].legend()



axs[0,1].set_title('normal wires')

axs[0,1].plot(signals[77, 0, :], label='Phase 0')

axs[0,1].plot(signals[77, 1, :], label='Phase 1')

axs[0,1].plot(signals[77, 2, :], label='Phase 2')

axs[0,1].legend()



axs[1,1].set_title('damaged wires')

axs[1,1].plot(signals[76, 0, :], label='Phase 0')

axs[1,1].plot(signals[76, 1, :], label='Phase 1')

axs[1,1].plot(signals[76, 2, :], label='Phase 2')

axs[1,1].legend()





plt.show()
target = train_df['target']

plt.figure(figsize=(15, 10))

sns.countplot(target)

plt.show()



print('number of damaged samples', sum(target))

print('number of normal sampes', target.shape[0]-sum(target))
#FFT to filter out HF components and get main signal profile

def low_pass(s, threshold=1e4):

    fourier = rfft(s)

    frequencies = rfftfreq(s.size, d=2e-2/s.size)

    fourier[frequencies > threshold] = 0

    return irfft(fourier)
#normal one

lf_normal1_1 = low_pass(signals[0, 0, :])

lf_normal1_2 = low_pass(signals[0, 1, :])

lf_normal1_3 = low_pass(signals[0, 2, :])

#normal two

lf_normal2_1 = low_pass(signals[77, 0, :])

lf_normal2_2 = low_pass(signals[77, 1, :])

lf_normal2_3 = low_pass(signals[77, 2, :])

#damaged one

lf_damaged1_1 = low_pass(signals[1, 0, :])

lf_damaged1_2 = low_pass(signals[1, 1, :])

lf_damaged1_3 = low_pass(signals[1, 2, :])

#damaged two

lf_damaged2_1 = low_pass(signals[76, 0, :])

lf_damaged2_2 = low_pass(signals[76, 1, :])

lf_damaged2_3 = low_pass(signals[76, 2, :])
fig, axs = plt.subplots(2, 2, constrained_layout=True,figsize=(15, 10))



axs[0,0].set_title('normal wires')

axs[0,0].plot(lf_normal1_1, label='Phase 0')

axs[0,0].plot(lf_normal1_2, label='Phase 1')

axs[0,0].plot(lf_normal1_3, label='Phase 2')

axs[0,0].legend()



axs[1,0].set_title('damaged wires')

axs[1,0].plot(lf_damaged1_1, label='Phase 0')

axs[1,0].plot(lf_damaged1_2, label='Phase 1')

axs[1,0].plot(lf_damaged1_3, label='Phase 2')

axs[1,0].legend()



axs[0,1].set_title('normal wires')

axs[0,1].plot(lf_normal2_1, label='Phase 0')

axs[0,1].plot(lf_normal2_2, label='Phase 1')

axs[0,1].plot(lf_normal2_3, label='Phase 2')

axs[0,1].legend()



axs[1,1].set_title('damaged wires')

axs[1,1].plot(lf_damaged2_1, label='Phase 0')

axs[1,1].plot(lf_damaged2_2, label='Phase 1')

axs[1,1].plot(lf_damaged2_3, label='Phase 2')

axs[1,1].legend()





plt.show()
fig, axs = plt.subplots(2, 2, constrained_layout=True,figsize=(15, 10))



axs[0,0].set_title('normal wires')

axs[0,0].plot((np.abs(lf_normal1_1)+np.abs(lf_normal1_2)+np.abs(lf_normal1_3)))

axs[0,0].plot(lf_normal1_1, label='Phase 0')

axs[0,0].plot(lf_normal1_2, label='Phase 1')

axs[0,0].plot(lf_normal1_3, label='Phase 2')

axs[0,0].legend()



axs[1,0].set_title('damaged wires')

axs[1,0].plot((np.abs(lf_damaged1_1)+np.abs(lf_damaged1_2)+np.abs(lf_damaged1_3)))

axs[1,0].plot(lf_damaged1_1, label='Phase 0')

axs[1,0].plot(lf_damaged1_2, label='Phase 1')

axs[1,0].plot(lf_damaged1_3, label='Phase 2')

axs[1,0].legend()



axs[0,1].set_title('normal wires')

axs[0,1].plot((np.abs(lf_normal2_1)+np.abs(lf_normal2_2)+np.abs(lf_normal2_3)))

axs[0,1].plot(lf_normal2_1, label='Phase 0')

axs[0,1].plot(lf_normal2_2, label='Phase 1')

axs[0,1].plot(lf_normal2_3, label='Phase 2')

axs[0,1].legend()



axs[1,1].set_title('damaged wires')

axs[1,1].plot((np.abs(lf_damaged2_1)+np.abs(lf_damaged2_2)+np.abs(lf_damaged2_3)))

axs[1,1].plot(lf_damaged2_1, label='Phase 0')

axs[1,1].plot(lf_damaged2_2, label='Phase 1')

axs[1,1].plot(lf_damaged2_3, label='Phase 2')

axs[1,1].legend()





plt.show()
###Filter out low frequencies from the signal to get HF characteristics

def high_pass(s, threshold=1e7):

    fourier = rfft(s)

    frequencies = rfftfreq(s.size, d=2e-2/s.size)

    fourier[frequencies < threshold] = 0

    return irfft(fourier)
#normal one

hf_normal1_1 = high_pass(signals[0, 0, :])

hf_normal1_2 = high_pass(signals[0, 1, :])

hf_normal1_3 = high_pass(signals[0, 2, :])

#normal two

hf_normal2_1 = high_pass(signals[77, 0, :])

hf_normal2_2 = high_pass(signals[77, 1, :])

hf_normal2_3 = high_pass(signals[77, 2, :])

#damaged one

hf_damaged1_1 = high_pass(signals[1, 0, :])

hf_damaged1_2 = high_pass(signals[1, 1, :])

hf_damaged1_3 = high_pass(signals[1, 2, :])

#damaged two

hf_damaged2_1 = high_pass(signals[76, 0, :])

hf_damaged2_2 = high_pass(signals[76, 1, :])

hf_damaged2_3 = high_pass(signals[76, 2, :])
fig, axs = plt.subplots(2, 2, constrained_layout=True,figsize=(15, 10))



axs[0,0].set_title('normal wires')

axs[0,0].plot(hf_normal1_1, label='Phase 0')

# axs[0,0].plot(hf_normal1_2, label='Phase 1')

# axs[0,0].plot(hf_normal1_3, label='Phase 2')

axs[0,0].legend()



axs[1,0].set_title('damaged wires')

axs[1,0].plot(hf_damaged1_1, label='Phase 0')

axs[1,0].plot(hf_damaged1_2, label='Phase 1')

axs[1,0].plot(hf_damaged1_3, label='Phase 2')

axs[1,0].legend()



axs[0,1].set_title('normal wires')

axs[0,1].plot(hf_normal2_1, label='Phase 0')

axs[0,1].plot(hf_normal2_2, label='Phase 1')

axs[0,1].plot(hf_normal2_3, label='Phase 2')

axs[0,1].legend()



axs[1,1].set_title('damaged wires')

axs[1,1].plot(hf_damaged2_1, label='Phase 0')

axs[1,1].plot(hf_damaged2_2, label='Phase 1')

axs[1,1].plot(hf_damaged2_3, label='Phase 2')

axs[1,1].legend()





plt.show()