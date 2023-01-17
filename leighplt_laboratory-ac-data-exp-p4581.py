import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import os.path as osp



import warnings

warnings.filterwarnings("ignore")
path = '../input/laboratory-acoustic-data-exp4581/p4581'
acDataStd = np.empty((0,))

acTime = np.empty((0,))

events = 297



for i in range(events):

    a = np.load(osp.join(path, f"earthquake_{i:03d}.npz"))['acoustic_data'] 

    t = np.load(osp.join(path, f"earthquake_{i:03d}.npz"))['ttf'] 

    acDataStd = np.hstack([acDataStd, a.std(axis=1)])

    acTime = np.hstack([acTime, t])
# smooth data

N = 1000

acDataStd = np.convolve(acDataStd, np.ones((N))/ N, mode='same')
fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title("Trends of acoustic_data and time")

plt.plot(acDataStd, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(acTime, color='g')

ax2.set_ylabel('time', color='g')

plt.legend(['time'], loc=(0.875, 0.9))

plt.grid(False)
event = 50

acData = np.load(osp.join(path, f"earthquake_{event:03d}.npz"))['acoustic_data'] 

acTime = np.load(osp.join(path, f"earthquake_{event:03d}.npz"))['ttf'] 
steps = np.arange(4096) * 0.252016890769332e-6

t = acTime[:, np.newaxis] + np.flip(steps)[np.newaxis]
fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title("Trends of acoustic_data and time")

plt.plot(acData.flatten(), color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(t.flatten(), color='g')

ax2.set_ylabel('time', color='g')

plt.legend(['time'], loc=(0.875, 0.9))

plt.grid(False)
event = 3

acData = np.load(osp.join(path, f"earthquake_{event:03d}.npz"))['acoustic_data'] 

acTime = np.load(osp.join(path, f"earthquake_{event:03d}.npz"))['ttf'] 
steps = np.arange(4096) * 0.252016890769332e-6

t = acTime[:, np.newaxis] + np.flip(steps)[np.newaxis]
fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title("Trends of acoustic_data and time")

plt.plot(acData.flatten(), color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(t.flatten(), color='g')

ax2.set_ylabel('time', color='g')

plt.legend(['time'], loc=(0.875, 0.9))

plt.grid(False)