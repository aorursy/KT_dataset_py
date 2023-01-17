import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
train_sig = pd.read_parquet('../input/train.parquet', columns=[str(i) for i in range(3)])
train_sig.head()
train_sig.shape
fig, ax = plt.subplots(figsize=(20,15))
for i in range(3):
    sns.lineplot(train_sig.index, train_sig[str(i)])
train_sig['0'][550000:].values
train_sig['0'][:550000].values
train_sig['0']=np.concatenate([train_sig['0'][525000:].values, train_sig['0'][:525000].values])
train_sig['2']=np.concatenate([train_sig['2'][275000:].values, train_sig['2'][:275000].values])
fig, ax = plt.subplots(figsize=(20,15))
for i in range(3):
    sns.lineplot(train_sig[:800000].index, train_sig[str(i)][:800000])
from scipy.signal import butter, lfilter
cutoff=50
measurements=800000
time=0.02
sampling_rate = measurements/time
sampling_rate
cutoff/(sampling_rate*0.5)
nyquist = sampling_rate*0.5
wn = cutoff/nyquist
b, a = butter(3, wn, btype='lowpass')

filtered_sig = lfilter(b, a, train_sig['0'].values)
filtered_sig.shape
filtered_sig
train_sig['3'] = filtered_sig
train_sig.head(20)
fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['3'][:800000])
train_sig['4'] = train_sig['0'] - train_sig['3']
train_sig.head()
fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['4'][:800000])
b, a = butter(3, wn, btype='highpass')

filtered_sig = lfilter(b, a, train_sig['0'].values)
filtered_sig.shape
filtered_sig
train_sig['5'] = filtered_sig
train_sig.head(20)
fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['5'][:800000])
b, a = butter(3, 5000/nyquist, btype='highpass')

filtered_sig = lfilter(b, a, train_sig['0'].values)
filtered_sig.shape
filtered_sig
train_sig['6'] = filtered_sig
train_sig.head(20)
fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['6'][:800000])
b, a = butter(3, 2500/nyquist, btype='highpass')

filtered_sig = lfilter(b, a, train_sig['0'].values)
filtered_sig.shape
filtered_sig
train_sig['7'] = filtered_sig
train_sig.head(20)
fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['7'][:800000])
b, a = butter(3, 1000/nyquist, btype='highpass')

filtered_sig = lfilter(b, a, train_sig['0'].values)
filtered_sig.shape
filtered_sig
train_sig['8'] = filtered_sig
train_sig.head(20)
fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(train_sig[:800000].index, train_sig['8'][:800000])