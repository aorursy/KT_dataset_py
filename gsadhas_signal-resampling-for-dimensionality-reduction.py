import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_parquet('../input/train.parquet')
print(df_train.shape)
train_meta = pd.read_csv('../input/metadata_train.csv')
print(train_meta.shape)
print(train_meta.head(3))
train_meta_pos = train_meta[train_meta['target'] == 1]
train_meta_neg = train_meta[train_meta['target'] == 0]
print(train_meta_pos.shape)
print(train_meta_neg.shape)
train_meta_pos_p0 = train_meta_pos[train_meta_pos['phase'] == 0]
train_meta_pos_p0.head(10)
# Negative signal
train_meta_neg_p0 = train_meta_neg[train_meta_neg['phase'] == 0]
train_meta_neg_p0.head(10)
# Take some samples that has postive and negative target
df_train_sample = df_train.iloc[:, 201:270]
print(df_train_sample.shape)
df_train_sample.head(5)
# Plot the given data points to see how its trend looks like
df_train_sample.iloc[:, :3].plot()
# Function to to FFT and reduce the dimensions
def sample_signals(df):
    cols = df.columns.values
    b, a = signal.butter(4, 0.03, analog=False)
    df_t = []
    for idx in range(df.shape[1]):
        sg = np.squeeze(df.iloc[:, idx:idx+1], axis=1)
        sg_ff = signal.filtfilt(b, a, sg)
        sg_rs = signal.resample(sg_ff, 16*2*50)
        df_t.append(sg_rs)
    df_t = np.asarray(df_t).T
    df_t = pd.DataFrame(df_t, columns=cols)
    return df_t
# Show how the processed signals look like
sample1 = sample_signals(df_train_sample.iloc[:,:3])
print(sample1.shape)
sample1.plot()
# Comparing given signals and the processed one - Positive signal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
ax1.plot(df_train_sample.iloc[:, :3])
ax2.plot(sample1)
# Negative signal - target = 0
sample2 = sample_signals(df_train_sample.iloc[:,3:6])
df_train_sample.iloc[:,3:6].plot()
# Comparing given signals and the processed one - Negative signal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
ax1.plot(df_train_sample.iloc[:, 3:6])
ax2.plot(sample2)
