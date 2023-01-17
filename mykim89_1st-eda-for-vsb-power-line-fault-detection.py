import os
import gc
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from numba import jit, int32
plt.style.use("fast")

import warnings
warnings.filterwarnings("ignore") 

plt.style.use('seaborn')
sns.set(font_scale=1)
df_train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(3000)]).to_pandas()
# df_train = pq.read_pandas('../input/train.parquet').to_pandas()
print(df_train.info())
df_train.head()
df_train.isnull().sum().sum()
meta_train = pd.read_csv('../input/metadata_train.csv')
meta_test = pd.read_csv('../input/metadata_test.csv')
meta_train.head()
meta_train.describe()
meta_train.shape
meta_test.head()
meta_test.describe()
meta_test.shape
meta_train['target'].unique()
print(meta_train.isnull().sum())
print(meta_test.isnull().sum())
meta_train['target'].value_counts()
print('Normal sample number: {:d}'.format(meta_train['target'].value_counts()[0]))
print('Fault sample number: {:d}'.format(meta_train['target'].value_counts()[1]))
print('Normal sample ratio: {:.3f} %'.format(meta_train['target'].value_counts()[0]/len(meta_train)*100))
print('Fault sample ratio: {:.3f} %'.format(meta_train['target'].value_counts()[1]/len(meta_train)*100))
f, ax = plt.subplots(1, 3, figsize=(24, 8))

ax[0].pie(meta_train['target'].value_counts(), explode=[0, 0.1], autopct='%1.3f%%', shadow=True)
ax[0].set_title('Pie plot - Fault')
ax[0].set_ylabel('')

sns.countplot('target', data=meta_train, ax=ax[1])
ax[1].set_title('Count plot - Fault')

sns.countplot('target', data=meta_train, hue ='phase', ax=ax[2])
meta_train.head()
diff = meta_train.groupby(['id_measurement']).sum().query("target != 3 & target != 0")
print('not all fault or normal data number: {:d}'.format(diff.shape[0]))
diff
pd.crosstab(meta_train['phase'], meta_train['target'], margins=True)
meta_train[['phase', 'target']].groupby(['phase'], as_index=True).mean()
meta_train[['phase', 'target']].groupby(['phase'], as_index=True).mean().sort_values(by='target', ascending=False).plot.bar()
targets = meta_train.groupby('id_measurement', as_index=True)[['target','id_measurement']].mean()
targets.iloc[67,:]
targets.head()
sns.countplot(x='target',data=round(targets, 2))
meta_train['phase'].value_counts()
meta_train['phase'].value_counts().plot.bar()
meta_train.groupby('phase').count().plot.bar()
meta_train.corr()
plt.figure(figsize = (10,10))

colormap = plt.cm.summer_r
sns.heatmap(meta_train.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})
#np.unique(meta_train.loc[meta_train['target']==1, 'id_measurement'].values)
# Fault measurement ID
fault_mid = meta_train.loc[meta_train['target']==1, 'id_measurement'].unique()
# Normal measurement ID
normal_mid = meta_train.loc[meta_train['target']==0, 'id_measurement'].unique()
print(fault_mid.shape)
print(normal_mid.shape)
print('8712//3={:d}'.format(8712//3))
print('2748+194-38={:d} (38 overlapped samples (some phases are normal, the others are fault))'.format(2748+194-38))
print(meta_train.signal_id.unique())
print(meta_train.id_measurement.unique())
meta_train.head(10)
fault_mid[1]
fault_sid = meta_train.loc[meta_train.id_measurement == fault_mid[0], 'signal_id']
normal_sid = meta_train.loc[meta_train.id_measurement == normal_mid[0], 'signal_id']
print(fault_sid)
print(normal_sid)
fault_sample = df_train.iloc[:, fault_sid]
normal_sample = df_train.iloc[:, normal_sid]
plt.figure(figsize=(24, 8))
plt.plot(normal_sample, alpha=0.7);
plt.ylim([-100, 100])
plt.figure(figsize=(24, 8))
plt.plot(fault_sample, alpha=0.8);
plt.ylim([-100, 100])
@jit('float32(float32[:,:], int32, int32)')
def flatiron(x, alpha=100., beta=1):
    new_x = np.zeros_like(x)
    zero = x[0]
    for i in range(1, len(x)):
        zero = zero*(alpha-beta)/alpha + beta*x[i]/alpha
        new_x[i] =  x[i] - zero
    return new_x
fault_sample_filt = flatiron(fault_sample.values)
normal_sample_filt = flatiron(normal_sample.values)
f, ax = plt.subplots(2, 2, figsize=(24, 16))

ax[0, 0].plot(fault_sample, alpha=0.8)
ax[0, 0].set_title('fault signal')
ax[0, 0].set_ylim([-100, 100])

ax[0, 1].plot(fault_sample_filt, alpha=0.5)
ax[0, 1].set_title('filtered fault signal')
ax[0, 1].set_ylim([-100, 100])

ax[1, 0].plot(normal_sample, alpha=0.7)
ax[1, 0].set_title('normal signal')
ax[1, 0].set_ylim([-100, 100])

ax[1, 1].plot(normal_sample_filt, alpha=0.5)
ax[1, 1].set_title('filtered normal signal')
ax[1, 1].set_ylim([-100, 100])
