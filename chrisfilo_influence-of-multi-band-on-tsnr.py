import pandas as pd
from json import load
import urllib.request, json 
from pandas.io.json import json_normalize
import seaborn as sns
import pylab as plt
%matplotlib inline
df = pd.read_csv('../input/mriqc-data-cleaning/bold.csv')
df.describe()
plt.figure(figsize=(10,14))
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='tsnr', data=df, jitter=0.4, alpha=0.3, size=4)
plt.figure(figsize=(10,14))
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='tsnr', data=df[df['tsnr']<100], jitter=0.4, alpha=0.3, size=4)