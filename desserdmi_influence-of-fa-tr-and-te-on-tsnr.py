import pandas as pd

import seaborn as sns

import pylab as plt

df = pd.read_csv('../input/mriqc-data-cleaning/bold.csv')
df
from json import load

import urllib.request, json 

from pandas.io.json import json_normalize

%matplotlib inline

import pylab as plt
dataset = pd.read_csv('../input/mriqc-data-cleaning/bold.csv',

                        usecols=[

                                'bids_meta.Manufacturer',

                                'bids_meta.MultibandAccelerationFactor',

                                'bids_meta.RepetitionTime',

                                'bids_meta.FlipAngle',

                                'bids_meta.EchoTime',

                                'tsnr'])

dataset.describe()
data = dataset.round(2)
plt.figure(figsize=(20,10))

sns.stripplot(x='bids_meta.FlipAngle', y='tsnr', data=data,

              jitter=0.4, alpha=0.3, size=10)

plt.ylim(0, 100)

plt.xlim(0, None)
plt.figure(figsize=(100,10))

sns.stripplot(x='bids_meta.RepetitionTime', y='tsnr', 

              data=data,  

              jitter=0.4, alpha=0.3, size=10)

plt.ylim(0, 100)

plt.xlim(0.5, None)

plt.show
plt.figure(figsize=(100,10))

sns.stripplot(x='bids_meta.EchoTime', y='tsnr', 

              data=data,  

              jitter=0.4, alpha=0.3, size=10)

plt.ylim(0, 100)

plt.show