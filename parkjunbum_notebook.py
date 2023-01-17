# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

sample = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

test.describe()

sample.describe()

train.describe()

train.tail()
import gc

import time

import math

from numba import jit

from math import log, floor



import numpy as np

import pandas as pd

from pathlib import Path



import seaborn as sns

from matplotlib import colors

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import pywt

from statsmodels.robust import mad



import scipy

from scipy import signal

from scipy.signal import butter, deconvolve



SAMPLE_RATE=25

SIGNAL_LEN=1000
plt.figure(figsize=(20, 10))

plt.plot(train['time'],train['signal'],color='r')

plt.title('signal data', fontsize=20)

plt.xlabel('time', fontsize=12)

plt.ylabel('signal', fontsize=12)

plt.show()
fig = make_subplots(rows=3,cols=1)

x_1=train.loc[:100]['time']

y_1=train.loc[:100]['signal']

x_2=train.loc[100:200]['time']

y_2=train.loc[100:200]['signal']

x_3=train.loc[200:300]['time']

y_3=train.loc[200:300]['signal']

fig.add_trace(go.Scatter(x=x_1,y=y_1,showlegend=False,mode='lines+markers',name='first sample',marker=dict(color='dodgerblue')),row=1,col=1)

fig.add_trace(go.Scatter(x=x_2,y=y_2,showlegend=False,mode='lines+markers',name='second sample',marker=dict(color='mediumseagreen')),row=1,col=1)

fig.add_trace(go.Scatter(x=x_3,y=y_3,showlegend=False,mode='lines+markers',name='third sample',marker=dict(color='violet')),row=1,col=1)

fig.update_layout(height=1200,width=800,title_text='sample signals')

fig.show()
def maddest(d, axis=None):

    return np.mean(np.absolute(d - np.mean(d,axis)),axis)

def denoise_signal(x,wavelet='db4',level=1):

    coeff=pywt.wavedec(x,wavelet,mode='per')

    sigma=(1/0.6745)*maddest(coeff[-level])

    uthresh=sigma*np.sqrt(2*np.log(len(x)))

    coeff[1:]=(pywt.threshold(i,value=uthresh,mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff,wavelet,mode='per')
fig = make_subplots(rows=3,cols=1)

x=train.loc[:100]['time']

y_1=train.loc[:100]['signal']

y_w1=denoise_signal(train.loc[:100]['signal'])

y_2=train.loc[100:200]['signal']

y_w2=denoise_signal(train.loc[100:200]['signal'])

y_3=train.loc[200:300]['signal']

y_w3=denoise_signal(train.loc[200:300]['signal'])

fig.add_trace(go.Scatter(x=x,y=y_1,showlegend=False,mode='lines+markers',name='original sample',marker=dict(color='lightskyblue')),row=1,col=1)

fig.add_trace(go.Scatter(x=x,y=y_w1,showlegend=False,mode='lines',name='denoised sample',marker=dict(color='navy')),row=1,col=1)

fig.add_trace(go.Scatter(x=x,y=y_2,showlegend=False,mode='lines+markers',marker=dict(color='mediumaquamarine')),row=2,col=1)

fig.add_trace(go.Scatter(x=x,y=y_w2,showlegend=False,mode='lines',marker=dict(color='mediumaquamarine')),row=2,col=1)

fig.add_trace(go.Scatter(x=x,y=y_3,showlegend=False,mode='lines+markers',marker=dict(color='thistle')),row=3,col=1)

fig.add_trace(go.Scatter(x=x,y=y_w3,showlegend=False,mode='lines',marker=dict(color='indigo')),row=3,col=1)

fig.update_layout(height=1200,width=800,title_text='Original (pale) vs. denoised (dark) signals')

fig.show()
output = pd.DataFrame({'time': test.time, 'open_channels': sample.open_channels})

output.to_csv('submission.csv', index=False)

print('submission is saved')
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()
x = train.loc[:100]["time"]

y1 = train.loc[:100]["signal"]

y_w1 = denoise_signal(train.loc[:100]["signal"])

y2 = train.loc[100:200]["signal"]

y_w2 = denoise_signal(train.loc[100:200]["signal"])

y3 = train.loc[200:300]["signal"]

y_w3 = denoise_signal(train.loc[200:300]["signal"])



fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(30, 20))



ax[0,0].plot(y1, color='seagreen', marker='o')

ax[0,0].set_title('Original Signal', fontsize=24)

ax[0,1].plot(y_w1, color='red', marker='.')

ax[0,1].set_title('After Wavelet Denoising', fontsize=24)



ax[1,0].plot(y2, color='seagreen', marker='o')

ax[1,0].set_title('Original Signal', fontsize=24)

ax[1,1].plot(y_w2, color='red', marker='.')

ax[1,1].set_title('After Wavelet Denoising', fontsize=24)



ax[2,0].plot(y3, color='seagreen', marker='o')

ax[2,0].set_title('Original Signal', fontsize=24)

ax[2,1].plot(y_w3, color='red', marker='.')

ax[2,1].set_title('After Wavelet Denoising', fontsize=24)



plt.show()