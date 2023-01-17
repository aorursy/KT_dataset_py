import numpy as np

import pandas as pd

from scipy.signal import butter, lfilter,filtfilt,savgol_filter

from scipy.stats import pearsonr 

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/liverpool-ion-switching/train.csv')

test = pd.read_csv('../input/liverpool-ion-switching/test.csv')

submission = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv', dtype={'time':str})

print('Train set has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

print('Test set has {} rows and {} columns'.format(test.shape[0], test.shape[1]))
train.shape[0]//500000 
bestcut = 0

bestscore = 0

for i in np.arange(9500,10000,1):

    x,y = butter(3,1./i,btype='highpass')

    batches = 10

    scores = np.zeros(batches)

    for c in range(batches):

        b1 = filtfilt(x, y, train.signal[c*500000:(c+1)*500000], padlen=20000)

        scores[c] = pearsonr(b1,train.open_channels[c*500000:(c+1)*500000])[0]

    mn = np.mean(scores)

    if(mn>bestscore):

        bestscore = mn

        bestcut = i

    
bestcut
bestscore
x,y = butter(3,1./bestcut,btype='highpass')

b1 = np.zeros(train.shape[0])

for c in range(10):

    b1[c*500000:(c+1)*500000] = filtfilt(x, y, train.signal[c*500000:(c+1)*500000], padlen=20000)



plt.figure(figsize=(15,15))

plt.scatter(train.time,b1,s=1)

plt.scatter(train.time,train.open_channels,s=1)