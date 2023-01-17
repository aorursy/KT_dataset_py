# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pyarrow.parquet as pq #data reading

import os

import matplotlib.pyplot as plt # visulation



#Efficient Implementation for moving average

def runningMeanFast(x, N):

    return np.convolve(x, np.ones((N,))/N)[(N-1):]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.



#In the power signals there are lot of verticals lines. These flucuations are seen in non faulty and faulty lines.

#Let us examine one of these signals. 

#The signal 2576 have lot of these patterns and till not faulty.

column = str((3*2576)+0)

signalOrg = pd.read_parquet('../input/train.parquet', engine='pyarrow', columns=[column]) 

plt.figure(figsize=(24, 10))

plt.plot(signalOrg,'g')

plt.show()
#let us zoom in for vertical line around 450,000

plt.figure(figsize=(24, 10))

plt.plot(signalOrg[495000:497000],'g')

plt.show()
#It is clearly visible that the voltage is following exponential decay pattern.

#let us define a pattern of these behaviour. 

# These pattern has approx. 10 length.

pattern = [42,-34,27,-19,14,-10,7,-5,3,-2,1] 

pattern = pattern - np.mean(pattern)

img=plt.plot(np.asarray(pattern),'b')
#Convluing with pattern will help detect  presense of these pattern. Let us see how it performs.

#Before we convul, we need to get signal stationary. We substract moving average from signal from the same.

signal = pd.read_parquet('../input/train.parquet', engine='pyarrow', columns=[column]) 

signal = signal[column]



mvSignal = runningMeanFast(signal,30)

con = np.convolve(signal-mvSignal,pattern,mode ='same')

mv = runningMeanFast(abs(con),10)

mv = np.roll(mv,7)

plt.figure(figsize=(24, 10))

plt.plot(mv,'b+')

plt.show()
#Let us replace the variations with moving average

signal [mv > 200] = mvSignal[mv > 200]

plt.figure(figsize=(24, 10))

img=plt.plot(np.asarray(signalOrg),'r-')

img=plt.plot(np.asarray(signal),'g-')