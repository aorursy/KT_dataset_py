# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
CGMDatenum_filepath = "../input/continuous-blood-glucose-monitor-data/CGMDatenumLunchPat1.csv"

CGMSeries_filepath = "../input/continuous-blood-glucose-monitor-data/CGMSeriesLunchPat1.csv"



CGMDatenumDf = pd.read_csv(CGMDatenum_filepath)

CGMSeriesDf = pd.read_csv(CGMSeries_filepath)



CGMSeriesList = CGMSeriesDf.values.tolist()

CGMDatenumList = CGMDatenumDf.values.tolist()





# CGM Velocity for day 20

CGMVel = np.subtract(CGMSeriesList[19][:-2], CGMSeriesList[19][1:-1])

print(np.flip(CGMVel))



plt.plot(CGMTimestampDf.iloc[19][1:-1],np.flip(CGMVel))

CGMSeriesDf.shape

CGMDatenumDf.shape



CGMSeriesDf.head()

CGMDatenumDf.head()



# function to convert datenum -> timestamp

def toDatetime(ts):

    timestamp = pd.to_datetime(ts-719529, unit='D').round('1s')

    return timestamp



#convert CGMDatenumDf to timestamp format Df

CGMTimestampDf = CGMDatenumDf.applymap(lambda i : toDatetime(i))



CGMTimestampDf.shape

CGMTimestampDf.head()
plt.figure(figsize=(12,6))



plt.plot_date(CGMTimestampDf.iloc[19], CGMSeriesDf.iloc[19])

#Day 2 series

#plt.plot_date(CGMTimestampDf[:2], CGMSeriesDf.iloc[:2,:]) 

plt.show()

CGMDay1 = pd.DataFrame(CGMSeriesList[0])



#construct a new dataset with our new columns

CGMDay1_lag = pd.concat([CGMDay1.shift(1),CGMDay1], axis=1)

CGMDay1_lag.columns = ['t-1','t+1']

print(CGMDay1_lag.head(5))



plt.plot(CGMDay1_lag)
CGMDay1 = pd.DataFrame(CGMSeriesList[19])

width = 2

shifted = CGMDay1.shift(width-1)

window = shifted.rolling(window=width)

#construct a new dataset with our new columns

CGMDay1_lag = pd.concat([window.min(),window.mean(),window.max(),window.max()-window.min()], axis=1)

CGMDay1_lag.columns = ['min', 'mean', 'max', 'max-min']

print(CGMDay1_lag.head(5))



plt.plot(CGMDay1_lag)
from statsmodels.graphics.tsaplots import plot_acf



#print(CGMSeriesDf.T[0])

plot_acf(CGMSeriesDf.T[0])

plt.show()
from statsmodels.graphics.tsaplots import plot_pacf



plot_pacf(CGMSeriesDf.T[0])

plt.show()
windowSize = 10

startSample = 1

endSample = startSample + windowSize -1



k=1

while endSample<len(CGMSeriesList[19]) and k<=6:

    meanCGM = np.mean(CGMSeriesList[19][startSample:endSample])

    startSample = startSample + windowSize/2

    endSample = startSample + windowSize

    k+=1

    print(startSample, endSample)