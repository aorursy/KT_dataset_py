# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv')

df.head()
df.shape
df.tail()
df['Month'] = pd.to_datetime(df['Month'])

df = df.set_index('Month')
from datetime import datetime

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10,6

plt.plot(df)
rolmean = df.rolling(window = 12).mean()

rolstd = df.rolling(window = 12).std()



print(rolmean,rolstd)
orig = plt.plot(df,color = 'blue', label = 'original')

mean = plt.plot(rolmean, color = 'red', label = 'rolling mean')

std = plt.plot(rolstd, color = 'black', label = 'rolling std')

plt.legend(loc = 'best')

plt.show(block = False)
from statsmodels.tsa.stattools import adfuller



print("Result of Dickey Fuller Test")



dftest = adfuller(df['#Passengers'], autolag = 'AIC') # AIC is a metric



dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', '#lags used', 'No. of observation used'])

for key, value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

    

print(dfoutput)
logscale = np.log(df)

plt.plot(logscale)
rolmean_log = logscale.rolling(window = 12).mean()

rolstd_log = logscale.rolling(window = 12).std()

plt.plot(logscale)

plt.plot(rolmean_log)