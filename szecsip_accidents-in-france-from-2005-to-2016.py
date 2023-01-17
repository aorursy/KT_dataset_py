# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime  

import statsmodels.api as sm  
import missingno as msno

import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import seaborn as sns
from collections import OrderedDict
sns.set()
import bokeh

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
caracteristics = pd.read_csv('../input/caracteristics.csv', encoding = 'latin-1')
caracteristics.describe()
caracteristics["hour"] = (caracteristics.hrmn - caracteristics.hrmn%100)/100
caracteristics["min"] = caracteristics.hrmn - caracteristics.hour*100;
caracteristics.an = caracteristics.an + 2000
caracteristics['date']= caracteristics.apply(lambda row :
                          datetime.date(row.an,row.mois,row.jour), 
                          axis=1)
caracteristics['date'] = pd.to_datetime(caracteristics['date'])
caracteristics['day_of_week'] = caracteristics['date'].dt.dayofweek
caracteristics['day_of_year'] = caracteristics['date'].dt.dayofyear
caracteristics.head()
msno.matrix(caracteristics)
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

ax = sns.distplot(caracteristics.an,bins=12,kde=False);
plt.title('Number of accidents in years')
plt.xlabel('years')
plt.ylabel('Number of accidents')
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

ax = sns.distplot(caracteristics.day_of_year,bins=365,kde=False);
plt.title('Number of accidents in days')
plt.xlabel('days in the year')
plt.ylabel('Number of accidents')
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

ax = sns.distplot(caracteristics.mois,bins=12,kde=False);
plt.title('Number of accidents in months')
plt.xlabel('months')
plt.ylabel('Number of accidents')
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

ax = sns.distplot(caracteristics.day_of_week,bins=7,kde=False);
plt.title('Number of accidents in weekdays')
plt.xlabel('weekdays')
plt.ylabel('Number of accidents')
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

ax = sns.distplot(caracteristics.hour,bins=24,kde=False);
plt.title('Number of accidents in hours')
plt.xlabel('hours')
plt.ylabel('Number of accidents')
import matplotlib.dates as mdates
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

caracteristics.index = caracteristics['date'] 
day_resample = caracteristics.resample('D').count()

day_resample.head()

sns.tsplot(data=day_resample.Num_Acc, time=day_resample.index)
ax = plt.gca()
# get current xtick labels
xticks = ax.get_xticks()
# convert all xtick labels to selected format from ms timestamp
ax.set_xticklabels([pd.to_datetime(tm).strftime('%Y-%m-%d\n') for tm in xticks],rotation=50)

plt.title('Number of accidents in a day')
plt.xlabel('date')
plt.ylabel('Number of accidents')
plt.show()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
test_stationarity(day_resample.Num_Acc)
full_df_ts = day_resample.Num_Acc
full_df_ts.index = day_resample.index
full_df_ts.loc[full_df_ts.index >= '2006']
res = sm.tsa.seasonal_decompose(full_df_ts, freq=365, model='additive')

plt.rcParams["figure.figsize"] = (20,10)
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(20,10))
res.trend.plot(ax=ax1)
res.resid.plot(ax=ax3)
res.seasonal.plot(ax=ax2)