# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/into-the-future/train.csv',parse_dates=True)

test=pd.read_csv('/kaggle/input/into-the-future/test.csv',parse_dates=True)
train['time']=pd.to_datetime(train['time'])
test['time']=pd.to_datetime(test['time'])
train
sns.scatterplot(data=train,x='feature_1',y='feature_2')
# We can see as our feature_1 increases feature_2 decreases
train.index=train['time']
train['feature_2'].plot()
train['feature_1'].plot()
train.min()
train.max()
test.min()
test.max()
# its basically 2.36 hr data
test['feature_1'].plot()
#feature_2 has upward trend and it shows non-stationarity,so differencing method is good to convert data into stationary

train_diff=train['feature_2']-train['feature_2'].shift(1)
# new column

train['diff']=train_diff
train.dropna(inplace=True)
# lets build up a Arima Model

# now we want to best combinations of p,d,q

import itertools

p=d=q=range(0,2)

#generate all different combinations of pdq

pdq=list(itertools.product(p,d,q))

# generate all seasonal combinations

seasonal_pdq=[(x[0],x[1],x[2],12) for x  in  pdq]

print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
#lets find best combination by AIC score



for pdq_param in pdq:

    for seasonal_param in seasonal_pdq:

        model=sm.tsa.statespace.SARIMAX(train['diff'],

                                            order=pdq_param,

                                            seasonal_order=seasonal_param,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

        results=model.fit()

        print('ARIMA {}x{} -AIC {}'.format(pdq_param,seasonal_param,results.aic))

       
#ARIMA (1, 1, 1)x(0, 1, 1, 12) -AIC 5894.786972338194

# Fitting an ARIMA time series model

mod=sm.tsa.statespace.SARIMAX(train['diff'],order=(1, 1, 1),seasonal_order=(0, 1, 1, 12),

                             enforce_stationarity=False,

                             enforce_invertibility=False)

results=mod.fit()

results.summary().tables[1]
results.plot_diagnostics(figsize=(15,12))

plt.show()
#forecasting for 26 minutes

train['forecast']=results.predict(start=pd.to_datetime('2019-03-19 01:10:00'),dynamic=False)

train[['diff','forecast']].plot()
#Chech RMSE Score

from sklearn.metrics import mean_squared_error

rmse=np.sqrt(mean_squared_error(train['diff']['2019-03-19 01:10:00':],train['forecast']['2019-03-19 01:10:00':]))

print('RMSE Score :',rmse)
#convert forecast value into real ones and see the result

#real value=previous value + commulative frequency of the forecasting values

train['forecast']=train['feature_2']['2019-03-19 01:10:00']+train['forecast'].cumsum()

train[['feature_2','forecast']].plot()
test.index=test['time']
#lets apply for test data

prediction=results.predict(start=pd.to_datetime('2019-03-19 01:34:00'),end=pd.to_datetime('2019-03-19 02:36:20'),dynamic=False)

test['feature_2']=(train['feature_2']['2019-03-19 01:33:50']+prediction.cumsum())
test
test.to_csv('Sagar_Badki.csv',index=False)
test.reset_index(drop=True,inplace=True)
test[['id','feature_2']].to_csv('Sagar_Badki.csv',index=False)