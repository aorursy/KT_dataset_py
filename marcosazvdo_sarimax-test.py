import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

from datetime import datetime

from pylab import rcParams

import statsmodels.api as sm

import itertools

from statsmodels.tsa.arima_model import ARIMA

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.stattools import adfuller



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df_train = pd.read_csv("../input/train.csv", parse_dates=['date'], index_col=['date'])

df_test = pd.read_csv("../input/test.csv", parse_dates=['date'], index_col=['date'])

df = pd.concat([df_train,df_test],sort=True)
df.head()
df.dtypes
df['store'].value_counts()
df['sales'].count()
df.isnull().sum()
df = df.fillna(0)
df.sales.plot(figsize=(15, 6))

plt.show()
lojas = len(df['store'].unique())

fig, axes = plt.subplots(lojas, figsize=(8, 16))



for x in df['store'].unique():

    m = df.loc[df['store'] == x, 'sales'].resample('W').sum()

    ax = m.plot(title = x, ax=axes[x-1])

    ax.grid()   

    ax.set_xlabel('time')

    ax.set_ylabel('sales')

fig.tight_layout();
one_store = df[(df.item==1)&(df.store==1)].copy()

rcParams['figure.figsize'] = 20, 10

decomposition = sm.tsa.seasonal_decompose(one_store.sales.dropna(), freq=365)

fig = decomposition.plot()

plt.show()
def test_stationarity(timeseries):

    

 # rolling statistics

    rolmean = timeseries.rolling(12).mean()

    rolstd = timeseries.rolling(12).std()



 #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False) 
test_stationarity(one_store['sales'])
one_store['log_sales'] = np.log(one_store['sales'])

plt.plot(one_store['log_sales'])
first_one = df[(df.item==1)&(df.store==1)].copy()
first_one['month'] = first_one.index.month
month_dum = pd.get_dummies(first_one['month'])
month_dum.columns = ['month_dum-'+ str(w) for w in range(0,12)]
first_one = pd.concat([first_one, month_dum], axis=1)
first_one['dayofweek_dum'] = first_one.index.weekday
week_dummies = pd.get_dummies(first_one['dayofweek_dum'])
week_dummies.columns = ['dayofweek_dum-'+ str(w) for w in range(0,7)]
first_one = pd.concat([first_one, week_dummies], axis=1, join_axes=[first_one.index]).drop(['dayofweek_dum'],axis=1)
first_one['weekend'] = (first_one.index.dayofweek>4).astype(int)
data_r = pd.date_range(start='2013-01-01', end='2018-03-31')

cal = calendar()

holidays = cal.holidays(start=data_r.min(), end=data_r.max())

first_one['holyday'] = first_one.index.isin(holidays)

first_one['holyday'] = first_one['holyday']*1
train_start,train_end = '2015-01-01','2017-09-30'

test_start,test_end = '2017-10-01','2017-12-31'

train = first_one['sales'][train_start:train_end].dropna()

test =first_one['sales'][test_start:test_end].dropna()

ex_train = first_one.drop(['id','store','item','sales'],axis = 1)[train_start:train_end].dropna()

ex_test = first_one.drop(['id','store','item','sales'],axis = 1)[test_start:test_end].dropna()
train_sample = train.sample(frac=0.3, replace=True)
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

minimo=[]

minimo1=[]

minimo2=[]

for param in pdq:

     for param_seasonal in seasonal_pdq:

            try:

                mod1 = sm.tsa.statespace.SARIMAX(train_sample,

                                                order=param,

                                                seasonal_order=param_seasonal, 

                                                enforce_stationarity=False,

                                                enforce_invertibility=False)

                results1 = mod1.fit()

                print('ARIMA{}x{}7 - AIC:{}'.format(param, param_seasonal, results1.aic))

            except:

                continue

                

                          
results1= sm.tsa.statespace.SARIMAX(train,

                               order=(0,0,1),

                               seasonal_order=(0,1,1,7),

                               exog = ex_train,

                               freq='D',

                               enforce_stationarity=False, 

                               enforce_invertibility=False).fit()



print(results1.summary())
pred = results1.predict(train_end,test_end,exog = ex_test)[1:]

print('ARIMAX model MSE:{}'.format(mean_squared_error(test,pred)))
pd.DataFrame({'test':test,'pred':pred}).plot();plt.show()
results1.plot_diagnostics(figsize=(15, 12))
searching_best = sm.tsa.arma_order_select_ic(train_sample, max_ar=7, max_ma=7, ic='aic', trend='c')

print('The bestpdq - ARMA(p,q) =',searching_best['aic_min_order'])
results2= sm.tsa.statespace.SARIMAX(train,

                               order=(0,0,1),

                               seasonal_order=(0,0,0,0),

                               exog = ex_train,

                               freq='D',

                               enforce_stationarity=False, 

                               enforce_invertibility=False).fit()



print(results2.summary())
pred = results2.predict(train_end,test_end,exog = ex_test)[1:]

print('ARIMAX model MSE:{}'.format(mean_squared_error(test,pred)))
pd.DataFrame({'test':test,'pred':pred}).plot();plt.show()
results2.plot_diagnostics(figsize=(15, 12))
#loading data

dfs_train = pd.read_csv("../input/train.csv", parse_dates=['date'], index_col=['date'])

dfs_test = pd.read_csv("../input/test.csv", parse_dates=['date'], index_col=['date'])

dfs = pd.concat([dfs_train,dfs_test],sort=True)
#loading data submission

subm = pd.read_csv('../input/sample_submission.csv')
#changing nan

dfs = dfs.fillna(0)
#creating dummies

dfs['month'] = dfs.index.month

month_dum = pd.get_dummies(dfs['month'])

month_dum.columns = ['month_dum-'+ str(w) for w in range(0,12)]

dfs = pd.concat([dfs, month_dum], axis=1)

dfs['dayofweek_dum'] = dfs.index.weekday

week_dummies = pd.get_dummies(dfs['dayofweek_dum'])

week_dummies.columns = ['dayofweek_dum-'+ str(w) for w in range(0,7)]

dfs = pd.concat([dfs, week_dummies], axis=1, join_axes=[dfs.index]).drop(['dayofweek_dum'],axis=1)

dfs['weekend'] = (dfs.index.dayofweek>4).astype(int)
data_r = pd.date_range(start='2013-01-01', end='2018-03-31')

first_one['holyday'] = data_r

cal = calendar()

holidays = cal.holidays(start=data_r.min(), end=data_r.max())

first_one['holyday'] = first_one.index.isin(holidays)
#creating dummies holidays

date_r = pd.date_range(start='2013-01-01', end='2018-03-31')

#dfs['holyday'] = date_r

cal = calendar()

holidays = cal.holidays(start=date_r.min(), end=date_r.max())

dfs['holyday'] = dfs.index.isin(holidays)

dfs['holyday'] = dfs['holyday']*1
#the prediction

results = []



for w in range(1,51):

    for m in range(1,11):

        sales1 = dfs[(dfs.item==w)&(dfs.store==m)].copy()

        train_start,train_end = '2015-01-01','2017-09-30'

        test_start,test_end = '2017-10-01','2017-12-31'

        train = sales1['sales'][train_start:train_end]

        test =  sales1['sales'][test_start:test_end]

        ex_train = sales1.drop(['id','store','item','sales'],axis = 1)[train_start:train_end]

        ex_test = sales1.drop(['id','store','item','sales'],axis = 1)[test_start:test_end]

        target_exog = sales1[test_start:].drop(['id','store','item','sales'],axis = 1) 

        predict_mod = sm.tsa.statespace.SARIMAX(train,

                                                order=(0,0,3),

                                                seasonal_order=(0,1,1,7),

                                                exog = ex_train,

                                                freq='D',

                                                enforce_stationarity=False,

                                                enforce_invertibility=False).fit()

        predict_train = predict_mod.get_prediction(train_end,'2018-03-31', exog = target_exog)

        results.extend(predict_train.predicted_mean['2018-01-01':])

        print('item:',w,'store:',m,'Predicted.')      

        
subm['sales'] = results
subm.to_csv('submission.csv',index=False)