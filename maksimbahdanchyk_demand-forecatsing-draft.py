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
train = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')

test  = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')

samp  = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv')
test.date = pd.to_datetime(test.date)
train.loc[:,'date'] = pd.to_datetime(train.loc[:,'date'])

train.info()
print('Store: ',train.store.unique(),'{} stores'.format(len(train.store.unique())))

print('Item: ',train.item.unique(),'{} items'.format(len(train.item.unique())))
import matplotlib.pyplot as plt



fig,ax = plt.subplots(figsize = (20,10))



for j in range(3):

    s = np.random.randint(min(train.store),max(train.store))

    i  = np.random.randint(min(train.item),max(train.item))

    

    temp = train.loc[(train.store == s) & (train.item ==i),:]

        

    ax.plot(temp.date,temp.sales,label = 'Store {} Item {}'.format(s,i))



ax.set_xlabel('datetime',fontsize = 20)

ax.set_ylabel('sales',fontsize = 20)

ax.tick_params(axis='both', which='major', labelsize=20)

plt.legend()
ex = train.loc[(train.store == 9) & (train.item ==12),:]



ex.loc[:,'day_of_month']  = ex.loc[:,'date'].dt.day

ex.loc[:,'month_of_year'] = ex.loc[:,'date'].dt.month

ex.loc[:,'year']          = ex.loc[:,'date'].dt.year

ex.loc[:,'day_of_year']   = ex.loc[:,'date'].dt.dayofyear

ex.loc[:,'day_of_week']   = ex.loc[:,'date'].dt.dayofweek



ex_train = ex.loc[ex.year <  2017,:]

ex_test  = ex.loc[ex.year == 2017,:]



plt.plot(ex_train.date,ex_train.sales)

plt.plot(ex_test.date,ex_test.sales)
temp
fig,ax = plt.subplots(figsize = (20,10))



for year in ex_train.year.unique():

    temp = ex_train.loc[ex_train.year == year,:]

    ax.plot(temp.day_of_year,temp.sales,label = year)

    ax.set_xlabel('Day of a year',fontsize = 20)

    ax.set_ylabel('Sales',fontsize = 20)

    ax.tick_params(labelsize = 20)

plt.legend()  
# Null hypothesis: Non Stationarity exists in the series.

# Alternative Hypothesis: Stationarity exists in the series



# Data: (-1.8481210964862593, 0.35684591783869046, 0, 1954, {'10%': -2.5675580437891359, 

# '1%': -3.4337010293693235, '5%': -2.863020285222162}, 21029.870846458849



# Lets break data one by one.

# First data point: -1.8481210964862593: Critical value of the data in your case

# Second data point: 0.35684591783869046: Probability that null hypothesis will not be rejected(p-value)

# Third data point: 0: Number of lags used in regression to determine t-statistic. So there are no auto correlations going back to '0' periods here.

# Forth data point: 1954: Number of observations used in the analysis.

# Fifth data point: {'10%': -2.5675580437891359, '1%': -3.4337010293693235, '5%': -2.863020285222162}: T values corresponding to adfuller test.





# Since critical value -1.8>-2.5,-3.4,-2.8 (t-values at 1%,5%and 10% confidence intervals), null hypothesis cannot be rejected. So there is non stationarity in your data

# Also p-value of 0.35>0.05(if we take 5% significance level or 95% confidence interval), null hypothesis cannot be rejected.



# Hence data is non stationary (that means it has relation with time)







ex_train_rm = ex_train.copy()

ex_train_rm.loc[:,'sales'] = ex_train_rm.loc[:,'sales'].rolling(window = 30).mean()

plt.plot(ex_train.date,ex_train.sales)

plt.plot(ex_train_rm.date,ex_train_rm.sales)





from statsmodels.tsa.stattools import adfuller



adf = adfuller(ex_train.sales,autolag='AIC')



print(adf[0],

      adf[1],

      adf[4])



adf = adfuller(ex_train_rm.sales.dropna(),autolag='AIC')



print(adf[0],

      adf[1],

      adf[4])

ex_train_log = ex_train.copy()

ex_train_log.loc[:,'sales'] = np.log(ex_train_log.loc[:,'sales'])



plt.plot(ex_train.date,ex_train.sales)

plt.plot(ex_train_log.date,ex_train_log.sales)



from statsmodels.tsa.stattools import adfuller



adf = adfuller(ex_train.sales,autolag='AIC')



print(adf[0],

      adf[1],

      adf[4])



adf = adfuller(ex_train_log.sales,autolag='AIC')



print(adf[0],

      adf[1],

      adf[4])



ex_train_diff= ex_train.copy()

ex_train_diff.loc[:,'sales'] = ex_train_diff.loc[:,'sales'].diff()



plt.plot(ex_train.date,ex_train.sales)

plt.plot(ex_train_diff.date,ex_train_diff.sales)



from statsmodels.tsa.stattools import adfuller

adf = adfuller(ex_train.sales,autolag='AIC')



print(adf[0],

      adf[1],

      adf[4])



adf = adfuller(ex_train_diff.dropna().sales,autolag='AIC')



print(adf[0],

      adf[1],

      adf[4])
from statsmodels.tsa import seasonal

# seasonal trend residual

decompose = seasonal.seasonal_decompose(ex_train.set_index('date')['sales'],model='additive',extrapolate_trend = 'freq',period = 365)



fig,ax = plt.subplots(4,1,figsize = (10,10))





ax[0].plot(decompose.observed.index,decompose.observed)

ax[1].plot(decompose.observed.index,decompose.trend,linewidth=10 )

ax[2].plot(decompose.observed.index,decompose.seasonal)

ax[3].plot(decompose.observed.index,decompose.resid)
decompose2 = seasonal.seasonal_decompose(decompose.seasonal,model='additive',extrapolate_trend = 'freq',period = 365)



fig,ax = plt.subplots(4,1,figsize = (10,10))





ax[0].plot(decompose2.observed.index,decompose2.observed)

ax[1].plot(decompose2.observed.index,decompose2.trend )

ax[2].plot(decompose2.observed.index,decompose2.seasonal)

ax[3].plot(decompose2.observed.index,decompose2.resid)
## diff

ex_train_diff = ex_train.copy()

ex_train_diff.sales = ex_train.sales.diff()



# trend,season,remove



from statsmodels.tsa.seasonal import seasonal_decompose



decomposed  = seasonal_decompose(ex_train.set_index('date').sales,model = 'additive',period=365,extrapolate_trend = 'freq')



plt.plot(ex_train_diff.date,ex_train_diff.sales)

plt.plot(ex_train.date,decomposed.resid,color = 'red',linestyle = '-')



# on diff data

from statsmodels.tsa.stattools import adfuller, kpss

adf = adfuller(ex_train_diff.dropna().sales,autolag='AIC')

print('diff',adf[1])

print('diff',adf[0],adf[4])



#on resid

adf = adfuller(decompose.resid,autolag='AIC')

print('Trend removal', adf[1])

print('Trend removal',adf[0],adf[4])




# on row data

from statsmodels.tsa.stattools import adfuller, kpss

adf = adfuller(ex_train.sales,autolag='AIC')

print(adf[1])

print(adf[0],adf[4])



#on resid

adf = adfuller(decompose.resid,autolag='AIC')

print(adf[1])

print(adf[0],adf[4])
from statsmodels.tsa.stattools import acf,pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



acf_50 = acf(ex_train_diff.sales.dropna(), nlags=10)

pacf_50 = pacf(ex_train_diff.sales.dropna(), nlags=10)



fig, axes = plt.subplots(1,2,figsize=(16,3))

plot_acf(ex_train_diff.sales.dropna(), lags=10, ax=axes[0])

plot_pacf(ex_train_diff.sales.dropna(),lags=10, ax=axes[1])
ex_train = ex_train.set_index('date')[['sales']].resample('D').mean()

ex_test = ex_test.set_index('date')[['sales']].resample('D').mean()
plt.plot(ex_train)
from statsmodels.tsa.statespace.sarimax import SARIMAX





sarima = SARIMAX(ex_train,

                 order=(1,1,1),

                 seasonal_order=(1,1,1,7),

                 freq='D')

sarima_fit = sarima.fit(disp = 0)
fig = sarima_fit.plot_diagnostics()
pred = sarima_fit.get_prediction(start = '2013-01-06',end = '2018-01-01',dynamic = False)

mean_pred = pred.predicted_mean

conf_mean = pred.conf_int()



fig1,ax1 = plt.subplots(figsize = (15,10))



ax1.plot(ex_train.index,ex_train.values,color = 'black',linestyle = '-',linewidth = 2)

ax1.plot(mean_pred.index,mean_pred.values,color = 'red',linestyle = '-',linewidth = 2)

ax1.plot(conf_mean['2014':].index,conf_mean['2014':],color = 'blue',linestyle = '--',linewidth = 1)

rmse = np.mean(np.abs((ex_train.values-mean_pred.values)))

print(rmse)


print(sarima_fit.summary())
forecast = sarima_fit.get_forecast(steps = 365)

mean_forecast = forecast.predicted_mean

conf_forecast = forecast.conf_int()



fig1,ax1 = plt.subplots(figsize = (15,10))



ax1.plot(ex_test.index,ex_test.values,color = 'black',linestyle = '-',linewidth = 2)

ax1.plot(mean_forecast.index,mean_forecast.values,color = 'red',linestyle = '-',linewidth = 2)

ax1.plot(conf_forecast.index,conf_forecast,color = 'blue',linestyle = '--',linewidth = 1)



print(np.mean(np.sqrt((ex_test.values - mean_forecast.values)**2)))
test_id = test.id

test.drop('id',axis = 1,inplace = True)
train['set'] = 'train'

test['set']  = 'test'
data = pd.concat([train,test])
data['day_of_month']     = data.date.dt.day

data['month_of_year']   = data.date.dt.month

data['year']    = data.date.dt.year

data['day_of_year'] = data.date.dt.dayofyear

data['day_of_week'] = data.date.dt.dayofweek

data['is_weekday'] = data['day_of_week'].apply(lambda x: 1 if x in (6,7) else 0)

data['is_month_start']   = data.date.dt.is_month_start.map({False:0,True:1})

data['is_month_end']     = data.date.dt.is_month_end.map({False:0,True:1})





from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

holidays = calendar().holidays(start=data.date.min(), end=data.date.max())

data['ia_holiday'] = data.date.isin(holidays).astype(int)
grouping = data.groupby(['store','item'])

lags = [90,91,92,93,94,95,96,97,98,99,100,180,270]

for lag in lags:

    col_name = 'lag-'+str(lag)

    data[col_name] = grouping.sales.shift(lag)

lags = [90,97,104]



for lag in lags:

    col_name = 'rolling_mean-'+str(lag)

    data[col_name] = grouping.sales.shift(lag).rolling(window=7).mean()

    

for lag in lags:

    col_name = 'rolling_std-'+str(lag)

    data[col_name] = grouping.sales.shift(lag).rolling(window=7).std()
data.columns
test  = data.loc[data.set  == 'test',:]

train = data.loc[data.set == 'train',:].dropna()

train.sales = np.log1p(train.sales)
X = train.drop(['date','sales','set'],axis=1).dropna()

y = train.sales



X_train,X_val = X.loc[X.year < 2017],X.loc[X.year == 2017]

y_train,y_val = y.loc[X.year < 2017],y.loc[X.year == 2017]
from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline



from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.preprocessing import OneHotEncoder



transformer = make_column_transformer(

    (OneHotEncoder(),['store','item','day_of_week']),

    (MinMaxScaler(), ['day_of_month','day_of_year']),

    (StandardScaler(),['lag-90', 'lag-91','lag-92', 'lag-93', 'lag-94', 'lag-95', 'lag-96', 'lag-97', 'lag-98','lag-99', 'lag-100', 'lag-180', 'lag-270',

                        'rolling_mean-90',

                        'rolling_mean-97', 'rolling_mean-104', 'rolling_std-90',

                        'rolling_std-97', 'rolling_std-104']),

    remainder = 'passthrough'

)



from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb



regressor = xgb.XGBRegressor(n_estimators = 500,

                             max_depth = 5)



pipeline = make_pipeline(transformer,regressor)



pipeline.fit(X_train,y_train)

from sklearn.metrics import mean_absolute_error

pred_val = pipeline.predict(X_val)

pred_train = pipeline.predict(X_train)



print(mean_absolute_error(y_val,pred_val))

print(mean_absolute_error(y_train,pred_train))
def smape(preds, target):

    '''

    Function to calculate SMAPE

    '''

    n = len(preds)

    masked_arr = ~((preds==0)&(target==0))

    preds, target = preds[masked_arr], target[masked_arr]

    num = np.abs(preds-target)

    denom = np.abs(preds)+np.abs(target)

    smape_val = (200*np.sum(num/denom))/n

    return smape_val



print(smape(pred_val,y_val))

print(smape(pred_train,y_train))
pipeline.fit(X,y)

pred = pipeline.predict(X)

print(smape(pred,y))
X_test = test.copy().drop(['date','sales','set'],axis = 1)

pred_test = np.expm1(pipeline.predict(X_test))



sub = pd.DataFrame({'id':test_id,'sales':np.round(pred_test)})

sub.to_csv('submission.csv',index = False)