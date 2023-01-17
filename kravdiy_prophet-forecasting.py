# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fbprophet import Prophet

from statsmodels.tsa.stattools import adfuller

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = {

    'tra': pd.read_csv('../input/air_visit_data.csv'),

    'as': pd.read_csv('../input/air_store_info.csv'),

    'hs': pd.read_csv('../input/hpg_store_info.csv'),

    'ar': pd.read_csv('../input/air_reserve.csv'),

    'hr': pd.read_csv('../input/hpg_reserve.csv'),

    'id': pd.read_csv('../input/store_id_relation.csv'),

    'tes': pd.read_csv('../input/sample_submission.csv'),

    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})

    }



data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])



for df in ['ar','hr']:

    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])

    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date

    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])

    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date    

    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)

    data[df] = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})

    data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])

    

data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek

data['tra']['year'] = data['tra']['visit_date'].dt.year

data['tra']['month'] = data['tra']['visit_date'].dt.month

data['tra']['visit_date'] = data['tra']['visit_date'].dt.date



data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])

data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))

data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])

data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek

data['tes']['year'] = data['tes']['visit_date'].dt.year

data['tes']['month'] = data['tes']['visit_date'].dt.month

data['tes']['visit_date'] = data['tes']['visit_date'].dt.date



unique_stores = data['tes']['air_store_id'].unique()

stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)



tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])



stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 

lbl = LabelEncoder()

stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])

stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])



data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])

data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])

data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 

test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 



train = pd.merge(data['tra'], stores, how='left', on=['air_store_id','dow']) 

test = pd.merge(data['tes'], stores, how='left', on=['air_store_id','dow'])



for df in ['ar','hr']:

    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 

    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

    

train = train.fillna(-1)

test = test.fillna(-1)



col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]

train = train.fillna(-1)

test = test.fillna(-1)



train.head()
train.describe()
import pandas_profiling
pandas_profiling.ProfileReport(train)
% time date = train.groupby('visit_date').nunique()

date['visitors'] = train.groupby('visit_date').visitors.agg('sum')
def test_stationarity(ts):

    

    #Determing rolling statistics

    rolmean = ts.rolling(window=2, center=False).mean()

    rolstd = ts.rolling(window=2, center=False).std()



    #Plot rolling statistics:

    orig = plt.plot(ts, color='blue', label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label='Rolling Std')

    plt.legend(loc='best')

    fig_size[0] = 20

    fig_size[1] = 10

    plt.rcParams["figure.figsize"] = fig_size

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(ts, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():

        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)
fig_size = plt.rcParams["figure.figsize"]

fig_size

fig_size[0] = 12

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size
date.index
ts = date['visitors']

ts.head(20)
X = ts.values

ts.dropna(inplace=True)

test_stationarity(ts)
ts_log = np.log(ts)

plt.plot(ts_log)
test_stationarity(ts_log)
import logging

logging.getLogger('fbprophet.forecaster').propagate = False



df_sub = pd.read_csv('../input/sample_submission.csv')

df_sub['store_id'] = df_sub['id'].apply(lambda x:x[:-11])



df_sub = df_sub.set_index('id')



number_of_stores = df_sub['store_id'].nunique()

date_range = pd.date_range(start=pd.to_datetime('2016-07-01'),

                           end=pd.to_datetime('2017-04-22'))

forecast_days = (pd.to_datetime('2017-05-31')-pd.to_datetime('2017-04-22')).days



for cnt, store_id in enumerate(df_sub['store_id'].unique()):

    print('Predicting %d of %d.'%(cnt, number_of_stores), end='\r')

    data = train[train['air_store_id'] == store_id]

    data = data[['visit_date', 'visitors']].set_index('visit_date')

    # Ensure we have full range of dates.

    data = data.reindex(date_range).fillna(0).reset_index()

    data.columns = ['ds', 'y']

    

    m = Prophet()

    #m = Prophet(yearly_seasonality=True, mcmc_samples=300)

    #m.add_seasonality(name='weekly', period=7, fourier_order=3)

    m.fit(data)

    future = m.make_future_dataframe(forecast_days)

    forecast = m.predict(future)

    forecast = forecast[['ds', 'yhat']]

    forecast.columns = ['id', 'visitors']

    forecast['id'] = forecast['id'].apply(lambda x:'%s_%s'%(store_id, x.strftime('%Y-%m-%d')))

    forecast = forecast.set_index('id')

    df_sub.update(forecast)

print('\n\nDone.')
df_sub
df_sub = df_sub.reset_index()[['id','visitors']]

df_sub['visitors'] = df_sub['visitors'].clip(lower=0)

df_sub.to_csv('submission_1.csv', index=False)

df_sub.head(10)
date_amount = pd.DataFrame(date, columns=['visit_date', 'visitors'])

date_amount.index
def to_prophet(df_ts):

    df = pd.DataFrame(df_ts, columns=['visit_date', 'visitors'])

    df.drop(['visit_date'], axis=1, inplace=True)

    df.reset_index(inplace=True)

    df.rename(columns={"visit_date": "ds", "visitors": "y"}, inplace=True)

    return df
date = to_prophet(date)
m = Prophet(daily_seasonality=True, mcmc_samples=150)

m.fit(date)

future = m.make_future_dataframe(periods=40)

future.head()

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
m.plot(forecast);
m.plot_components(forecast);
from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(m, horizon = '30 days')

df_cv.head()
from sklearn.metrics import mean_squared_error
def RMSLE(y, pred):

    return mean_squared_error(y, pred)**0.5
RMSLE(df_cv['y'], df_cv['yhat'])
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(39)
from sklearn import *

X = train[col]

y = pd.DataFrame()

y['visitors'] = np.log1p(train['visitors'].values)



y_test_pred = 0



K = 4

kf = model_selection.KFold(n_splits = K, random_state = 1, shuffle = True)

np.random.seed(1)



params = {'n_estimators': 100, 

        'max_depth': 8,

        'min_samples_split': 200, 

        'min_samples_leaf': 50,

        'learning_rate': 0.005,

        'max_features':  9,

        'subsample': 0.9,

        'loss': 'ls'}



model = ensemble.GradientBoostingRegressor(**params)

for i, (train_index, test_index) in enumerate(kf.split(train)):



    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]

    X_train, X_valid = X.iloc[train_index, :].copy(), X.iloc[test_index, :].copy()

    X_test = test[col]

    print("\nFold ", i)



    fit_model = model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    print('RMSE GBM Regressor, fold ', i, ': ', RMSLE(y_valid, pred))

    print('Prediction length on validation set, GBM Regressor, fold ', i, ': ', len(pred))



    pred = model.predict(X_test)

    print('Prediction length on test set, GBM Regressor, fold ', i, ': ', len(pred))

    y_test_pred += pred



    del X_test, X_train, X_valid, y_train



% time y_test_pred /= K  

df_sub1 = pd.DataFrame()

df_sub1['id'] = test['id']

df_sub1['visitors'] = np.expm1(y_test_pred) 

df_sub1.to_csv('submission_2.csv', float_format='%.6f', index=False)

df_sub1.head(10)
x1 = df_sub1['visitors']

x2 = df_sub['visitors']

compr = x1 - x2

compr
from statsmodels.tsa.stattools import acf, pacf



lag_acf = acf(ts, nlags=20)

lag_pacf = pacf(ts, nlags=20, method='ols')



plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0, linestyle='--', color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ts)), linestyle='--', color='gray')

plt.axhline(y=1.96/np.sqrt(len(ts)), linestyle='--', color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--', color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ts)), linestyle='--', color='gray')

plt.axhline(y=1.96/np.sqrt(len(ts)), linestyle='--', color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
from statsmodels.tsa.seasonal import seasonal_decompose



decomposition = seasonal_decompose(np.asarray(ts), freq=7)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



decomposition.plot()

plt.show()