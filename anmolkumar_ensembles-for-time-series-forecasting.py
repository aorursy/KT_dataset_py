import warnings

warnings.filterwarnings("ignore")

import itertools

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import pandas as pd

import statsmodels.api as sm

from pylab import rcParams

rcParams['figure.figsize'] = 18, 6

from tqdm import tqdm_notebook as tqdm



!pip install dateparser

import dateparser



from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV



from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



from fbprophet import Prophet



import matplotlib

matplotlib.rcParams['axes.labelsize'] = 24

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'
def date_parser(x):

    return pd.datetime.strptime(x, '%d-%m-%Y %H:%M')



train_data = pd.read_csv('../input/train.csv', index_col = 1, parse_dates = [1], date_parser = date_parser)

test_data = pd.read_csv('../input/test.csv', index_col = 1, parse_dates = [1], date_parser = date_parser)

train_data.columns = train_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

test_data.columns = test_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

y = train_data['count'].resample('D').sum()

train_data = pd.DataFrame(y.astype(int))

train_data.head(20)
train_data.shape
train_data.isnull().sum()
test_data.isnull().sum()
train_data.dtypes
train_data.nunique()
test_ID = test_data['id'].values

test_data = test_data.drop('id', axis = 1)

#train_data = train_data.drop('id', axis = 1)

train_data.head()
train_data.index[1]
train_data.plot()

plt.show()
plot_acf(train_data)

plt.show()

plot_pacf(train_data)

plt.show()
#calling auto correlation function

lag_acf = acf(train_data, nlags = 250)

#Plot ACF:

plt.figure(figsize = (16, 7))

plt.plot(lag_acf, marker = '+')

plt.axhline(y = 0, linestyle = '--', color = 'gray')

plt.axhline(y = -1.96/np.sqrt(len(train_data)), linestyle = '--', color = 'gray')

plt.axhline(y = 1.96/np.sqrt(len(train_data)), linestyle = '--', color = 'gray')

plt.title('Autocorrelation Function')

plt.xlabel('number of lags')

plt.ylabel('correlation')

plt.tight_layout()
rcParams['figure.figsize'] = 18, 22

sm.tsa.seasonal_decompose(train_data['count'], freq = 24).plot()

result = sm.tsa.stattools.adfuller(train_data['count'])

plt.show()
X = train_data.values

train = X[0:535] # 534 data as train data

test = X[535:]  # 228 data as test data

predictions = []
p = d = q = range(0, 3)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
param_aic_seasonal = {}

for param in tqdm(pdq):

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(train, 

                                            order = param, 

                                            seasonal_order = param_seasonal)

            results = mod.fit()

            #print('ARIMA{}x{}7 - AIC:{}'.format(param, param_seasonal, results.aic))

            param_aic_seasonal[param, param_seasonal] = results.aic

        except:

            continue
{k: v for k, v in sorted(param_aic_seasonal.items(), key = lambda item: item[1])}
"""

ARIMA(0, 1, 2)x(0, 2, 2, 7)7 - AIC:7796.930684951293

ARIMA(0, 1, 3)x(0, 2, 2, 7)7 - AIC:7798.195336121731

ARIMA(1, 1, 3)x(0, 2, 2, 7)7 - AIC:7800.689430109041

ARIMA(1, 1, 3)x(1, 2, 3, 7)7 - AIC:7801.455544313485

ARIMA(1, 1, 3)x(0, 2, 3, 7)7 - AIC:7817.536127810834

ARIMA(1, 1, 3)x(1, 2, 2, 7)7 - AIC:7821.356956716344



ARIMA(2, 2, 2)x(2, 2, 3, 7)7 - AIC:7812.823078011056

ARIMA(2, 2, 2)x(1, 2, 3, 7)7 - AIC:7813.068983822397



ARIMA(0, 0, 1)x(1, 2, 2, 7)7 - AIC:7838.228692638588

ARIMA(0, 0, 1)x(1, 2, 3, 7)7 - AIC:7840.423833342043

ARIMA(0, 0, 1)x(0, 2, 2, 7)7 - AIC:7836.734523887294

ARIMA(0, 0, 1)x(0, 2, 3, 7)7 - AIC:7838.20996590997

ARIMA(0, 0, 1)x(0, 2, 4, 7)7 - AIC:7840.067090844341

ARIMA(0, 0, 1)x(0, 2, 5, 7)7 - AIC:7840.052586356935



ARIMA(0, 0, 0)x(0, 2, 2, 7)7 - AIC:7923.896707311315

ARIMA(0, 0, 0)x(0, 2, 3, 7)7 - AIC:7912.299111269296

ARIMA(0, 0, 0)x(0, 2, 4, 7)7 - AIC:7925.219651372339

ARIMA(0, 0, 0)x(0, 2, 5, 7)7 - AIC:7914.754898950585

ARIMA(0, 0, 0)x(0, 3, 5, 7)7 - AIC:7942.688159983001

ARIMA(0, 0, 0)x(1, 2, 2, 7)7 - AIC:7912.265579854375

ARIMA(0, 0, 0)x(1, 2, 3, 7)7 - AIC:7933.789327232746

ARIMA(0, 0, 0)x(1, 2, 4, 7)7 - AIC:7916.247023781987

ARIMA(0, 0, 0)x(1, 2, 5, 7)7 - AIC:7928.90160963447

ARIMA(0, 0, 0)x(4, 2, 2, 7)7 - AIC:7918.9959792007885

ARIMA(0, 0, 0)x(5, 2, 1, 7)7 - AIC:7923.534611519698

ARIMA(0, 0, 0)x(5, 2, 2, 7)7 - AIC:7924.175712677705

ARIMA(0, 0, 0)x(5, 2, 4, 7)7 - AIC:7932.990955173896

ARIMA(0, 0, 0)x(5, 3, 2, 7)7 - AIC:7902.697468081234

"""
# 1. Based on minimal AIC : {((0, 1, 2), (0, 2, 2, 7)): 7796.9306841312355}



mod = sm.tsa.statespace.SARIMAX(train, order = (0, 1, 2), seasonal_order = (0, 2, 2, 7))

results = mod.fit()

print(results.summary())



y_pred = results.predict(start = 535, end = 761, dynamic = True)

rcParams['figure.figsize'] = 18, 6

plt.plot(test)

plt.plot(y_pred, color = 'red')

plt.show()

# preds_old = results.predict(start = 0, end = 761, dynamic = False)

print(y_pred.sum())
train_df = train_data[:]



mod = sm.tsa.statespace.SARIMAX(train_df['count'], order = (0, 1, 2), seasonal_order = (0, 2, 2, 7))

results = mod.fit()

print(results.summary())



y_new = results.predict(start = 762, end = 974, dynamic = True)

y_old = results.predict(start = 0, end = 761, dynamic = False)

print(y_new.sum())
plt.plot(train_data.index, train_data['count'])

plt.plot(y_new.index, y_new)

plt.plot(y_old.index, y_old, 'm')

plt.show()
train_data = pd.read_csv('../input/train.csv', parse_dates = [1], date_parser = date_parser)

test_data = pd.read_csv('../input/test.csv', parse_dates = [1], date_parser = date_parser)

train_data.columns = train_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

test_data.columns = test_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')



train_data['date'] = train_data['datetime'].dt.date

train_data = train_data.groupby(['date'])['count'].sum().reset_index()

train_data.columns = ['ds', 'y']

train_data['y'] = np.log(train_data['y'])

model = Prophet(daily_seasonality = False)

model.add_seasonality(name = 'weekly', period = 7, fourier_order = 300)

model.fit(train_data)

future = model.make_future_dataframe(periods = 213)

forecast = model.predict(future)

fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
f_preds = np.exp(forecast[-213:]['yhat']).values

er = []

listm = [0.044286588, 0.035343014, 0.029911076, 0.024714453, 0.02080223, 0.018621427, 

         0.020023091, 0.023221497, 0.026741002, 0.034555218, 0.049047207, 0.05437526, 

         0.054951351, 0.048600186, 0.051965438, 0.051309072, 0.049999488, 0.051164262, 

         0.052423477, 0.055626605, 0.053455246, 0.049894816, 0.050075828, 0.048892166]

for p in range(len(f_preds)):

    for l in range(len(listm)):

        er.append(f_preds[p]*listm[l])



Result = {'Count': er}



submission_df = pd.DataFrame({

                  "ID": pd.Series([x + 18288 for x in range(5112)]), 

                  "Count": pd.Series(Result['Count'])})

submission_df.to_csv('submission_1.csv', index = False)
#checking how well the learning happened



y_old_df = y_old.reset_index(name = "count")

y_new_df = y_new.reset_index(name = "count")

plt.plot(range(0, len(y_old_df)), train_df['count'], color = 'red')

plt.plot(range(0, len(y_old_df)), y_old_df['count'])
df = pd.read_csv("../input/train.csv")

df['Datetime'] = pd.to_datetime(df['Datetime'], format = '%d-%m-%Y %H:%M')

df['date'] = df['Datetime'].dt.date

df = df.groupby(['date'])['Count'].sum().reset_index()

df['date'] = pd.to_datetime(df['date'])

df['month'] = df.date.dt.month

df['day'] = df.date.dt.weekday

df['month_start'] = df.date.apply(lambda x: 1 if x.is_month_start else 0)

df['month_end'] = df.date.apply(lambda x: 1 if x.is_month_end else 0)

df['week_start'] = df.day.apply(lambda x: 1 if x == 0 or x == 1 else 0)

df['week_end'] = df.day.apply(lambda x: 1 if x == 5 or x == 6 else 0)

df['month_day'] = df.date.dt.day

df['predicted'] = y_old_df['count']

df['diff'] = df.date.apply(lambda x: (x - dateparser.parse('2012-08-25')).days)

df.head()
dfh = df.groupby(['date'])['Count'].sum().reset_index()

dfh.sort_values(['Count'], ascending = [False]).head(50)
df.groupby(['month'])['Count'].mean().plot()
df.groupby(['day'])['Count'].sum().plot()
df.groupby(['month_day'])['Count'].sum().plot()
df.groupby(['month_start'])['Count'].sum().plot()
def forecast_accuracy(forecast, actual):

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE

    me = np.mean(forecast - actual)             # ME

    mae = np.mean(np.abs(forecast - actual))    # MAE

    mpe = np.mean((forecast - actual)/actual)   # MPE

    rmse = np.mean((forecast - actual)**2)**.5  # RMSE

    corr = np.corrcoef(forecast, actual)[0, 1]   # corr

    mins = np.amin(np.hstack([forecast[:, None], 

                              actual[:, None]]), axis = 1)

    maxs = np.amax(np.hstack([forecast[:, None], 

                              actual[:, None]]), axis = 1)

    minmax = 1 - np.mean(mins/maxs)             # minmax

    acf1 = 0#acf(fc-test)[1]                      # ACF1

    return({'mape': mape, 'me': me, 'mae': mae, 

            'mpe': mpe, 'rmse': rmse, 'acf1': acf1, 

            'corr': corr, 'minmax': minmax})
new_df = df.drop(['date'], axis = 1)

new_df.shape
test_df = pd.read_csv("../input/test.csv")

test_df['Datetime'] = pd.to_datetime(test_df['Datetime'], format = '%d-%m-%Y %H:%M')

test_df['date'] = test_df['Datetime'].dt.date

test_df = test_df.groupby(['date']).sum().reset_index()

test_df['date'] = pd.to_datetime(test_df['date'])

test_df['month'] = test_df.date.dt.month

test_df['day'] = test_df.date.dt.weekday

test_df['month_start'] = test_df.date.apply(lambda x: 1 if x.is_month_start else 0)

test_df['month_end'] = test_df.date.apply(lambda x: 1 if x.is_month_end else 0)

test_df['week_start'] = test_df.day.apply(lambda x: 1 if x == 0 or x == 1 else 0)

test_df['week_end'] = test_df.day.apply(lambda x: 1 if x == 5 or x == 6 else 0)

test_df['month_day'] = test_df.date.dt.day

test_df['predicted'] = y_new_df['count']

test_df['diff'] = test_df.date.apply(lambda x: (x - dateparser.parse('2012-08-25')).days)

test_df.head()
X_train = new_df[:]

y_train = X_train['Count']

dropcols = ['Count']

X_train.drop(dropcols, axis = 1, inplace = True)



X_test = new_df[700:]

y_test = X_test['Count']

X_test.drop(dropcols, axis = 1, inplace = True)



testing_df = test_df.drop(["ID", "date"], axis = 1)

testing_df
linear = linear_model.LinearRegression()

linear.fit(X_train, y_train)



linear1 = linear_model.LinearRegression()

linear1.fit(X_train, np.log(y_train))



rtp = linear.predict(X_test)

rtp1 = np.exp(linear1.predict(X_test))



new_rtp = (rtp + rtp1) / 2



print(forecast_accuracy(np.array(rtp), np.array(y_test)))

print(forecast_accuracy(np.array(rtp1), np.array(y_test)))



for c, cc in enumerate(linear.coef_):

    print(X_train.columns[c])

    print(cc)
plt.plot(range(0, len(rtp)), rtp, color = 'red')

plt.plot(range(0, len(rtp)), y_test)
plt.plot(range(0, len(y_old_df[700:])), train_df[700:]['count'], color = 'red')

plt.plot(range(0, len(y_old_df[700:])), y_old_df[700:]['count'])
rf = RandomForestRegressor(random_state = 42)

random_grid = {'bootstrap': [True, False], 

               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None], 

               'max_features': ['auto', 'sqrt'], 

               'min_samples_leaf': [1, 2, 4], 

               'min_samples_split': [2, 5, 10, 15], 

               'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

              }



rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose = 1, random_state = 42, n_jobs = -1)

rf_random.fit(X_train, y_train)
rf_model = RandomForestRegressor(**rf_random.best_params_)

rf_model.fit(X_train, y_train)
rtp = rf_model.predict(X_test)

print(forecast_accuracy(np.array(rtp), np.array(y_test)))
plt.plot(range(0, len(rtp)), rtp, color = 'red')

plt.plot(range(0, len(rtp)), y_test)
linear_pred = linear.predict(testing_df)
plt.plot(df.index, df.Count)

plt.plot(testing_df.index + 761, linear_pred)
rf_test_pred = rf_model.predict(testing_df)

plt.plot(df.index, df.Count)

plt.plot(testing_df.index + 761, rf_test_pred)
rf_test_pred.sum()
ensemble_preds = (y_new + f_preds + linear_pred) / 3
plt.plot(X_train.index, y_train)

plt.plot(testing_df.index + 761, ensemble_preds)
er = []

listm = [0.044286588, 

         0.035343014, 

         0.029911076, 

         0.024714453, 

         0.02080223, 

         0.018621427, 

         0.020023091, 

         0.023221497, 

         0.026741002, 

         0.034555218,

         0.049047207, 

         0.05437526, 

         0.054951351, 

         0.048600186, 

         0.051965438, 

         0.051309072, 

         0.049999488, 

         0.051164262, 

         0.052423477, 

         0.055626605, 

         0.053455246, 

         0.049894816, 

         0.050075828, 

         0.048892166]

for p in range(len(ensemble_preds)):

    for l in range(len(listm)):

        er.append(ensemble_preds[p]*listm[l])

d = {'Count': er}

predf = pd.DataFrame(data = d)

submission_df = pd.DataFrame({

                  "ID": pd.Series([x + 18288 for x in range(5112)]),

                  "Count": pd.Series(predf['Count'])})

submission_df.to_csv('submission_FBprophet.csv', index = False)