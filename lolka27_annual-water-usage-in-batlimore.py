import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from pandas.plotting import autocorrelation_plot

from sklearn.metrics import mean_squared_error
series = pd.read_csv('/kaggle/input/watercsv/water.csv', header=0, index_col=0,parse_dates=True ,squeeze=True)

series.head()
series.describe()
#smoothing average good way to check trend, popular check in stock data



fig = plt.figure(figsize=(15,8))

plt.plot(series, color='black')

plt.plot(series.rolling(window=3, min_periods=1).mean(), color='yellow')

plt.plot(series.ewm(alpha=0.7).mean(), color='red')

plt.show()
autocorrelation_plot(series) # it is not whote noise!!! Data are correlated
fig = plt.figure(figsize=(15,8))

plt.subplot(1,3,1)

plt.boxplot(series)

plt.subplot(1,3,2)

plt.hist(series)

plt.subplot(1,3,3)

series.plot(kind='kde')

plt.show()
X = series.values

size = int(len(X)*0.6)

train, test = X[0:size], X[size:]





history = [x for x in train]

predictions = list()

residuals = list()



for i in range(len(test)):

    yhat = history[-1]

    predictions.append(yhat)

    obs = test[i]

    history.append(obs)

    residual = yhat - obs

    residuals.append(residual)

    print('Predicted=%.3f, Expected=%.3f' % (yhat,obs))



rmse = np.sqrt(mean_squared_error(test, predictions))

print('RMSE: %.3f' % rmse)    

    

    
fig = plt.figure(figsize=(15,8))

plt.plot(history, color='blue')

plt.plot([None for i in train] + [x for x in test], color='red')

plt.plot([None for i in train] + [x for x in predictions], color='pink')

plt.show()
residuals = pd.Series(residuals)

residuals.describe()
residuals.plot(kind='kde')
autocorrelation_plot(residuals) #no correclation  - good!!!
#test Dickey-Fuller



def adful_test(X):

    results = adfuller(X)

    print("statistic:%.3f" % results[0])

    print("p_value: %.7f" % results[1])

    print("num: %3.f" % results[2])

    print("samples: %1.f" % results[3])

    if results[1]<0.05:

        print("stationary")

    else:

        print("non-stationary")
adful_test(series)
def differ(series,d):

    X = series.values

    diff = list()

    for i in range(1, len(X)):

        dif = X[i] - X[i-1]

        diff.append(dif)

    return pd.Series(diff)
series = pd.read_csv('/kaggle/input/watercsv/water.csv', header=0, index_col=0,parse_dates=True ,squeeze=True)

differences = differ(series,1)

adful_test(differences)
differences.plot()
plt.figure() 

plt.subplot(211) 

plot_acf(series, lags=20, ax=plt.gca()) 

plt.subplot(212) 

plot_pacf(series, lags=20, ax=plt.gca()) 

plt.show()
X = series.values

size = int(len(X)*0.5)

train, test = X[0:size], X[size:]



history = [x for x in train]

prediction = list()



for i in range(len(test)):

    model = ARIMA(history, order=(4,1,1))

    model_fit = model.fit(disp=0)

    yhat = model_fit.forecast()[0]

    prediction.append(yhat)

    

    obs = test[i]

    history.append(obs)

    print('Predicted:%.3f, Expected:%.3f' % (yhat, obs))

rmse = np.sqrt(mean_squared_error(test, prediction))

rmse
def evaluating_arima_model(X, arima_order):

    X = X.astype('float32')

    size = int(len(X)*0.5)

    train, test = X[0:size], X[size:]

    history = [x for x in train]

    predictions = list()

    

    for t in range(len(test)):

        model = ARIMA(history, order = arima_order)

        model_fit = model.fit(trend='nc', disp=0)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        

        obs = test[t]

        history.append(obs)

    rmse = np.sqrt(mean_squared_error(test, predictions))

    return rmse
def evaluate_param(dataset, p_values, d_values, q_values):

    dataset = dataset.astype('float32')

    best_score, best_cfg = np.float('inf'), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p, d, q)

                try:

                    rmse = evaluating_arima_model(dataset, order)

                    if rmse < best_score:

                        best_score, best_cfg = rmse, order

                    print('ARIMA%s, rmse:%.3f' % (order, rmse))

                except:

                    continue

    print('Best ARIMA%s, rmse:%.3f' % (best_cfg, best_score))

                
series = pd.read_csv('/kaggle/input/watercsv/water.csv', header=0, index_col=0,parse_dates=True ,squeeze=True)



p_values = range(0,5)

d_values = range(0,3)

q_values = range(0,5)

import warnings

warnings.filterwarnings('ignore')

evaluate_param(series.values, p_values, d_values, q_values)
series = pd.read_csv('/kaggle/input/watercsv/water.csv', header=0, index_col=0, parse_dates=True, squeeze=True)



X = series.values



size = int(len(X)*0.5)



train, test = X[0:size], X[size:]

history = [x for x in train]

predictions = list()

res = list()



for i in range(len(test)):

    model = ARIMA(history, order=(2,1,0))

    model_fit = model.fit(trend='nc', disp=0)

    yhat = model_fit.forecast()[0]

    predictions.append(yhat)

    obs = test[i]

    history.append(obs)

    r = obs - yhat

    res.append(r)

rmse = np.sqrt(mean_squared_error(test, predictions))
res = pd.DataFrame(res, columns=['val'])

res.describe()
autocorrelation_plot(res)
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

plt.hist(res.val)

plt.subplot(1,2,2)

plt.boxplot(res.val)
from statsmodels.graphics.gofplots import qqplot

qqplot(res.val, line='s')
series = pd.read_csv('/kaggle/input/watercsv/water.csv', header=0, index_col=0, squeeze=True, parse_dates=True)



X = series.values

size = int(len(X)*0.5)



train, test = X[0:size], X[size:]

history = [x for x in train]

predictions = list()

bias = 3.064462 



for i in range(len(test)):

    model = ARIMA(history, order=(2,1,0))

    model_fit = model.fit(tresd='nc',disp=0)

    yhat = bias + np.float( model_fit.forecast()[0])

    predictions.append(yhat)

    history.append(test[i])

    

rmse = np.sqrt(mean_squared_error(test, predictions))

residuals = [test[i] - predictions[i] for i in range(len(test))]

residuals = pd.DataFrame(residuals)





plt.figure(figsize=(15,8))

plt.plot(test, color='r')

plt.plot(predictions, color='b')

plt.legend(['test', 'pred'])

plt.show()
rmse
residuals.describe()
residuals.plot(kind='kde')
from statsmodels.tsa.api import Holt, ExponentialSmoothing, SimpleExpSmoothing
X = series.values

size = int(len(X)*0.7)

train, test = X[0:size], X[size:]



model_holt = Holt(train, exponential=True)

model_fit = model_holt.fit()



model_HW = ExponentialSmoothing(train, trend='add')

model_fit_HW = model_HW.fit()
model_pred = model_fit.forecast(steps=24)

model_HW_pred = model_fit_HW.forecast(steps=24)
model_fit_HW.params
plt.figure()

plt.plot(X, color='r')

plt.plot([None for x in train] + [b for b in model_pred], color='green')

plt.plot([None for i in train] + [g for g in model_HW_pred], color='black')

plt.legend(['history', 'Holt', 'HW'])

plt.show()