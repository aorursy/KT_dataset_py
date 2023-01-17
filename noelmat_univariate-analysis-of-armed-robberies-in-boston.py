import pandas as pd

import numpy as np



pd.read_csv('/kaggle/input/monthly-robberies.csv',header=0,index_col=0,parse_dates=True,squeeze=True).head()
#split into a training and validation dataset

series= pd.read_csv('/kaggle/input/monthly-robberies.csv', header=0,index_col=0, parse_dates=True, squeeze=True)

split_point= len(series) -12

dataset, validation = series[0:split_point],series[split_point:]

print('Dataset %d, Validation %d' % (len(dataset),len(validation)))

dataset.to_csv('dataset.csv')

validation.to_csv('validation.csv')
#prepare data

X = series.values

X = X.astype('float32')

train_size = int(len(X)*0.50)

train,test = X[0:train_size], X[train_size:]
#evaluate a persistence model

import pandas as pd

from sklearn.metrics import mean_squared_error

from math import sqrt



#load data

series = pd.read_csv('/kaggle/working/dataset.csv',header=0,index_col=0,parse_dates=True,squeeze=True)



#prepare data

X = series.values

X = X.astype('float32')

train_size = int(len(X)* 0.50)

train, test = X[0:train_size],X[train_size:]



#walk forward validation

history = [x for x in train]

predictions = list()



for i in range(len(test)):

    #predict

    yhat = history[-1]

    predictions.append(yhat)

    #observation

    obs= test[i]

    history.append(obs)

    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))

#report performance

rmse = sqrt(mean_squared_error(test,predictions))

print('RMSE: %.3f' % rmse)
print(series.describe())
#line plots of time series

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

series.plot()

plt.show()
# density plots of time series

series = pd.read_csv('/kaggle/working/dataset.csv',header=None,index_col=0,parse_dates=True,squeeze=True)

plt.figure(1,figsize=(10,8))

plt.subplot(211)

series.hist()

plt.subplot(212)

series.plot(kind='kde')

plt.show()

series = pd.read_csv('/kaggle/working/dataset.csv',header=None,index_col=0,parse_dates=True,squeeze=True)

print(series)

groups = series['1966':'1973'].groupby(pd.Grouper(freq='A'))

years = pd.DataFrame()

for name, group in groups:

    years[name.year] = group.values

plt.figure(figsize=(10,8))

years.boxplot()

plt.show()
#statistical test for the stationarity fof the timeseries

from statsmodels.tsa.stattools import adfuller



#create a differenced time series

def difference(dataset):

    diff = list()

    for i in range(1,len(dataset)):

        value = dataset[i] - dataset[i-1]

        diff.append(value)

    return pd.Series(diff)



series = pd.read_csv('/kaggle/working/dataset.csv',header=None,index_col=0,parse_dates=True,squeeze=True)

X = series.values

# difference datra

stationary = difference(X)

stationary.index = series.index[1:]



#check if stationary

result = adfuller(stationary)

print('ADF statistic: %f' % result[0] )

print('p-value: %f' % result[1])

print('Critical values:')

for key,value in result[4].items():

    print('\t%s: %.3f' % (key,value))



#save

stationary.to_csv('stationary.csv')
# ACF and PACF plots of the time series

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf



series = pd.read_csv('/kaggle/working/dataset.csv',header=None,index_col=0,parse_dates=True,squeeze=True)

plt.figure(figsize=(10,8))

plt.subplot(211)

plot_acf(series,lags=50,ax=plt.gca())

plt.subplot(212)

plot_pacf(series,lags=50,ax=plt.gca())

plt.show()
#evaluate manually configured ARIMA model

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA

from math import sqrt



series = pd.read_csv('/kaggle/working/dataset.csv',header=None,index_col=0,parse_dates=True,squeeze=True)



X = series.values

X = X.astype('float32')

train_size = int(len(X)*0.50)

train, test = X[0: train_size],X[train_size:]

#walkforward validation

history = [x for x in train]

predictions = list()

for i in range(len(test)):

    #predict

    model = ARIMA(history, order=(0,1,2))

    model_fit= model.fit(disp=0)

    yhat = model_fit.forecast()[0]

    predictions.append(yhat)

    #observation

    obs = test[i]

    history.append(obs)

    print('>Predicted=%.3f, Expected=%3.f'% (yhat, obs))

#report performance

rmse = sqrt(mean_squared_error(test,predictions))

print('RMSE: %.3f' % rmse)
import warnings

import pandas as pd

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

from math import sqrt



#evaluate an ARIMA model for a given order (p,d,q) and return RMSE

def evaluate_arima_model(X, arima_order):

    #prepare training dataset

    X = X.astype('float32')

    train_size = int(len(X) * 0.50)

    train, test = X[0:train_size],X[train_size:]

    history = [x for x in train]

    #make predictions

    predictions = list()

    for t in range(len(test)):

        model = ARIMA(history,order=arima_order)

        model_fit = model.fit(disp=0)

        yhat= model_fit.forecast()[0]

        predictions.append(yhat)

        history.append(test[t])

    #calculate out of sample error

    rmse = sqrt(mean_squared_error(test,predictions))

    return rmse



def evaluate_models(dataset,p_values, d_values, q_values):

    data = dataset.astype('float32')

    best_score, best_cfg = float('inf'), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    rmse = evaluate_arima_model(dataset, order)

                    if rmse < best_score:

                        best_score, best_cfg = rmse,order

                    print('ARIMA%s RMSE=%.3f' % (order,rmse))

                except:

                    continue

    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

    

# load dataset

series = pd.read_csv('/kaggle/working/dataset.csv',header=None, index_col=0,parse_dates=True,squeeze=True)

#evaluate parameters

p_values = range(0,13)

d_values = range(0,4)

q_values = range(0,13)

warnings.filterwarnings('ignore')

# evaluate_models(series.values,p_values,d_values,q_values)

        
series = pd.read_csv('/kaggle/working/dataset.csv',header=None,index_col=0,parse_dates=True,squeeze=True)

#prepare data

X = series.values

X = X.astype('float32')

train_size= int(len(X)*0.5)

train,test = X[:train_size],X[train_size:]

#walk-forward validation

history = [x for x in train]

predictions = list()

for i in range(len(test)):

    #predict

    model = ARIMA(history, order=(0,1,2))

    model_fit=model.fit(disp=0)

    yhat=model_fit.forecast()[0]

    predictions.append(yhat)

    #observation

    obs = test[i]

    history.append(obs)

# errors

residuals = [test[i] - predictions[i] for i in range(len(test))]

residuals = pd.DataFrame(residuals)

plt.figure(figsize=(10,8))

plt.subplot(211)

residuals.hist(ax=plt.gca())

plt.subplot(212)

residuals.plot(kind='kde',ax=plt.gca())

plt.show()
series = pd.read_csv('/kaggle/working/dataset.csv',header=None, index_col=0,parse_dates=True,squeeze=True)

X = series.values

X = X.astype('float32')

train_size = int(len(X) * 0.5)

train, test = X[:train_size],X[train_size:]

#walk forward validation

history = [x for x in train]

predictions=list()

for i in range(len(test)):

    model = ARIMA(history, order=(0,1,2))

    model_fit= model.fit(disp=0)

    yhat=model_fit.forecast()[0]

    predictions.append(yhat)

    obs= test[i]

    history.append(obs)

#errors

residuals = [test[i]-predictions[i] for i in range(len(test))]

residuals = pd.DataFrame(residuals)

plt.figure(figsize=(10,8))

plt.subplot(211)

plot_acf(residuals,lags=25,ax=plt.gca())

plt.subplot(212)

plot_pacf(residuals,lags=25,ax=plt.gca())

plt.show()
from scipy.stats import boxcox

from statsmodels.graphics.gofplots import qqplot



X = series.values

transformed, lam =boxcox(X)

print("Lambda: %f" % lam)

plt.figure(1,figsize=(10,8))

plt.subplot(311)

plt.plot(transformed)

#histogram

plt.subplot(312)

plt.hist(transformed)

plt.subplot(313)

qqplot(transformed, line='r',ax=plt.gca())

plt.show()
#invert Box-Cox transform

from math import log,exp

def boxcox_inverse(value,lam):

    if lam == 0:

        return exp(value)

    return exp(log(lam * value + 1)/ lam)
#evaluate ARIMA models with box-cox transformed time series



series = pd.read_csv('/kaggle/working/dataset.csv',header=None,index_col=0,parse_dates=True,squeeze=True)

#prepare data

X = series.values

X - X.astype('float32')

train_size = int(len(X)*0.5)

train, test= X[:train_size],X[train_size:]

#walk-forward validation

history=[x for x in train]

predictions= list()

for i in range(len(test)):

    #transform

    transformed, lam = boxcox(history)

    if lam < -5:

        transformed, lam= history, 1

    # predict

    model = ARIMA( transformed, order=(0,1,2))

    model_fit = model.fit(disp=0)

    yhat = model_fit.forecast()[0]

    #invert transformed prediction

    yhat = boxcox_inverse(yhat,lam)

    predictions.append(yhat)

    

    obs = test[i]

    history.append(obs)

    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))

#report performance

rmse = sqrt(mean_squared_error(test,predictions))

print('RMSE: %.3f' % rmse)
# finalize model and save to file

import numpy as np



#monkey patch around bug in ARIMA class

def __getnewargs__(self):

    return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))



ARIMA.__getnewargs__  = __getnewargs__



#load data

series = pd.read_csv('dataset.csv',header=None,index_col=0,parse_dates=True,squeeze=True)

#prepare data

X = series.values

X = X.astype('float32')



transformed, lam = boxcox(X)



model = ARIMA(transformed, order=(0,1,2))

model_fit= model.fit(disp=0)



model_fit.save('model.pkl')

np.save('model_lambda.npy',[lam])
#load the finalized model and make a prediction

from statsmodels.tsa.arima_model import ARIMAResults

from math import exp,log



model_fit= ARIMAResults.load('model.pkl')

lam = np.load('model_lambda.npy')

yhat = model_fit.forecast()[0]

yhat = boxcox_inverse(yhat,lam)

print('Predicted: %.3f' % yhat)
#load and prepare datasets

dataset = pd.read_csv('dataset.csv',header=None,index_col=0,parse_dates=True,squeeze=True)

X = dataset.values.astype('float32')

history = [x for x in X]

validation = pd.read_csv('validation.csv',header=None,index_col=0,parse_dates=True,squeeze=True)

y = validation.values.astype('float32')

#load model

model_fit = ARIMAResults.load('model.pkl')

lam = np.load('model_lambda.npy')

#make first prediction

predictions = list()

yhat = model_fit.forecast()[0]

yhat = boxcox_inverse(yhat,lam)

predictions.append(yhat)

history.append(y[0])

print('>Predicted=%.3f, Expected=%.3f' % (yhat,y[0]))

#rolling forecasts

for i in range(1,len(y)):

    transformed,lam = boxcox(history)

    if lam < -5:

        transformed, lam = history, 1

    #predict

    model = ARIMA(transformed, order=(0,1,2))

    model_fit= model.fit(disp=0)

    yhat = model_fit.forecast()[0]

    #invert transformed prediction

    yhat = boxcox_inverse(yhat,lam)

    predictions.append(yhat)

    #observation

    obs = y[i]

    history.append(obs)

    print('>Predicted=%.3f, Expected=%3.f' % (yhat,obs))

#report performance

rmse = sqrt(mean_squared_error(y,predictions))

print('RMSE: %.3f' %rmse)

plt.plot(y)

plt.plot(predictions,color='red')

plt.show()