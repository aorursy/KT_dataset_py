%matplotlib  inline

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from fbprophet import Prophet

import os
print(os.listdir("../input"))
df = pd.read_csv("../input/avocado.csv", parse_dates=["Date"])

del df['Unnamed: 0']
cols = ['Date', 'AveragePrice', 'type', 'region']
df = df[cols]
df = df[(df.region =='TotalUS') & (df.type == 'conventional') ] #& (df.Date >= '2016-01-01')

del df['region']
del df['type']

df = df.sort_values("Date")

df.columns = ['ds', 'y']
df.set_index('ds', inplace=True)

# Train test split 
train = df[:-12]
test = df[-12:]

train.info()
train.head()
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

ax = sns.scatterplot(x=train.index, y=train.y)
ax = sns.scatterplot(x=test.index, y=test.y)

ax.axes.set_xlim(train.index.min(), test.index.max());
from pandas import DataFrame
from datetime import datetime, timedelta
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import statsmodels.api as sm
from itertools import product
from math import sqrt
from sklearn.metrics import mean_squared_error 

import warnings
warnings.filterwarnings('ignore')

colors = ["windows blue", "amber", "faded green", "dusty purple"]
sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 })
seasonal_decompose(train.y, model='additive').plot()
print("Dickey–Fuller test: p=%f" % adfuller(train.y)[1])
train['y_box'], lmbda = stats.boxcox(train.y)

seasonal_decompose(train.y_box, model='additive').plot()
print("Dickey–Fuller test: p=%f" % adfuller(train.y_box)[1])
# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
ax = plt.subplot(211)

# Plot the autocorrelation function
plot_acf(train.y_box[0:].values.squeeze(), lags=16, ax=ax)
ax = plt.subplot(212)
plot_pacf(train.y_box[0:].values.squeeze(), lags=16, ax=ax)
plt.tight_layout()
train['y_box_1d'] = train['y_box'].diff(periods=1)
train.head()
fig, ax_arr = plt.subplots(2,1)

ax_arr[0].plot(train.y_box)
ax_arr[1].plot(train.y_box_1d)
plt.tight_layout();
# STL-decomposition
seasonal_decompose(train.y_box_1d[1:]).plot()   
print("Dickey–Fuller test: p=%f" % adfuller(train.y_box_1d[1:])[1])
# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
ax = plt.subplot(211)
# Plot the autocorrelation function
plot_acf(train.y_box_1d[1:].values.squeeze(), lags=16, ax=ax)
ax = plt.subplot(212)
plot_pacf(train.y_box_1d[1:].values.squeeze(), lags=16, ax=ax)
plt.tight_layout()
# Initial approximation of parameters
ps = range(0, 2)
d = 1
qs = range(0, 2)

parameters = product(ps, qs)
parameters_list = list(parameters)
len(parameters_list)
%%time 

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model = SARIMAX(train.y_box, order=(param[0], d, param[1])).fit(disp=-1)
    except ValueError:
        print('bad parameter combination:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by='aic', ascending=True).head())
print(best_model.summary())
print("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[13:])[1])
best_model.plot_diagnostics(figsize=(15, 12))
plt.show()
# Inverse Box-Cox Transformation Function
def invboxcox(y,lmbda):
   if lmbda == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(lmbda*y+1)/lmbda))
test['yhat_ARIMA'] = invboxcox(best_model.forecast(12), lmbda)
test['yhat_ARIMA'] = np.round(test.yhat_ARIMA, 2)

test.tail()
test.y.plot(linewidth=3)
test.yhat_ARIMA.plot(color='r', ls='--', label='Predicted Units', linewidth=3)

plt.legend()
plt.grid()
plt.title('Price - weekly forecast')
plt.ylabel('$');
test['e'] = test.y - test.yhat_ARIMA

rmse = np.sqrt(np.mean(test.e**2)).round(2)
mape = np.round(np.mean(np.abs(100*test.e/test.y)), 0)

print('RMSE = $', rmse)
print('MAPE =', mape, '%')
%%time 

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model = SARIMAX(train.y_box, order=(param[0], d, param[1]), trend='ct').fit(disp=-1)
    except ValueError:
        print('bad parameter combination:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())
print("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[13:])[1])
best_model.plot_diagnostics(figsize=(15, 12))
plt.show()
test['yhat_ARIMAct'] = invboxcox(best_model.forecast(12), lmbda)
test['yhat_ARIMAct'] = np.round(test.yhat_ARIMAct, 2)

test.tail()
test.y.plot(linewidth=3)
test.yhat_ARIMAct.plot(color='r', ls='--', label='Predicted Units', linewidth=3)

plt.legend()
plt.grid()
plt.title('Price - weekly forecast')
plt.ylabel('$');
test['e'] = test.y - test.yhat_ARIMAct

rmse = np.sqrt(np.mean(test.e**2)).round(2)
mape = np.round(np.mean(np.abs(100*test.e/test.y)), 0)

print('RMSE = $', rmse)
print('MAPE =', mape, '%')
%%time 

# Initial approximation of parameters
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D=1
d=1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

i = 52 # weekly seasonality 

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model = SARIMAX(train.y_box, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], i)).fit(disp=-1)
    except ValueError:
        print('bad parameter combination:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())
print("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[i+1:])[1])
best_model.plot_diagnostics(figsize=(15, 12))
plt.show()
# STL-decomposition
plt.subplot(211)
best_model.resid[i+1:].plot()
plt.ylabel(u'Residuals')
ax = plt.subplot(212)

plot_acf(best_model.resid[i+1:].values.squeeze(), lags=i, ax=ax)

print("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[i+1:])[1])

plt.tight_layout()
test['yhat_SARIMA'] = invboxcox(best_model.forecast(12), lmbda)
test['yhat_SARIMA'] = np.round(test.yhat_SARIMA, 2)

test.tail()
test.y.plot(linewidth=3)
test.yhat_SARIMA.plot(color='r', ls='--', label='Predicted Units', linewidth=3)

plt.legend()
plt.grid()
plt.title('Price - weekly forecast')
plt.ylabel('$');
test['e'] = test.y - test.yhat_SARIMA

rmse = np.sqrt(np.mean(test.e**2)).round(2)
mape = np.round(np.mean(np.abs(100*test.e/test.y)), 0)

print('RMSE = $', rmse)
print('MAPE =', mape, '%')
%%time 

# Initial approximation of parameters
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D=1
d=1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

i = 52 # weekly seasonality 

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model = SARIMAX(train.y_box, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], i), trend='ct').fit(disp=-1)
    except ValueError:
        print('bad parameter combination:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())
print("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[i+1:])[1])
best_model.plot_diagnostics(figsize=(15, 12))
plt.show()
# STL-decomposition
plt.subplot(211)
best_model.resid[i+1:].plot()
plt.ylabel(u'Residuals')
ax = plt.subplot(212)

plot_acf(best_model.resid[i+1:].values.squeeze(), lags=i, ax=ax)

print("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[i+1:])[1])

plt.tight_layout()
test['yhat_SARIMAct'] = invboxcox(best_model.forecast(12), lmbda)
test['yhat_SARIMAct'] = np.round(test.yhat_SARIMAct, 2)

test.tail()
test.y.plot(linewidth=3)
test.yhat_SARIMAct.plot(color='r', ls='--', label='Predicted Units', linewidth=3)

plt.legend()
plt.grid()
plt.title('Price - weekly forecast')
plt.ylabel('$');
test['e'] = test.y - test.yhat_SARIMAct

rmse = np.sqrt(np.mean(test.e**2)).round(2)
mape = np.round(np.mean(np.abs(100*test.e/test.y)), 0)

print('RMSE = $', rmse)
print('MAPE =', mape, '%')

del test['e']
test
test.y.plot(linewidth=3)

test.yhat_ARIMA.plot(color='r', ls='--', label='ARIMA forecast', linewidth=3)
test.yhat_ARIMAct.plot(color='r', ls=':', label='ARIMA constant trend', linewidth=3)
test.yhat_SARIMA.plot(color='grey', ls='--', label='SARIMA', linewidth=3)
test.yhat_SARIMAct.plot(color='grey', ls=':', label='SARIMA constant trend', linewidth=3)

plt.legend()
plt.grid()
plt.title('Price - weekly forecast')
plt.ylabel('$');
