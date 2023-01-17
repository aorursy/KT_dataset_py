import numpy as np 
import pandas as pd 
import itertools
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
#Training Data
train = pd.read_csv('../input/predice-el-futuro/train_csv.csv')
train.head()
#Testing Data Shape
train.shape
#Checking Null Values
train.isnull().sum()
#Checking Data Types
train.dtypes
#Converting time object to datetime
train['time']= pd.to_datetime(train['time']) 
train.info()
#Min time
train['time'].min()
#Max time
train['time'].max()
# Checking Anomalies by using Describe 
train.describe()
#Drop id column
train=train.drop(columns='id',axis=1)
train.head()
# Indexing with time series data
train = train.set_index('time')
train.index
train.plot(figsize=(15, 6))
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose
season = seasonal_decompose(train, freq=2)
fig = season.plot();
fig.set_size_inches(18,8)
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(train, model='additive',freq=2)
fig = decomposition.plot()
plt.show()
#Dickey Fuller Test
adfinput = adfuller(train['feature'])
adftest = pd.Series(adfinput[0:4], index=['Dickey Fuller Statistical Test', 'P-value',
                                          'Used Lags', 'Number of comments used'])
adftest = round(adftest,4)
    
for key, value in adfinput[4].items():
    adftest["Critical Value (%s)"%key] = value.round(4) 
adftest
#Kpss
kpss_input = kpss(train['feature'])
kpss_test = pd.Series(kpss_input[0:3], index=['Statistical Test KPSS', 'P-Value', 'Used Lags'])
kpss_test = round(kpss_test,4)
    
for key, value in kpss_input[3].items():
    kpss_test["Critical Value (%s)"%key] = value 
kpss_test
#Autocorrelation Plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(train['feature'])
plt.show()
# ACF and PACF Plots
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
pyplot.figure()
pyplot.subplot(211)
plot_acf(train, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(train, ax=pyplot.gca())
pyplot.show()
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train['feature'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
mod = sm.tsa.statespace.SARIMAX(train['feature'],
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()
pred = results.get_prediction(start=pd.to_datetime('2019-03-19 00:10:00'), dynamic=False)
pred_ci = pred.conf_int()

ax = train['feature'].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('time')
ax.set_ylabel('Feature')
plt.legend()

plt.show()
y_forecasted = pred.predicted_mean
y_truth = train['feature']['2019-03-19 00:10:00':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=40)
pred_ci = pred_uc.conf_int()

ax = train['feature'].plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('time')
ax.set_ylabel('Feature')

plt.legend()
plt.show()
pred_uc.predicted_mean
# Converting y_pred to a dataframe which is an array
ytest= pd.DataFrame(pred_uc.predicted_mean)
ytest
#Reset Index
ytest = ytest.reset_index(drop=True)
ytest
#Testing Data
test=pd.read_csv('../input/predice-el-futuro/test_csv.csv')
test.head()
#Converting time object to datetime
test['time']= pd.to_datetime(test['time']) 
test.info()
#Adding two Dataframes
test_pred = pd.concat([test, ytest], axis=1)
test_pred
#Adding column name
test_pred['feature']=test_pred[0]
test_pred
#Which are not require removing that columns
test_pred=test_pred.drop(columns=['time',0],axis=1)
test_pred
#Final Solution file
test_pred.to_csv('solution.csv',index=False)
