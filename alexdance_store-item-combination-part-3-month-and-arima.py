import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from pylab import rcParams
import statsmodels.api as sm
import warnings
import itertools
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
df = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv")
df.head()
df['date'] =  pd.to_datetime(df['date'])
df = df.set_index('date')
df.head()
df.sales.sum()
salesbymonth = df.sales.resample('M').sum()
salesbymonth.head()
split = "2017-01-01"
salesbymonth_train= salesbymonth[:split]
salesbymonth_train.head()
salesbymonth_test= salesbymonth[split:]
salesbymonth_test_final=salesbymonth_test.copy() # This file is used to compare all the predections
salesbymonth_test_final = pd.DataFrame(salesbymonth_test_final)
salesbymonth_test_final.head()
salesbymonth_test_final.info()
salesbymonth.sample(5)
salesbyday = df.sales.resample('D').sum()
salesbyday_train= salesbyday[:split]
salesbyday_test= salesbyday[split:]
salesbyday_test_final=salesbyday_test.copy() # This file is used to compare all the daily forecasts
salesbyday_test_final = pd.DataFrame(salesbyday_test_final)
salesbyday_test_final.head()
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(salesbymonth_train, model='additive')
fig = decomposition.plot()
plt.show()
p = d = q = range(0, 2)
pdqa = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
for param in pdqa:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(salesbymonth_train, order=param, seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)                                
            results = modl.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

SARIMAMonth = sm.tsa.statespace.SARIMAX(salesbymonth, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12) ,enforce_stationarity=False,enforce_invertibility=False)
SARIMA_results_month = SARIMAMonth.fit()
print(SARIMA_results_month.summary().tables[1])
SARIMA_results_month.plot_diagnostics(figsize=(16, 8))
plt.show()
#SARIMA_predict_month_1 = SARIMA_results_month.predict(start=1461,end=1825) # this is from 1 Jan 2017 to 31 Dec 2017
#SARIMA_predict_month_1 = SARIMA_month_model.predict(start=48,end=60,rder=(1, 1, 1), seasonal_order=(1, 1, 1, 12) ,enforce_stationarity=False,enforce_invertibility=False) # this is from Jan 2017 to  Dec 2017

SARIMA_predict_month_1 = SARIMA_results_month.predict(start=48,end=60) #,order=(1, 1, 1), seasonal_order=(1, 1, 1, 12) ,enforce_stationarity=False,enforce_invertibility=False) # this is from Jan 2017 to  Dec 2017
print(SARIMA_predict_month_1)
salesbymonth_test_final['SeasonalARIMA'] = SARIMA_predict_month_1
salesbymonth_test_final.head()
RMSE_Month_Seasonal_ARIMA  = np.mean(np.sqrt((salesbymonth_test_final['SeasonalARIMA'] - salesbymonth_test_final['sales']) ** 2)) 
print(RMSE_Month_Seasonal_ARIMA)
model_ar_month = AR (salesbymonth_train)
model_ar_month_fit = model_ar_month.fit()
predictions_month_1 = model_ar_month_fit.predict(start=48,end=59)
AR_month_predictions=pd.DataFrame(predictions_month_1, columns =['AR'])
AR_month_predictions.head()
salesbymonth_test.head(3)
plt.plot(salesbymonth_test)
plt.plot(AR_month_predictions['AR'], color = 'red' )
salesbymonth_test_final['sales']
salesbymonth_test_final['AR'] = AR_month_predictions['AR']
RMSE_Month_AR  = np.mean(np.sqrt((salesbymonth_test_final['AR'] - salesbymonth_test_final['sales']) ** 2)) 
print(RMSE_Month_AR)
salesbymonth_test_final['AR_error'] = salesbymonth_test_final['AR'] - salesbymonth_test_final['sales']
salesbymonth_test_final['AR_error_percent'] = salesbymonth_test_final['AR_error'] / salesbymonth_test_final['sales']
salesbymonth_test_final.sample(10)
salesbymonth_test_final.sum()
salesbymonth_test_final.head()
salesbymonth_train.head()
decomposition_day = sm.tsa.seasonal_decompose(salesbyday_train, model='additive')
fig = decomposition_day.plot()
plt.show()
p = d = q = range(0, 2)
pdqb = list(itertools.product(p, d, q))
seasonal_pdq_day = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
for param in pdqb:
    for param_seasonal_day in seasonal_pdq_day:
        try:
            mod = sm.tsa.statespace.SARIMAX(salesbyday_train, order=param, seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)                                
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal_day, results.aic))
        except:
            continue
SARIMADay = sm.tsa.statespace.SARIMAX(salesbyday, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12) ,enforce_stationarity=False,enforce_invertibility=False)
SARIMA_results_day = SARIMADay.fit()
print(SARIMA_results_day.summary().tables[1])
SARIMA_results_day.plot_diagnostics(figsize=(16, 8))
plt.show()
SARIMA_predict_day_1 = SARIMA_results_day.predict(start=1461,end=1825) # this is from 1 Jan 2017 to 31 Dec 2017
print(SARIMA_predict_day_1)

salesbyday_test_final['SeasonalARIMA'] = SARIMA_predict_day_1
RMSE_Day_SeasonalARIMA  = np.mean(np.sqrt((salesbyday_test_final['SeasonalARIMA'] - salesbyday_test_final['sales']) ** 2)) 
print(RMSE_Day_SeasonalARIMA)
model_arima_month = ARIMA(salesbymonth_train, order = (7,1,0))
salesbymonth_train.tail(12)
model_arima_month_fit = model_arima_month.fit()
arima_predictions_month = model_arima_month_fit.forecast(steps=12)[0]
print(arima_predictions_month)
ARIMA_month_predictions=pd.DataFrame(arima_predictions_month, columns =['ARIMA'])
ARIMA_month_predictions['ARIMA']
salesbymonth_test_final =salesbymonth_test_final.reset_index()
salesbymonth_test_final.head()
salesbymonth_test_final['ARIMA'] =ARIMA_month_predictions['ARIMA']
salesbymonth_test_final.tail(14)
plt.plot(salesbymonth_test_final['sales'],linestyle='dashed',linewidth=5)
plt.plot(salesbymonth_test_final['ARIMA'], color = 'red' )
RMSE_Month_ARIMA  = np.mean(np.sqrt((salesbymonth_test_final['ARIMA'] - salesbymonth_test_final['sales']) ** 2)) 
print(RMSE_Month_ARIMA)
p=d=q =range(0,8)
pdqmontha = list(itertools.product(p,d,q))
for param in pdqmontha:
    try:
        model_arima_month = ARIMA(salesbymonth_train, order = pdqmontha)
        model_arima_month_fit = model_arima_month.fit()
        print(param,model_arima_month_fit.aic)
    except:
        continue
model_arima_day = ARIMA(salesbyday_train, order = (2,1,0))
model_arima_day_fit = model_arima_day.fit()
arima_predictions_day = model_arima_day_fit.forecast(steps=365)[0]
ARIMA_day_predictions=pd.DataFrame(arima_predictions_day, columns =['ARIMA'])
ARIMA_day_predictions['ARIMA']
salesbyday_test_final =salesbyday_test_final.reset_index()
salesbyday_test_final['ARIMA'] = ARIMA_day_predictions['ARIMA']
salesbyday_test_final.head()
plt.plot(salesbyday_test_final['sales'],linestyle='dashed',linewidth=5)
plt.plot(salesbyday_test_final['ARIMA'], color = 'red' )
p=d=q =range(0,5)
pdqday = list(itertools.product(p,d,q))
warnings.filterwarnings('ignore')
for param in pdqday:
    try:
        model_arima_month = ARIMA(salesbymonth_train, order = param)
        model_arima_month_fit = model_arima_month.fit()
        print(param,model_arima_month_fit.aic)
    except:
        continue
ARIMA_day_predictions.tail()
RMSE_Day_ARIMA  = np.mean(np.sqrt((salesbyday_test_final['ARIMA'] - salesbyday_test_final['sales']) ** 2))
print(RMSE_Day_ARIMA)
salesbyday_test_final.shape
model_ar_day = AR (salesbyday_train)
model_ar_day_fit = model_ar_day.fit()
predictions_day_1 = model_ar_day_fit.predict(start=1461,end=1825)
predictions_day_1.head()
AR_day_predictions=pd.DataFrame(predictions_day_1, columns =['AR'])
AR_day_predictions.shape
AR_day_predictions.head()
salesbyday_test_final['sales']
salesbyday_test_final.head()
AR_day_predictions.shape
salesbyday_test_final['AR'] = AR_day_predictions['AR']
salesbyday_test_final.head()
RMSE_Day_AR  = np.mean(np.sqrt((salesbyday_test_final['AR'] - salesbyday_test_final['sales']) ** 2))
print(RMSE_Day_AR)
salesbyday_test_final['AR_error'] = salesbyday_test_final['AR'] - salesbyday_test_final['sales']
salesbyday_test_final['AR_error_percent'] = salesbyday_test_final['AR_error'] / salesbyday_test_final['sales']
salesbyday_test_final.head(12)
salesbyday_test_final.sum()
salesbymonth_test_final.sum()
salesbyday_test_final.reset_index()
plt.plot(salesbyday_test_final['sales'],linestyle='dashed',linewidth=5)
plt.plot(salesbyday_test_final['ARIMA'], color = 'red' )
plt.plot(salesbymonth_test_final['sales'],linestyle='dashed',linewidth=5)
plt.plot(salesbymonth_test_final['ARIMA'], color = 'red' )
plt.plot(salesbymonth_test_final['AR'], color = 'blue' )
plt.plot(salesbymonth_test_final['SeasonalARIMA'], color = 'orange' )
