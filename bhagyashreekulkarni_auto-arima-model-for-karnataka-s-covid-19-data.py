## AUTO ARIMA model for Karnataka's Covid Data

import numpy as np          
import pandas as pd
import seaborn as sns
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sts
import statsmodels.graphics.tsaplots as sgt

## import the dataset
covid_data = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
covid_data.info()
## choose data only for Karnataka
state_name = 'Karnataka'
state_data = covid_data.loc[covid_data['State/UnionTerritory'] == state_name]
state_data.info()
## To continue further data should have constant time intervals,hence considering only date and confirmed column
data = state_data[['Date','Confirmed']]

## subset rows which are not equal to 0
data = data[data.Confirmed != 0]

## Dates are used as index values in order to be easily understood by python as true date object
data['Date'] = pd.to_datetime(data['Date'],dayfirst = True)
data.set_index ('Date',inplace = True)
data.plot(figsize=(30,10),title = state_name)
plt.show()
## Check for stationarity, DF Test 
series_index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used']
sns.set()
dftest = sts.adfuller(data.Confirmed)
dfoutput = pd.Series(dftest[0:4], index=series_index)
print (dfoutput)
## converting the series to stationary by taking log transformation and differencing
data_log = np.log(data)
data_log_diff = data_log - data_log.shift(1)
data_log_diff.dropna(inplace = True)

## Check for Stationarity again
dftest = sts.adfuller(data_log_diff)
dfoutput = pd.Series(dftest[0:4], index=series_index)
print (dfoutput)
plt.plot(data_log_diff)
## Dividing the data into Train and Test data
train = data_log_diff.iloc[:len(data)-30] 
test = data_log_diff.iloc[len(data)-30:]

## Removing values with 0
train = train[train.Confirmed != 0]
test = test[test.Confirmed != 0]
# ACF plot 
sgt.plot_acf(train,lags=10,zero=False) 
plt.title("ACF Plot - {}".format(state_name), size=24)
plt.show()

#PACF plot 
sgt.plot_pacf(train,lags=30,zero=False, method=("ols"))
plt.title("PACF Plot - {}".format(state_name), size=24)
plt.show()
## Auto Arima
arima_model = auto_arima(data_log_diff['Confirmed'], start_p = 0, start_q = 0, 
                          max_p = 3, max_q = 4, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = 0, D = 0, trace = True, 
                          error_action ='warn',   # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True,
                          random_state = 20,
                          n_fits = 10)  

arima_model.summary() 
## Predicting on the test data
prediction = pd.DataFrame(arima_model.predict(n_periods = 29),index = test.index)
prediction.columns = ['Predicted_Value']

## Plotting the data
plt.figure(figsize=(15,10))
plt.plot(train,label = "training")
plt.plot(test, label = "test")
plt.plot(prediction, label = "pred_values")
plt.legend(loc = 'Left corner')
plt.show()
## Use plot diagnostics to get a clear picture
arima_model.plot_diagnostics(figsize = (8,8)) 
## Further checks on the model

## r2 score
from sklearn.metrics import r2_score
test['pred_value'] = prediction
r2_score(test['Confirmed'],prediction['Predicted_Value'])


## MAPE Value
mape = np.mean(np.abs(prediction.Predicted_Value.values - test.Confirmed)/np.abs(test.Confirmed)) * 100
print("mape : {}".format(mape))

## RMSE value
rmse = np.mean((prediction.Predicted_Value.values - test.Confirmed)**2)**.5
print("rmse : {}".format(rmse))

## MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test.Confirmed , prediction.Predicted_Value.values)
print("mse : {}".format(mse))