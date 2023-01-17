# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARMA, ARIMA
data = pd.read_csv("../input/sales_data_sample.csv", encoding = "unicode_escape")
data.info()
data.columns
data.isnull().sum()
data.describe()
data.head(2)
data["ORDERDATE"] = data["ORDERDATE"].astype("datetime64[ns]")

data["ORDERDATE"].max(), data["ORDERDATE"].min()
plt.figure(figsize =(50,8))

mean_group = data.groupby(["ORDERDATE"])["SALES"].mean()

plt.plot(mean_group)

plt.title("Time series average")

plt.show()
plt.figure(figsize =(50,8))

mean_group = data.groupby(["ORDERDATE"])["SALES"].median()

plt.plot(mean_group)

plt.title("Time series median")

plt.show()
plt.figure(figsize =(50,8))

mean_group = data.groupby(["ORDERDATE"])["SALES"].std()

plt.plot(mean_group)

plt.title("Time series standard deviation")

plt.show()
#data.set_index("ORDERDATE", inplace = True)
data.index
forcast = data[["ORDERDATE","SALES"]]

forcast = forcast.sort_values("ORDERDATE").reset_index()

forcast = forcast.drop("index", axis = True)

forcast["ORDERDATE"] = forcast["ORDERDATE"].astype("datetime64[ns]")

forcast.head()
forcast["ORDERDATE"].min(), forcast["ORDERDATE"].max()
leng = len(forcast["ORDERDATE"])

leng
forcast.set_index("ORDERDATE", inplace = True)

forcast.tail()
forcast.index
model = AR(forcast)

model_fit = model.fit()
#make prediction

#import statsmodels.tsa.ar_model.ARResultsWrapper.predict 

yhat = model_fit.predict(start = 2800, end = 2823)
yhat.columns = ["SALES"]

yhat.head()
#plt.plot(forcast["SALES"], color = "blue", label = 'Actual')

plt.plot(yhat, color = "blue", label = "predicted")

plt.show()
mod2 = ARMA(forcast["SALES"], order = (2,1))

res2 = mod2.fit(disp=False)
yhat2 = res2.predict(start = 2823,end = 3000)
#plt.plot(test_forcast, color='blue', label='Original')

plt.plot(yhat, color='red', label='AR')

plt.plot(yhat2, color='blue', label='ARMA')

plt.legend(loc='best')

plt.title('Comparing two models')

plt.show(block=False)
mvg_avg = forcast.rolling(window = 12). mean()

mvg_std = forcast.rolling(window = 12). std()



#orig = plt.plot(forcast, color='blue', label='Original')

mean = plt.plot(mvg_avg, color='red', label='Rolling Mean')

std = plt.plot(mvg_std, color='black', label='Rolling Std')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')

plt.show(block=False)
y = forcast['SALES'].resample('MS').mean()

forcast["2003":].head()
#Plot to see sales data visually

y.plot(figsize = (50,6))

plt.show()
from statsmodels.tsa.stattools import adfuller

result = adfuller(y)

print("ADF Statistics", result[0])

print("P value", result[1])
print('Results of Dickey Fuller Test:')

dftest = adfuller(forcast['SALES'], autolag='AIC')



dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

    

print(dfoutput)
#from statsmodels.tsa.seasonal import seasonal_decompose

#decompose = seasonal_decompose(y)



#plt.title("Original plot")

#plt.legend(loc = "best" )



#trend = decompose.trend

#plt.show()

#plt.plot(trend, label = "Trend")

#plt.title("Trend Plot")

#plt.legend(loc = "best")



#seasonal = decompose.seasonal

#plt.show()

#plt.plot(seasonal, label = "Seasonal")

#plt.title("Seasonal Plot")

#plt.legend(loc = "best")



#residual = decompose.resid

#plt.show()

#plt.plot(residual, label = "Residual")

#plt.legend(loc = "best")
#Transformation

forcast["SALES_log"] = np.log(forcast["SALES"])

#forcast["log_difference"] = forcast["SALES_log"] - forcast["SALES_log"].shift(1)

forcast_diff = forcast["SALES_log"]

#forcast["log_difference"].dropna().plot()
#moving_avg = forcast_diff.rolling(12).mean()

#plt.plot(forcast_diff)

#plt.plot(moving_avg, color='red')
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(forcast["SALES"], nlags=10)

lag_pacf = pacf(forcast["SALES"], nlags=10, method='ols')



#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(forcast["SALES"])),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(forcast["SALES"])),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0, linestyle='--', color='gray')

plt.axhline(y=-1.96/np.sqrt(len(forcast["SALES"])), linestyle='--', color='gray')

plt.axhline(y=1.96/np.sqrt(len(forcast["SALES"])), linestyle='--', color='gray')

plt.title('Partial Autocorrelation Function')

            

plt.tight_layout()            
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(forcast["SALES"], order=(2,1,0))

results_AR = model.fit(disp=-1)

plt.plot(forcast["SALES"])

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'%sum((results_AR.fittedvalues - forcast["SALES"])**2))

print('Plotting AR model')
model = ARIMA(forcast["SALES"], order=(0,1,2))

results_AR = model.fit(disp=-1)

plt.plot(forcast["SALES"])

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'%sum((results_AR.fittedvalues - forcast["SALES"])**2))

print('Plotting AR model')
# AR+I+MA = ARIMA model

model = ARIMA(forcast["SALES"], order=(2,1,2))

results_ARIMA = model.fit(disp=-1)

plt.plot(forcast["SALES"])

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - forcast["SALES"])**2))

print('Plotting ARIMA model')
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print(predictions_ARIMA_diff.head())
#Convert to cumulative sum

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print(predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(forcast['SALES'].iloc[0], index=forcast.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(forcast["SALES"])

plt.plot(predictions_ARIMA)
results_ARIMA.plot_predict(1,264) 
