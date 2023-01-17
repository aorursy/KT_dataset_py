import pandas as pd

import numpy as np

from datetime import datetime

ts=pd.read_csv("../input/MonthlyPassengers.csv",parse_dates=["Month"],index_col="Month")
ts.info()
ts.head()
# Rename column name to Passenger

ts=ts.rename(columns={"#Passengers":"Passenger"})
ts.head()
import matplotlib.pyplot as plt

ts.Passenger.resample("M").mean().plot()



from statsmodels.graphics.tsaplots import plot_acf
plot_acf(ts)
ts_diff=ts.diff(periods=1)

ts_diff=ts_diff[1:]
plot_acf(ts_diff)



# now looking at below visualization, data seems to be converted to stationary data.
import matplotlib.pyplot as plt

ts_diff.Passenger.resample("M").mean().plot()
# Determine rolling statistics

rolmean=ts.rolling(window=12).mean()

rolstd=ts.rolling(window=12).std()

print(rolmean)

print(rolstd)





# plot rolling statistics

oringdnal=plt.plot(ts,color="blue",label="orignal")

mean=plt.plot(rolmean,color="black",label="rolmean")

std=plt.plot(rolstd,color="red",label="rolstd")

plt.legend(loc='best')

# Perform dickey fuller test



from statsmodels.tsa.stattools import adfuller



dftest = adfuller(ts["Passenger"],autolag='AIC')

print(dftest)

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

print(dfoutput)

#critical values



dftest[4]
ts_log=np.log(ts)

plt.plot(ts_log)

ts_log.head()
# now applying moving avg. technique to soothining the tend 



movavg=ts_log.rolling(window=12).mean()

movstd=ts_log.rolling(window=12).std()

oringdnal=plt.plot(ts_log,color="blue",label="orignal")

mean=plt.plot(movavg,color="black",label="rolmean")

std=plt.plot(movstd,color="red",label="rolstd")

plt.legend(loc='best')



# we can see that data is not stationary
logdiffmov=ts_log-movavg

plt.plot(logdiffmov)

logdiffmov.head()
logdiffmov.dropna(inplace=True)
logdiffmov.head()
movavg1=logdiffmov.rolling(window=12).mean()

movstd1=logdiffmov.rolling(window=12).std()

oringdnal=plt.plot(logdiffmov,color="blue",label="orignal")

mean=plt.plot(movavg1,color="black",label="movavg1")

std=plt.plot(movstd1,color="red",label="movstd1")

plt.legend(loc='best')
from statsmodels.tsa.stattools import adfuller



dftest1 = adfuller(logdiffmov["Passenger"],autolag='AIC')

print(dftest1)

dfoutput1 = pd.Series(dftest1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

print(dfoutput1)


#ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(logdiffmov, nlags=20)

lag_pacf = pacf(logdiffmov, nlags=20, method='ols')
# To determine value of Q when chart crosses the upper interval level for the fist time

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(logdiffmov)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(logdiffmov)),linestyle='--',color='gray')



# value of P = 2

# To determine value of P when chart crosses the upper interval level for the fist time



plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='green')

plt.axhline(y=-1.96/np.sqrt(len(logdiffmov)),linestyle='--',color='green')

plt.axhline(y=1.96/np.sqrt(len(logdiffmov)),linestyle='--',color='green')



#Value of P =2
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log, order=(2, 1, 2))  

results_ARIMA = model.fit()  

plt.plot(logdiffmov)

plt.plot(results_ARIMA.fittedvalues, color='red')
series_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy=True)

print(series_ARIMA.head())
# the above series is not showing the first month so lets find cumulative sum



series_ARIMA_cumsum = series_ARIMA.cumsum()

series_ARIMA_cumsum.head()
predictions_ARIMA_log = pd.Series(ts_log["Passenger"].iloc[0], index=ts_log.index)

series_predictions_ARIMA_log = predictions_ARIMA_log.add(series_ARIMA_cumsum,fill_value=0)

series_predictions_ARIMA_log.head()
# now lets make data as original and do compare by make a plot



predictions_orignal_data = np.exp(series_predictions_ARIMA_log)
predictions_orignal_data.head()
plt.plot(predictions_orignal_data)

plt.plot(ts)
ts_log.shape
results_ARIMA.plot_predict(1,156) 
final_data=results_ARIMA.forecast(steps=24)[0]
final_data
passenger_projection= np.exp(final_data)
passenger_projection
passengers = pd.DataFrame({"Proj_passenger":[443.87099559, 470.03826902, 504.93502836, 540.45485906,

       567.73073473, 580.32812271, 577.00266573, 561.93886205,

       542.38464104, 525.56673272, 516.65911974, 518.14327442,

       529.9371715 , 549.71679011, 573.36070617, 595.78903063,

       612.33154129, 620.24253267, 619.59555149, 613.03677623,

       604.59668575, 598.2664506 , 596.94854054, 601.96248871]})
passengers
passengers.info()
#now lets create date range of projected 24 months and put passanger data infront of each month.
rng=pd.date_range(start="1/1/1961",periods=24,freq="M")
rng
Month = pd.DataFrame({"proj_month":['1961-01-31', '1961-02-28', '1961-03-31', '1961-04-30',

               '1961-05-31', '1961-06-30', '1961-07-31', '1961-08-31',

               '1961-09-30', '1961-10-31', '1961-11-30', '1961-12-31',

               '1962-01-31', '1962-02-28', '1962-03-31', '1962-04-30',

               '1962-05-31', '1962-06-30', '1962-07-31', '1962-08-31',

               '1962-09-30', '1962-10-31', '1962-11-30', '1962-12-31']})

print(Month)

Month.info()