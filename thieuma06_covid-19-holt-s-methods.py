#import library

import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

#import library to create the Neural Network model and evaluate it:

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import accuracy_score,mean_squared_error,mean_squared_log_error , r2_score

from sklearn.preprocessing import StandardScaler 

#import library for Time series 

from statsmodels.tsa.api import  Holt
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

submission = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')



print("Number of Country/Region: ", train['Country/Region'].nunique())

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")
fr_confirmed=train[train['Province/State'] == 'France'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fr_fatalities=train[train['Province/State'] == 'France'].groupby(['Date']).agg({'Fatalities':['sum']})



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

fr_confirmed.plot(ax=ax1)

ax1.set_title("Global confirmed cases in french ", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fr_fatalities.plot(ax=ax2, color='red')

ax2.set_title("Global deceased cases in french", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
#prepare the data

ts=train[train['Province/State'] == 'France']

data=ts['ConfirmedCases'].values

data1=ts['Fatalities'].values



#define the range of date

start_date=ts['Date'].iloc[0]

end_date=ts['Date'].iloc[-1]

index1= pd.date_range(start=start_date, end=end_date, freq='D')



#make a panda data frame series

confirmed = pd.Series(data,index1)

fatali = pd.Series(data1,index1)
fit1 = Holt(confirmed).fit(smoothing_level=0.8, smoothing_slope=0.3, optimized=False)

fcast1 = fit1.forecast(7).rename("Holt's linear trend")





fit2 = Holt(fatali).fit(smoothing_level=0.8, smoothing_slope=0.5, optimized=False)

fcast2 = fit2.forecast(7).rename("Holt's linear trend")





#plot the graph

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))



confirmed.plot(color="black",  ax=ax1)

fit1.fittedvalues.plot(ax=ax1, color='blue')

fcast1.plot(ax=ax1, color='green')





ax1.set_title("French confirmed cases", size=13)

ax1.legend(['Actual cases','holts linear fitted','Predicted cases'], loc='upper left')

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)





fatali.plot(color="black", ax=ax2)

fit2.fittedvalues.plot(ax=ax2, color='blue')

fcast2.plot(ax=ax2, color='green')





ax2.set_title("French fatalities cases", size=13)

ax2.legend(['Actual cases','holts linear fitted','Predicted cases'], loc='upper left')

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)



plt.show()



print("metrics for Confirmed holt's linear:")

print("mean squared log error:",np.sqrt(mean_squared_log_error(confirmed,fit1.fittedvalues)))

print("r2 score:",r2_score(confirmed,fit1.fittedvalues))

print('***************************************************************')

print("metrics for fatalities holt's linear:")

print("mean squared log error:",np.sqrt(mean_squared_log_error(fatali,fit2.fittedvalues)))

print("r2 score:",r2_score(fatali,fit2.fittedvalues))
#prepare the data with the sum for all the china

china_conf=pd.DataFrame(train[train['Country/Region'] == 'China'].groupby(['Date']).agg({'ConfirmedCases':['sum']}))

china_fata=pd.DataFrame(train[train['Country/Region'] == 'China'].groupby(['Date']).agg({'Fatalities':['sum']}))



data3=np.array(china_conf)

data4=np.array(china_fata)



#make a panda data frame series

confirmed_ch = pd.Series(data3[:,0],index1)

fatal_ch = pd.Series(data4[:,0],index1)
fit3 = Holt(confirmed_ch).fit(smoothing_level=0.8, smoothing_slope=0.3, optimized=False)

fcast3 = fit3.forecast(7).rename("Holt's linear trend")





fit4 = Holt(fatal_ch).fit(smoothing_level=0.8, smoothing_slope=0.3, optimized=False)

fcast4 = fit4.forecast(7).rename("Holt's linear trend")





#plot the graph

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))



confirmed_ch.plot(color="black", ax=ax1)

fit3.fittedvalues.plot(ax=ax1, color='blue')

fcast3.plot(ax=ax1, color='green')





ax1.set_title("China confirmed cases", size=13)

ax1.legend(['Actual cases','holts linear fitted','Predicted cases'], loc='lower right')

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)





fatal_ch.plot(color="black", ax=ax2)

fit4.fittedvalues.plot(ax=ax2, color='blue')

fcast4.plot(ax=ax2, color='green')





ax2.set_title("China fatalities cases", size=13)

ax2.legend(['Actual cases','holts linear fitted','Predicted cases'], loc='lower right')

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)



plt.show()



print("metrics for Confirmed holt's linear:")

print("mean squared log error:",np.sqrt(mean_squared_log_error(confirmed_ch,fit3.fittedvalues)))

print("r2 score:",r2_score(confirmed_ch,fit3.fittedvalues))

print('***************************************************************')

print("metrics for fatalities holt's linear:")

print("mean squared log error:",np.sqrt(mean_squared_log_error(fatal_ch,fit4.fittedvalues)))

print("r2 score:",r2_score(fatal_ch,fit4.fittedvalues))
#prepare the data with the sum for all the china

ind_conf=pd.DataFrame(train[train['Country/Region'] == 'India'].groupby(['Date']).agg({'ConfirmedCases':['sum']}))

ind_fata=pd.DataFrame(train[train['Country/Region'] == 'India'].groupby(['Date']).agg({'Fatalities':['sum']}))



data5=np.array(ind_conf)

data6=np.array(ind_fata)



#make a panda data frame series

confirmed_ind = pd.Series(data5[:,0],index1)

fatal_ind = pd.Series(data6[:,0],index1)
fit5 = Holt(confirmed_ind).fit(smoothing_level=0.8, smoothing_slope=0.3, optimized=False)

fcast5 = fit5.forecast(7).rename("Holt's linear trend")





fit6 = Holt(fatal_ind).fit(smoothing_level=0.8, smoothing_slope=0.5, optimized=False)

fcast6 = fit6.forecast(7).rename("Holt's linear trend")





#plot the graph

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))



confirmed_ind.plot(color="black", ax=ax1)

fit5.fittedvalues.plot(ax=ax1, color='blue')

fcast5.plot(ax=ax1, color='green')





ax1.set_title("India confirmed cases", size=13)

ax1.legend(['Actual cases','holts linear fitted','Predicted cases'], loc='upper left')

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)





fatal_ind.plot(color="black", ax=ax2)

fit6.fittedvalues.plot(ax=ax2, color='blue')

fcast6.plot(ax=ax2, color='green')





ax2.set_title("India fatalities cases", size=13)

ax2.legend(['Actual cases','holts linear fitted','Predicted cases'], loc='upper left')

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)



plt.show()



print("metrics for Confirmed holt's linear:")

print("mean squared log error:",np.sqrt(mean_squared_log_error(confirmed_ind,fit5.fittedvalues)))

print("r2 score:",r2_score(confirmed_ind,fit5.fittedvalues))

print('***************************************************************')

print("metrics for fatalities holt's linear:")

print("mean squared log error:",np.sqrt(mean_squared_log_error(fatal_ind,fit6.fittedvalues)))

print("r2 score:",r2_score(fatal_ind,fit6.fittedvalues))