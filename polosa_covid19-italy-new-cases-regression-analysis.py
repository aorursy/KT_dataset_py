import numpy as np 

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.optimize import curve_fit

import datetime

import operator 

plt.style.use('fivethirtyeight')

%matplotlib inline 
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-02-2020.csv')





cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]

recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()

world_cases = []

total_deaths = [] 

mortality_rate = []

recovery_rate = [] 

total_recovered = [] 

total_active = [] 



# Nations confirmed

china_cases = [] 

italy_cases = []

us_cases = [] 

spain_cases = [] 

france_cases = [] 

germany_cases = [] 

uk_cases = [] 



for i in dates:

    confirmed_sum = confirmed[i].sum()

    death_sum = deaths[i].sum()

    recovered_sum = recoveries[i].sum()

    

    # confirmed, deaths, recovered, and active

    world_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    total_recovered.append(recovered_sum)

    total_active.append(confirmed_sum-death_sum-recovered_sum)

    

    # calculate rates

    mortality_rate.append(death_sum/confirmed_sum)

    recovery_rate.append(recovered_sum/confirmed_sum)



    # case studies 

    china_cases.append(confirmed_df[confirmed_df['Country/Region']=='China'][i].sum())

    italy_cases.append(confirmed_df[confirmed_df['Country/Region']=='Italy'][i].sum())

    us_cases.append(confirmed_df[confirmed_df['Country/Region']=='US'][i].sum())

    spain_cases.append(confirmed_df[confirmed_df['Country/Region']=='Spain'][i].sum())

    france_cases.append(confirmed_df[confirmed_df['Country/Region']=='France'][i].sum())

    germany_cases.append(confirmed_df[confirmed_df['Country/Region']=='Germany'][i].sum())

    uk_cases.append(confirmed_df[confirmed_df['Country/Region']=='United Kingdom'][i].sum())

    

italy_cases[50] = 15113
def daily_increase(data):

    d = [] 

    for i in range(len(data)):

        if i == 0:

            d.append(data[0])

        else:

            d.append(data[i]-data[i-1])

    return d 



world_daily_increase = daily_increase(world_cases)

china_daily_increase = daily_increase(china_cases)

italy_daily_increase = daily_increase(italy_cases)

us_daily_increase = daily_increase(us_cases)

spain_daily_increase = daily_increase(spain_cases)

france_daily_increase = daily_increase(france_cases)

germany_daily_increase = daily_increase(germany_cases)

uk_daily_increase = daily_increase(uk_cases)

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)

total_deaths = np.array(total_deaths).reshape(-1, 1)

total_recovered = np.array(total_recovered).reshape(-1, 1)
days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-10]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
adjusted_dates = adjusted_dates.reshape(1, -1)[0]

italy_daily_increase_pct = pd.Series(italy_cases).pct_change()

france_daily_increase_pct = pd.Series(france_cases).pct_change()

spain_daily_increase_pct = pd.Series(spain_cases).pct_change()

us_daily_increase_pct = pd.Series(us_cases).pct_change()

germany_daily_increase_pct = pd.Series(germany_cases).pct_change()

uk_daily_increase_pct = pd.Series(uk_cases).pct_change()



#italy_cases, france_cases



def func1(x, a, b):

    return a*np.log(x-32)+b



def func2(x, b2, c2):

    return np.exp(b2*(x-32))+c2



def func3(x, b3, c3):

    return b3/x+c3



def func4(x, b4, c4):

    return b4*x+c4





Y1 = italy_daily_increase_pct.iloc[33:]

X1 = adjusted_dates[33:]# np.arange(1,Y1.shape[0]+1)



popt1, pcov1 = curve_fit(func1, X1, Y1.to_numpy()+0.0001, maxfev=100000)

residuals1 = Y1.to_numpy()+0.0001 - func1(X1, *popt1)

perr1 = np.sum(residuals1**2)





popt2, pcov2 = curve_fit(func2, X1, Y1.to_numpy()+0.0001, maxfev=100000)

perr2 = np.sqrt(np.diag(pcov2))

residuals2 = Y1.to_numpy()+0.0001 - func2(X1, *popt2)

perr2 = np.sum(residuals2**2)



popt3, pcov3 = curve_fit(func3, X1, Y1.to_numpy()+0.0001, maxfev=100000)

perr3 = np.sqrt(np.diag(pcov3))

residuals3 = Y1.to_numpy()+0.0001 - func3(X1, *popt3)

perr3 = np.sum(residuals3**2)



popt4, pcov4 = curve_fit(func4, X1, Y1.to_numpy()+0.0001, maxfev=100000)

perr4 = np.sqrt(np.diag(pcov4))

residuals4 = Y1.to_numpy()+0.0001 - func4(X1, *popt4)

perr4 = np.sum(residuals4**2)



pdate = pd.to_datetime(dates[33:])



d = pdate.strftime('%m/%d')

#print (d.append(pd.Index(['05/04'])))

plt.figure(figsize=(16, 13))

plt.plot(d, Y1, linewidth=4) #Italy data



plt.plot(d, func1(X1, *popt1), '--', label = "LOG REG", linewidth=3 )

plt.plot(d, func2(X1, *popt2), ':', label = "EXP REG", linewidth=3 )

plt.plot(d, func3(X1, *popt3), '-.', label = "INV REG", linewidth=3 )

plt.plot(d, func4(X1, *popt4), linestyle=(0, (1, 1)), label = "LIN REG", linewidth=3 )



plt.xticks(rotation=60)

#plt.plot(france_daily_increase_pct[33:].rolling(6).mean())

#plt.plot(spain_daily_increase_pct.iloc[33:].rolling(6).mean())

#plt.plot(us_daily_increase_pct[33:].rolling(6).mean())

plt.title('ITALY - Daily Increases % in Confirmed Cases', size=30)

plt.xlabel('Days', size=20)

plt.ylabel('% of Cases', size=20)

plt.xticks(size=14)

plt.yticks(size=14)

plt.legend(['Italy', 'Log. ' + "ε={:1.4f}".format(perr1),'Exp. '+ "ε={:1.4f}".format(perr2),'Inv. '+ "ε={:1.4f}".format(perr3),'Lin. '+ "ε={:1.4f}".format(perr4)], prop={'size': 13})



#plt.text(18.5, 0.5, s, horizontalalignment='right', verticalalignment='top', fontsize=15)

plt.show()

a = pd.date_range(pdate[-1] + pd.offsets.DateOffset(1), periods=30)

d2 = pdate.append(a)



X2 = np.append(X1,np.arange(X1[-1]+1,X1[-1]+31))



plt.figure(figsize=(16, 13))



plt.plot(d2, func1(X2, *popt1), '--', label = "LOG REG", linewidth=3 )

plt.plot(d2, func2(X2, *popt2), ':', label = "EXP REG", linewidth=3 )

plt.plot(d2, func3(X2, *popt3), '-.', label = "INV REG", linewidth=3 )

plt.plot(d2, func4(X2, *popt4), linestyle=(0, (1, 1)), label = "LIN REG", linewidth=3 )



plt.xticks(rotation=60)

#plt.plot(france_daily_increase_pct[33:].rolling(6).mean())

#plt.plot(spain_daily_increase_pct.iloc[33:].rolling(6).mean())

#plt.plot(us_daily_increase_pct[33:].rolling(6).mean())

plt.title('Predict Daily Increases % in Confirmed Cases', size=30)

plt.xlabel('Days', size=20)

plt.ylabel('% of Cases', size=20)

plt.xticks(size=14)

plt.yticks(size=14)

plt.legend(['Log. ' + "ε={:1.4f}".format(perr1),'Exp. '+ "ε={:1.4f}".format(perr2),'Inv. '+ "ε={:1.4f}".format(perr3),'Lin. '+ "ε={:1.4f}".format(perr4)], prop={'size': 13})



#plt.text(18.5, 0.5, s, horizontalalignment='right', verticalalignment='top', fontsize=15)

plt.show()





Y1 = france_daily_increase_pct.iloc[40:]

X1 = adjusted_dates[40:]# np.arange(1,Y1.shape[0]+1)



popt1, pcov1 = curve_fit(func1, X1, Y1.to_numpy()+0.0001, maxfev=100000)

residuals1 = Y1.to_numpy()+0.0001 - func1(X1, *popt1)

perr1 = np.sum(residuals1**2)





popt2, pcov2 = curve_fit(func2, X1, Y1.to_numpy()+0.0001, maxfev=100000)

perr2 = np.sqrt(np.diag(pcov2))

residuals2 = Y1.to_numpy()+0.0001 - func2(X1, *popt2)

perr2 = np.sum(residuals2**2)



popt3, pcov3 = curve_fit(func3, X1, Y1.to_numpy()+0.0001, maxfev=100000)

perr3 = np.sqrt(np.diag(pcov3))

residuals3 = Y1.to_numpy()+0.0001 - func3(X1, *popt3)

perr3 = np.sum(residuals3**2)



popt4, pcov4 = curve_fit(func4, X1, Y1.to_numpy()+0.0001, maxfev=100000)

perr4 = np.sqrt(np.diag(pcov4))

residuals4 = Y1.to_numpy()+0.0001 - func4(X1, *popt4)

perr4 = np.sum(residuals4**2)



pdate = pd.to_datetime(dates[40:])



d = pdate.strftime('%m/%d')

#print (d.append(pd.Index(['05/04'])))

plt.figure(figsize=(16, 13))

plt.plot(d, Y1, linewidth=4) #france data



plt.plot(d, func1(X1, *popt1), '--', label = "LOG REG", linewidth=3 )

plt.plot(d, func2(X1, *popt2), ':', label = "EXP REG", linewidth=3 )

plt.plot(d, func3(X1, *popt3), '-.', label = "INV REG", linewidth=3 )

plt.plot(d, func4(X1, *popt4), linestyle=(0, (1, 1)), label = "LIN REG", linewidth=3 )



plt.xticks(rotation=60)

#plt.plot(france_daily_increase_pct[40:].rolling(6).mean())

#plt.plot(spain_daily_increase_pct.iloc[40:].rolling(6).mean())

#plt.plot(us_daily_increase_pct[40:].rolling(6).mean())

plt.title('FRANCE - Daily Increases % in Confirmed Cases', size=30)

plt.xlabel('Days', size=20)

plt.ylabel('% of Cases', size=20)

plt.xticks(size=14)

plt.yticks(size=14)

plt.legend(['france', 'Log. ' + "ε={:1.4f}".format(perr1),'Exp. '+ "ε={:1.4f}".format(perr2),'Inv. '+ "ε={:1.4f}".format(perr3),'Lin. '+ "ε={:1.4f}".format(perr4)], prop={'size': 13})



#plt.text(18.5, 0.5, s, horizontalalignment='right', verticalalignment='top', fontsize=15)

plt.show()

plt.figure(figsize=(16, 9))

plt.plot(italy_daily_increase_pct.iloc[33:].rolling(6).mean())

plt.plot(france_daily_increase_pct[33:].rolling(6).mean())

plt.plot(spain_daily_increase_pct.iloc[33:].rolling(6).mean())

plt.plot(us_daily_increase_pct[33:].rolling(6).mean())

plt.plot(germany_daily_increase_pct.iloc[33:].rolling(6).mean())

plt.plot(uk_daily_increase_pct.iloc[33:].rolling(6).mean())

plt.title('Daily Increases % in Confirmed Cases', size=30)

plt.xlabel('Days', size=14)

plt.ylabel('% of Cases', size=14)

plt.xticks(size=20)

plt.yticks(size=20)

plt.legend(['Italy', 'France','Spain','US','Germany','UK'], prop={'size':14})

plt.show()