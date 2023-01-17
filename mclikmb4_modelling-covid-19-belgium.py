import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

from datetime import date

df_national = pd.read_csv( "../input/covid19-belgium/covid19-belgium.csv")

df_nl = pd.read_csv( "../input/covid19-belgium/covid19-netherlands.csv",  parse_dates=['Datum'])

df_italy = pd.read_csv( "../input/coronavirusdataset-france/contagioitalia.csv", parse_dates=['date'])

#import dateparser

#dateparser.parse(df_national['date'], settings={'DATE_ORDER': 'DMY'})



df_national['date'] = pd.to_datetime(df_national['date'], format = "%Y/%m/%d")
df_national.rename(columns={'cumul_cases':'cases'},inplace=True) 

df_national.tail()
latest_date = max(df_national['date'])

print(latest_date)

national_latest = df_national[df_national['date'] == latest_date]



df_nl = df_nl[df_nl['Type'] == 'Totaal']

df_nl.rename(columns={'Aantal':'cases', 'Datum':'date'},inplace=True)

df_nl = df_nl.reset_index()

df_nl= df_nl[['date','cases']]

y = df_national['cases'].values # transform the column to differentiate into a numpy array



deriv_y = np.gradient(y) # now we can get the derivative as a new numpy array



output = np.transpose(deriv_y)

#now add the numpy array to our dataframe

df_national['ContagionRate'] = pd.Series(output)

df_national.to_csv('contagiobelgio.csv')

df_national['%hospitalized'] = (df_national['hospitalized']/df_national['cases'])*100

df_national['%dead'] = (df_national['cumul_deceased']/df_national['cases'])*100

df_national['%released'] = (df_national['cumul_released']/df_national['cases'])*100
#national data fit

from scipy.optimize import curve_fit

from numpy import exp, linspace, random

from math import pi

# build an extrapolated gaussian based on italian data fit

def gaussian(x, amp, cen, wid):

    """1-d gaussian: gaussian(x, amp, cen, wid)"""

    return (amp / (np.sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2)) 

def gauss_function(X, amp, cen, sigma):

    return amp*exp(-(X-cen)**2/(2*sigma**2))



#belgian data fit

x1 = df_national.index.values

y1 = (df_national['ContagionRate'].values)



init_vals1 = [2000, 30, 200]  # for [amp, cen, wid]

best_vals1, covar1 = curve_fit(gaussian, x1, y1, p0=init_vals1)

print('best_vals1: {}'.format(best_vals1))
# extrapolated gaussian



x_e = np.arange(0, 70)

timerange = pd.date_range(start='3/1/2020', periods=70)

y_e = gauss_function(x_e, 2500,best_vals1[1],best_vals1[2])

#plt.plot(timerange,y_e)

#plt.xticks(rotation=90)
dummy = np.zeros(70)

plt.figure(figsize= (12,12))

plt.subplot(321)

plt.plot(df_national['date'],df_national['cases'], color = 'c') #trend cases

plt.plot(timerange,dummy, ':', color = 'w') 

plt.title('Cases over time')

plt.ylabel('number of cases')

plt.xticks(df_national['date']," ")



plt.subplot(323)

plt.plot(timerange,y_e, '--', color = 'orange') 

plt.plot(df_national['date'],df_national['daily_cases'], color = 'r') 

plt.title('Confirmed cases rate')

plt.ylabel('Rate/daily cases')

plt.xticks(rotation=90)



plt.subplot(322)

plt.plot(df_national['date'],df_national['%hospitalized'], color = 'b') #trend cases

plt.title('Hospitalized over time')

plt.ylabel('Hospitalized(%)')

plt.xticks(df_national['date']," ")





plt.subplot(324)

plt.plot(df_national['date'],df_national['%released'], color = 'g', label = 'Released') 

plt.plot(df_national['date'],df_national['%dead'], color = 'k', label = 'Dead') 

plt.title('Mortality and recovery')

plt.ylabel('Percentages over total cases')

plt.legend()

plt.xticks(rotation=90)

plt.suptitle('Covid national stats - Belgium')

plt.show()
population_italy = 60488373

population_belgium = 11585253

population_netherlands = 17126567

populratio = population_italy/population_belgium

populratio_nl = population_italy/population_netherlands
y_it = df_italy['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_y_it = np.gradient(y_it) # now we can get the derivative as a new numpy array

#np.savetxt("contagioitalia.csv", deriv_y, delimiter=",")

output_it = np.transpose(deriv_y_it)

#now add the numpy array to our dataframe

df_italy['ContagionRate'] = pd.Series(output_it)



x = df_italy.index.values

y = df_italy['ContagionRate'].values



init_vals = [40, 35, 60]  # for [amp, cen, wid]

best_vals, covar = curve_fit(gaussian, x, y, p0=init_vals)

print('best_vals: {}'.format(best_vals))
#belgian data fit normalized to population ratio Italy/Belgium

x1 = df_national.index.values

y1 = (df_national['ContagionRate'].values)*populratio



init_vals1 = [8000, 33, 200]  # for [amp, cen, wid]

best_vals1, covar1 = curve_fit(gaussian, x1, y1, p0=init_vals1)

print('best_vals1: {}'.format(best_vals1))
# extrapolated gaussian

timeframe_days = 80

x_it= np.arange(0, timeframe_days)

y_it = gauss_function(x_it, 6000,33,best_vals[2])

y_be = gauss_function(x_it, 8000,best_vals1[1],best_vals1[2])

plt.plot(x_it, y_it, label = 'Model Italy')

plt.plot(x_it, y_be, label = 'Model Belgium')

plt.xlabel('Days', fontsize=14)

plt.ylabel('Rates (a.u.)', fontsize=14)

plt.legend()

plt.title('Modelled curves for Covid-19 spread over time')
plt.figure(figsize=(12, 10))

plt.subplot(221)

plt.plot(df_national.index,df_national['cases'], label = 'Belgium') #trend cases

plt.plot(df_italy.index,df_italy['TotalPositiveCases'], label = 'Italy') #trend cases

plt.plot(df_nl.index,df_nl['cases'], label = 'Netherlands') #trend cases

plt.title('International comparison of cases growth', fontsize = 20)

plt.xlabel('Days', fontsize=14)

plt.ylabel('Num. cases', fontsize=14)





plt.subplot(222)

plt.plot(df_national.index,(df_national['cases']/population_belgium)*100, label = 'Belgium') #trend cases

plt.plot(df_italy.index,(df_italy['TotalPositiveCases']/population_italy)*100, label = 'Italy') #trend cases

plt.plot(df_nl.index,(df_nl['cases']/population_netherlands)*100, label = 'Netherlands') #trend cases

plt.xlabel('Days', fontsize=14)

plt.yscale('log')

plt.ylabel('Log %cases over population total', fontsize=14)





plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.figure(figsize= (6,6))

#plot the fit results



#plt.plot(X1,gauss_function(X1, *popt1), ':', label = 'Italy-modelled gaussian')



#confront the given data

plt.plot(df_national.index,df_national['ContagionRate']*(populratio), label = 'Belgium (normalized)') #trend cases

plt.plot(df_italy.index,df_italy['ContagionRate'],color='g', label = 'Italy') #trend cases

# with models

plt.plot(x_it, y_it, ':',color='r', label = 'Italy-modelled gaussian')

plt.plot(x_it, y_be, ':',color='orange', label = 'Belgium-modelled gaussian')



# set timeframes

plt.axvline(x=33 , color='k', linewidth = 0.5)

plt.axvline(x=59, ymin=0.05, ymax=0.2, color='k', linewidth = 0.5)

plt.text(32, 9000, ' Italy: 2020-03-27\n Belgium: 2020-04-03')

plt.text(49, 2000, ' Italy: 2020-04-25\n Belgium: 2020-05-02')

plt.title('Cases over time')

plt.ylabel('Spread rate')



plt.xticks(rotation=90)

plt.xlim(0, 80)

plt.ylabel('Contagion rate (first derivative of cases count)', fontsize=14)

plt.xlabel('Days', fontsize = 14)

plt.legend



plt.title('International comparison of spread rate', fontsize = 20)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)