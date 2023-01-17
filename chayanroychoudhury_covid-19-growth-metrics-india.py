import scipy as sc

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

!pip install lmfit

import datetime

import os

os.getcwd()

global_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

global_data.head()
country='India'

country_data = global_data[global_data['Country/Region']==country] # Chooses rows with your country data.

table = country_data.drop(['SNo','Province/State', 'Last Update'], axis=1) # drops the unecessary columns



table['ActiveCases'] = table['Confirmed'] - (table['Recovered'] + table['Deaths']) # Calculates the active cases 

table2 = pd.pivot_table(table, values=['ActiveCases','Confirmed', 'Recovered','Deaths'],\

                        index=pd.to_datetime(country_data['ObservationDate']), aggfunc=np.sum) # creates a pivote table based on ObsDates

table3 = table2.drop(['Deaths'], axis=1) #drops the death columns for later use.

table2.head()
def growth_factor(confirmed):

    confirmed_iminus1 = confirmed.shift(1, axis=0)

    confirmed_iminus2 = confirmed.shift(2, axis=0)

    return (confirmed-confirmed_iminus1)/(confirmed_iminus1-confirmed_iminus2)



def growth_ratio(confirmed):

    confirmed_iminus1 = confirmed.shift(1, axis=0)

    return (confirmed/confirmed_iminus1)



def smoother(inputdata,w,imax):

    data = 1.0*inputdata

    data = data.replace(np.nan,1)

    data = data.replace(np.inf,1)

    #print(data)

    smoothed = 1.0*data

    normalization = 1

    for i in range(-imax,imax+1):

        if i==0:

            continue

        smoothed += (w**abs(i))*data.shift(i,axis=0)

        normalization += w**abs(i)

    smoothed /= normalization

    return smoothed
w=0.5

table2['GrowthFactor'] = growth_factor(table2['Confirmed'])

table2['GrowthFactor'] = smoother(table2['GrowthFactor'],w,5)



    # 2nd Derivative

table2['2nd_Derivative'] = np.gradient(np.gradient(table2['Confirmed'])) #2nd derivative

table2['2nd_Derivative'] = smoother(table2['2nd_Derivative'],w,7)





    #Plot confirmed[i]/confirmed[i-1], this is called the growth ratio

table2['GrowthRatio'] = growth_ratio(table2['Confirmed'])

table2['GrowthRatio'] = smoother(table2['GrowthRatio'],w,5)

    

    #Plot the growth rate, we will define this as k in the logistic function presented at the beginning of this notebook.

table2['GrowthRate']=np.gradient(np.log(table2['Confirmed']))

table2['GrowthRate'] = smoother(table2['GrowthRate'],0.5,3)
# horizontal line at growth rate 1.0 for reference

x_coordinates = [1, 100]

print(x_coordinates)

y_coordinates = [1, 1]

#plots

table2['Deaths'].plot(title='Deaths')

plt.show()

table3.plot() 

plt.show()

table2['GrowthFactor'].plot(title='Growth Factor')

plt.plot(x_coordinates, y_coordinates) 

plt.show()

table2['2nd_Derivative'].plot(title='2nd_Derivative')

plt.show()

table2['GrowthRatio'].plot(title='Growth Ratio')

plt.plot(x_coordinates, y_coordinates)

plt.show()

table2['GrowthRate'].plot(title='Growth Rate')

plt.show()
table2['GrowthFactor'] = (growth_factor(table2['Confirmed']))

table2['GrowthFactor'] = table2['GrowthFactor'].rolling(7).mean()



    # 2nd Derivative

table2['2nd_Derivative'] = np.gradient(np.gradient(table2['Confirmed'])) #2nd derivative

table2['2nd_Derivative'] = table2['2nd_Derivative'].rolling(7).mean()





    #Plot confirmed[i]/confirmed[i-1], this is called the growth ratio

table2['GrowthRatio'] = growth_ratio(table2['Confirmed'])

table2['GrowthRatio'] = table2['GrowthRatio'].rolling(7).mean()

    

    #Plot the growth rate, we will define this as k in the logistic function presented at the beginning of this notebook.

table2['GrowthRate']= np.gradient(np.log(table2['Confirmed']))

table2['GrowthRate'] = table2['GrowthRate'].rolling(7).mean()
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(15, 5))

fig.subplots_adjust(hspace=0.5)

axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

axes[0,0].plot(table2.index,table2['Deaths'])

axes[0,0].set_title('Deaths')



axes[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

axes[0,1].plot(table2.index,table2['GrowthFactor'])

axes[0,1].set_title('Growth Factor')

axes[0,1].axhline(1,c='red',ls='--')



axes[0,2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

axes[0,2].plot(table2.index,table2['2nd_Derivative'])

axes[0,2].set_title('2nd_Derivative')

axes[0,2].axhline(0,c='red',ls='--')



axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

axes[1,0].plot(table2.index,table2['GrowthRatio'])

axes[1,0].set_title('GrowthRatio')

axes[1,0].axhline(1,c='red',ls='--')

fig.autofmt_xdate()

axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

axes[1,1].plot(table2.index,table2['GrowthRate'])

axes[1,1].set_title('Growth Rate')

axes[1,1].axhline(0,c='red',ls='--')

fig.autofmt_xdate()

axes[1,2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

table3.reset_index()

table3.plot(ax=axes[1,2])

axes[1,2].set_title('Active, Recovered and Confirmed Cases')



fig.autofmt_xdate()

plt.show()
from scipy.ndimage.filters import gaussian_filter1d



table2['GrowthFactor'] = gaussian_filter1d((growth_factor(table2['Confirmed'])),sigma=1)

    # 2nd Derivative

table2['2nd_Derivative'] = gaussian_filter1d(np.gradient(np.gradient(table2['Confirmed'])),sigma=1)



    #Plot confirmed[i]/confirmed[i-1], this is called the growth ratio

table2['GrowthRatio'] = gaussian_filter1d(growth_ratio(table2['Confirmed']),sigma=1)

    

    #Plot the growth rate, we will define this as k in the logistic function presented at the beginning of this notebook.

table2['GrowthRate']= gaussian_filter1d(np.gradient(np.log(table2['Confirmed'])),sigma=1)

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(15, 5))

fig.subplots_adjust(hspace=0.5)

axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

axes[0,0].plot(table2.index,table2['Deaths'])

axes[0,0].set_title('Deaths')



axes[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

axes[0,1].plot(table2.index,table2['GrowthFactor'])

axes[0,1].set_title('Growth Factor')

axes[0,1].axhline(1,c='red',ls='--')



axes[0,2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

axes[0,2].plot(table2.index,table2['2nd_Derivative'])

axes[0,2].set_title('2nd_Derivative')

axes[0,2].axhline(0,c='red',ls='--')



axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

axes[1,0].plot(table2.index,table2['GrowthRatio'])

axes[1,0].set_title('GrowthRatio')

axes[1,0].axhline(1,c='red',ls='--')

fig.autofmt_xdate()

axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

axes[1,1].plot(table2.index,table2['GrowthRate'])

axes[1,1].set_title('Growth Rate')

axes[1,1].axhline(0,c='red',ls='--')

fig.autofmt_xdate()

axes[1,2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

table3.reset_index()

table3.plot(ax=axes[1,2])

axes[1,2].set_title('Active, Recovered and Confirmed Cases')



fig.autofmt_xdate()

plt.show()
from scipy.optimize import curve_fit # TO FINALLY FIT THE LOGISTIC FUNCTION


table2.tail()

dates = (table2.index.astype(str))

dates = [datetime.datetime.strptime(date,"%Y-%m-%d") for date in dates] # FOR GETTING THE DATES
# X data will be number of days since first confirmed

# Y data to be no. of confirmed.

x = np.arange(0,len(country_data.index))

y = country_data['Confirmed']

x.shape, y.shape
# Define the logistic function

def log_curve(x, k, x_0, Nmax):

    return Nmax / (1 + np.exp(-k*(x-x_0)))
# Fit the curve

popt, pcov = curve_fit(log_curve, x, y, bounds=([0,0,0],np.inf), maxfev=10000)

estimated_k, estimated_x_0, Nmax= popt
# Plot the fitted curve

k = estimated_k

x_0 = estimated_x_0

y_fitted = log_curve(x, k, x_0, Nmax)

print(f'The estimated parameters are \t k={k} \t x0={x_0} \t Nmax={Nmax}')

print("Inflection point should be around",np.ceil(x_0),'days and no of cases should max out at',np.ceil(Nmax))

inflection_Date = dates[0]+datetime.timedelta(days=np.int(np.ceil(64)))

print('The inflection date is ', inflection_Date.strftime('%d %B %Y'))

from lmfit import Model



logmodel = Model(log_curve)

print('parameter names: {}'.format(logmodel.param_names))

print('independent variables: {}\n'.format(logmodel.independent_vars))



y_fittedd = logmodel.fit(y,x=x, k=estimated_k,x_0=estimated_x_0,Nmax=Nmax)

print(y_fittedd.fit_report()) 

print('-------------------..------------------')

 #CHeck the parameters here for their best estimate
y_fittedd.params.keys()
# Acc to lmfit, your best estimate for inflection point should be,

print('Your best estimate for inflection point is about ',y_fittedd.params['x_0'])

print("\n and the cases should max out at ", y_fittedd.params['Nmax'])
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

ax.plot(x, y_fitted, 'b--', label='fitted by scipy_curvefit', lw=1.5)

ax.scatter(x,y_fittedd.best_fit , c='green', marker='+', label='fitted by lmfit', s=40)

ax.scatter(x, y, color = 'red', label='Confirmed Data', alpha=0.6)

ax.scatter(x,y, c='b', visible=False, label=country)

ax.legend()

ax.set_xlabel('No. of Days')

ax.set_ylabel('No. of cases')