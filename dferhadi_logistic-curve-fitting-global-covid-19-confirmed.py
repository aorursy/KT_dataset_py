# Input data files are available in the "../input/" directory.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



global_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

global_data.head()
# This functions smooths data, thanks to Dan Pearson. We will use it to smooth the data for growth factor.

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



def growth_factor(confirmed):

    confirmed_iminus1 = confirmed.shift(1, axis=0)

    confirmed_iminus2 = confirmed.shift(2, axis=0)

    return (confirmed-confirmed_iminus1)/(confirmed_iminus1-confirmed_iminus2)



def growth_ratio(confirmed):

    confirmed_iminus1 = confirmed.shift(1, axis=0)

    return (confirmed/confirmed_iminus1)



# This is a function which plots (for in input country) the active, confirmed, and recovered cases, deaths, and the growth factor.

def plot_country_active_confirmed_recovered(country):

    

    # Plots Active, Confirmed, and Recovered Cases. Also plots deaths.

    country_data = global_data[global_data['Country/Region']==country]

    table = country_data.drop(['Id','Province/State', 'Lat','Long'], axis=1)



    table2 = pd.pivot_table(table, values=['ConfirmedCases','Fatalities'], index=['Date'], aggfunc=np.sum)

    table3 = table2.drop(['Fatalities'], axis=1)

   

    # Growth Factor

    w = 0.5

    table2['GrowthFactor'] = growth_factor(table2['ConfirmedCases'])

    table2['GrowthFactor'] = smoother(table2['GrowthFactor'],w,5)



    # 2nd Derivative

    table2['2nd_Derivative'] = np.gradient(np.gradient(table2['ConfirmedCases'])) #2nd derivative

    table2['2nd_Derivative'] = smoother(table2['2nd_Derivative'],w,7)





    #Plot confirmed[i]/confirmed[i-1], this is called the growth ratio

    table2['GrowthRatio'] = growth_ratio(table2['ConfirmedCases'])

    table2['GrowthRatio'] = smoother(table2['GrowthRatio'],w,5)

    

    #Plot the growth rate, we will define this as k in the logistic function presented at the beginning of this notebook.

    table2['GrowthRate']=np.gradient(np.log(table2['ConfirmedCases']))

    table2['GrowthRate'] = smoother(table2['GrowthRate'],0.5,3)

    

    # horizontal line at growth rate 1.0 for reference

    x_coordinates = [1, 100]

    y_coordinates = [1, 1]

    #plots

    table2['Fatalities'].plot(title='Fatalities')

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





    return 

plot_country_active_confirmed_recovered('China')
plot_country_active_confirmed_recovered('US')

plot_country_active_confirmed_recovered('Germany')
plot_country_active_confirmed_recovered('Italy')
restofworld_data = global_data

for country in restofworld_data['Country/Region']:

    if country != 'China': 

        restofworld_data['Country/Region'] = restofworld_data['Country/Region'].replace(country, "RestOfWorld")



plot_country_active_confirmed_recovered('RestOfWorld')
world_data = global_data



world_data['Country/Region'] = restofworld_data['Country/Region'].replace(country, "World Data")



plot_country_active_confirmed_recovered('World Data')
from scipy.optimize import curve_fit

# We want number of confirmed for each date for each country

#country_data = global_data[global_data['Country/Region']=='Mainland China']

global_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

country_data = global_data[global_data['Country/Region']=='Italy']

country_data = country_data.drop(['Id','Province/State', 'Lat','Long'], axis=1)

country_data = pd.pivot_table(country_data, values=['ConfirmedCases','Fatalities'], index=['Date'], aggfunc=np.sum)

country_data.tail()
#country_data['GrowthFactor'] = growth_factor(country_data['Confirmed'])



# we will want x_data to be the number of days since first confirmed and the y_data to be the confirmed data. This will be the data we use to fit a logistic curve

x_data = range(len(country_data.index))

y_data = country_data['ConfirmedCases']



def log_curve(x, k, x_0, ymax):

    return ymax / (1 + np.exp(-k*(x-x_0)))



# Fit the curve

popt, pcov = curve_fit(log_curve, x_data, y_data, bounds=([0,0,0],np.inf), maxfev=1000)

estimated_k, estimated_x_0, ymax= popt





# Plot the fitted curve

k = estimated_k

x_0 = estimated_x_0

y_fitted = log_curve(x_data, k, x_0, ymax)

print(k, x_0, ymax)

#print(y_fitted)

y_data.tail()
# Plot everything for illustration

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(x_data, y_fitted, '--', label='fitted')

ax.plot(x_data, y_data, 'o', label='Confirmed Data')
