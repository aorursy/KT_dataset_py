# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# basic packages

import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime

from scipy.optimize import curve_fit

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import all of them

confirmed_df = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/2019-novel-coronavirus-covid-19-2019-ncov-data-repository-confirmed-cases.csv')

death_df = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/2019-novel-coronavirus-covid-19-2019-ncov-data-repository-deaths.csv')

recovered_df = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/2019-novel-coronavirus-covid-19-2019-ncov-data-repository-recovered.csv')
# this notebook is about how covid-19 infection increase in india



# filter out india cases

df1 = confirmed_df[confirmed_df['country_region']=='India']

df2 = death_df[death_df['country_region']=='India']

df3 = recovered_df[recovered_df['country_region']=='India']



# get the column name

india_confirmed = df1['confirmed'].values.tolist()

india_death = df2['deaths'].values.tolist()

india_recovered = df3['recovered'].values.tolist()



# get date column

dates = df1['date']

dates = list(pd.to_datetime(dates))



# get date from 8th index because in india cases comes from 8th indexx

dates = dates[8:]           

india_confirmed = india_confirmed[8:]    

india_death = india_death[8:]

india_recovered = india_recovered[8:]
# we implement how confirmed cases grow, means we find growth_factor, growth_rate

# growth_factor = means on day N+1 is the number of confirmed cases on day N+1 minus confirmed cases on day N divided by the number of the confirmed cases on day N minus confirmed cases on day N-1. if india growth factor has stabilize  around 1 then this can be a sign that india reached it's infecton point.

# growth_ratio = means on day N+1 is the number of confirmed cases on day N divided by the number of confirmed cases on day N.

# growth_rate = first derivative means which rate it is growing at a point.





a = []        #growth ratio

b = []        #for implementing growth factor 

growth_factor = []

growth_rate = []





for i in range(1,len(india_confirmed)-1):

    a.append(india_confirmed[i+1]-india_confirmed[i])

    b.append(india_confirmed[i]-india_confirmed[i-1])

    

for n, i in enumerate(a):

    if i == 0:

       a[n] = 1



for n, i in enumerate(b):

    if i == 0:

       b[n] = 1





growth_factor = []

for i in range(len(a)):

    growth_factor.append(a[i]/b[i])









growth_factor.insert(0, 1)              #fill the (i-1)th and ith elements of dataframe

growth_factor.insert(1, 1)          

a.insert(0, 1)

a.insert(1, 1)

# create new dataframe 

df4 = pd.DataFrame()



# and creaate new column for growth 

df4['days_count'] = list(range(1, len(dates)+1))

df4['growth_factor'] = growth_factor

df4['growth_ratio'] = a





der = np.array(india_confirmed, dtype=np.float)

p = np.gradient(der) 

df4['growth_rate'] = np.gradient(der)

# here we fit the logistic curve on confirmed cases so that we predict how much it contain peak of confirmed cases in india and when start decrease confirmed cases. 

# x = it shows shape of the curve, how in the infection progress, if it is smaller that means softer the logistic shape is.

# y = the point where curve start to flatten.

# z = it show the predicted maximum infected peoples.



# define sigmoid

def log_curve(x, k, x_0, ymax):

    return ymax/(1+np.exp(-k*(x-x_0)))





popt, pcov = curve_fit(log_curve, df4['days_count'], india_confirmed, bounds=([0,0,0], np.inf), maxfev=50000)

x, y, z = popt  

y_fitted = log_curve(df4['days_count'], x, y, z)

print(x, y, z)

# plot how confirmed_cases increses in india and fit on logistic curve

# second plot for grwoth_factor, growth_ratio and growth_rate



fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(df4['days_count'], y_fitted, '--', label='fitted')

ax.plot(df4['days_count'], india_confirmed, 'o', label='confirmed_data')



fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

ax1.plot(dates, growth_factor, color='black', label='growth_factor')

ax2.plot(dates, a, color='green', label='growth_ratio')

ax3.plot(dates, p, color='red', label='growth_rate')

print(plt.show())
# reproduction number means if x number of person infected then how much other persons is infected by him 

# here we find how reproduction number is vary at the 10 days time interval.

m = []

n = []

reproduction_num = []



for i in range(0, 110, 10):

    m.append(india_confirmed[i:i+10])

    for j in range(len(m[0])):      # bcuz m is list of list type                  

        n.append(int(m[0][j])/int(m[0][0]))           

    reproduction_num.append(n[-1])

    m = []

    n = []

    

print(reproduction_num)