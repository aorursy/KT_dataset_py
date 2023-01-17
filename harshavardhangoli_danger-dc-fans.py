#Team DangerDCFans
#Team members: Harsha vardhan reddy Goli
               #Abhinav mamidipelly
               #Pradeep Ujwal Kammadhanam
               #Prakyath Reddy Kandimalla
# We've used data from kaggle to get this plots 
#the website we used for reference 
#geeksforgeeks.com
#Medium.com
#github
#We changed the labels and also the colors
#We also made 2 different types plots for better understanding
#We made it easier to understand
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import random
import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
df['month'] = pd.DatetimeIndex(df['Date']).month
df.head()
sorted_by_country = df[df.Date == '2020-04-07'].groupby('Country_Region').agg('sum').reset_index().sort_values(by = ['ConfirmedCases', 'Fatalities'], ascending = False)
sorted_by_country['Death Ratio'] = sorted_by_country.Fatalities / sorted_by_country.ConfirmedCases * 100
sorted_by_country.drop(columns = ['month'], inplace = True)
sorted_by_country.head(20)
# New in APRIL
march = df[df.Date == '2020-03-31'].groupby('Country_Region').agg('sum').reset_index()
april = df[df.Date == '2020-04-07'].groupby('Country_Region').agg('sum').reset_index()
Confirmed = april.ConfirmedCases - march.ConfirmedCases
Fatalities = april.Fatalities - march.Fatalities
april['ConfirmedCases_new'] = Confirmed
april['Fatalities_new'] = Fatalities

april_data = april.sort_values(by = ['ConfirmedCases_new', 'Fatalities_new'], ascending = False)
april_data['Death Ratio'] = april_data.Fatalities_new / april_data.ConfirmedCases_new * 100
april_data.drop(columns = ['month'], inplace = True)
april_data.head(20)
#Fatalities by month
fig = plt.figure(figsize = (15, 8))

for idx, country in enumerate(random.sample(list(df.Country_Region.unique()), 9)):
    temp = df[df['Country_Region'] == country].groupby('Date').agg('sum').reset_index()
    temp['month'] = pd.DatetimeIndex(temp['Date']).month
    temp = temp.groupby('month').agg('sum').reset_index()
    name = 'ax' + str(idx)
    name = fig.add_subplot(3, 3, idx+1)
    name.bar(temp.month, temp.ConfirmedCases)
    name.set_title("Montly Confirmed Cases in {}".format(country))
    name.set_xticks([1,2,3,4])
    name.grid(True)
            
plt.tight_layout()
plt.show()
#Confirmed cases by month
fig = plt.figure(figsize = (15, 8))

for idx, country in enumerate(random.sample(list(df.Country_Region.unique()), 9)):
    temp = df[df['Country_Region'] == country].groupby('Date').agg('sum').reset_index()
    temp['month'] = pd.DatetimeIndex(temp['Date']).month
    temp = temp.groupby('month').agg('sum').reset_index()
    name = 'ax' + str(idx)
    name = fig.add_subplot(3, 3, idx+1)
    name.bar(temp.month, temp.Fatalities)
    name.set_title("Montly Death in {}".format(country))
    name.set_xticks([1,2,3,4])
    name.grid(True)
            
plt.tight_layout()
plt.show()

#confirmed vs fatalities
fig = plt.figure(figsize = (15, 8))
every_nth = 10

for idx, country in enumerate(random.sample(list(df.Country_Region.unique()), 10)):
    temp = df[df['Country_Region'] == country].groupby('Date').agg('sum').reset_index()
    name = 'ax' + str(idx)
    name = fig.add_subplot(5, 2, idx+1)
    name.plot(temp.Date, temp.ConfirmedCases)
    name.plot(temp.Date, temp.Fatalities)
    name.set_title("Day by Day Confirmed Cases vs Fatalities in {}".format(country))
    name.legend(('Confirmed Cases', 'Fatalities'))
    name.grid(True)
    for n, label in enumerate(name.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
            
plt.tight_layout()
plt.show()
# Death ratio by months
fig = plt.figure(figsize = (15, 8))
every_nth = 10

for idx, country in enumerate(random.sample(list(df.Country_Region.unique()), 10)):
    temp = df[df['Country_Region'] == country].groupby('Date').agg('sum').reset_index()
    temp['Death_Ratio'] = temp.Fatalities / temp.ConfirmedCases * 100
    name = 'ax' + str(idx)
    name = fig.add_subplot(5, 2, idx+1)
    name.plot(temp.Date, temp.Death_Ratio)
    name.axhline(temp.Death_Ratio.mean(), linestyle = '--', color = 'red')
    name.set_title("Day by Day Death Ratio (%) in {}".format(country))
    name.legend(['Death Ratio', 'Average Death Ratio'], fontsize = 8)
    name.grid(True)
    for n, label in enumerate(name.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
            
plt.tight_layout()
plt.show()
df.to_csv('submission.csv', index=False)
df