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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
%matplotlib inline 
world_covid = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
world_covid = world_covid.rename(columns = {
    'ObservationDate': 'Date',
    'Province/State': 'State',
    'Country/Region': 'Country'
})
world_covid.head()
covid = pd.read_csv('/kaggle/input/covid19-corona-virus-india-dataset/complete.csv')
covid['Date'] = covid['Date'].astype('str')
covid.head()
labs_covid = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingLabs.csv')
labs_covid.head()
age_covid = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
age_covid.head()
x = world_covid.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].max()
x['Active'] = x['Confirmed'] - (x['Deaths']+x['Recovered'])
active = x['Active'].sum()
death = x['Deaths'].sum()
rec = x['Recovered'].sum()
z = {
    'Active': active,
    'Deaths': death,
    'Recovered': rec
}
ser = pd.Series(z)
ser.name = ''
ser.plot(kind = 'pie', autopct = '%1.2f%%', explode = (0.1,0.1,0.1), figsize = (15,6), startangle = 90)
plt.title('World Covid - 19 cases', fontsize = 20)
x = world_covid.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum()
x.plot(xticks = [], linestyle = '-', marker = '.', figsize = (10,4),color = ('b','r','g'))
plt.title('World Covid 19\n Confirmed v/s Deaths v/s Recovered', fontsize = 20)
x = world_covid.groupby(['Country', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum()
x['Active'] = x['Confirmed'] - (x['Deaths'] + x['Recovered'])
x = x.sort_values('Confirmed',ascending = False).T
countrys = ['US', 'Mainland China', 'Germany', 'South Korea', 'Iran', 'Japan', 'Italy', 'Spain','India']
fig = plt.figure(figsize = (15, 15))
plt.suptitle('Active, Recovered, Deaths in Hotspot Countries and India as of April 26',fontsize = 20,y=1.0)
for i in range(9):
    ax = fig.add_subplot(5,2,i+1)
    country = countrys[i]
    z = x[country].T.sort_values('Confirmed')
    ax.bar(z.index,z['Active'], color = 'g', alpha = 0.3, label = 'Active')
    ax.bar(z.index,z['Deaths'], color = 'r', label = 'Deaths')
    ax.bar(z.index,z['Recovered'], color = 'gray', alpha = 0.6, label = 'Recovered')
    ax.set_xticks([])
    plt.title(country)
    handle, label = ax.get_legend_handles_labels()
    fig.legend(handle, label, loc = 'upper left')
x = world_covid.groupby(['Country', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum()
x = x.T
countrys = ['Mainland China', 'US', 'Germany', 'Italy','India']
plt.figure(figsize = (15,5))
for i in range(5):
    country = countrys[i]
    z = x[country].T
    plt.plot(z.index, z['Confirmed'], marker = '.', label = country)
    plt.xticks([])
plt.legend(loc = 'upper left')
plt.xlabel('Date')
plt.title('Total Confirmed cases in Hotspot Countries and India', fontsize = 20)
x = world_covid.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].max()
x['Active'] = x['Confirmed']-(x['Deaths'] + x['Recovered'])
x['Death Rate'] = np.around(x['Deaths']/ x['Confirmed'],2)
x['Recovery Rate'] = np.around(x['Recovered'] / x['Confirmed'],2)
x.sort_values('Confirmed',ascending = False).fillna(0)[:50].style.background_gradient(cmap='Blues')

x = covid.groupby('Date')['Total Confirmed cases'].sum()
x.plot(xticks = [], marker = '.', figsize = (10,4), color = 'r')
plt.title('Total Confirmed cases', fontsize = 20)
x = covid.groupby('Date')['Total Confirmed cases'].sum()
x = x[:-2]
s = 0
for i in range(len(x.values)):
    t = x.values[i].copy()
    x.values[i] -= s
    s = t
    
x.plot(xticks=[], marker = '.', figsize = (10,4))
plt.title('Daily Confirmed Cases', fontsize = 20)
x = age_covid.groupby('AgeGroup')['TotalCases'].sum()
explode = []
for i in range(len(x.index)):
    explode.append(0.1)
explode = tuple(explode)
x.plot(kind = 'pie', figsize = (15,10), autopct  = '%1.2f%%', explode = explode)
plt.title('Coronavirus cases distribution over various age groups', fontsize = 20)
x = covid.pivot_table('Total Confirmed cases', index = 'Date', columns = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
plt.figure(figsize = (15,5))
plt.plot(x.index, x['Delhi'], color = 'r', marker = '.', label = 'Delhi')
plt.plot(x.index, x['Kerala'], color = 'b', marker = '.',label = 'Kerala')
plt.plot(x.index, x['Maharashtra'], color = 'g', marker = '.', label = 'Maharashtra')
plt.plot(x.index, x['Gujarat'], color = 'k', marker = '.', label = 'Gujarat')
plt.xticks([])
plt.xlabel('Date')
plt.title('Statewise trends of coronavirus cases', fontsize = 20)
plt.legend()
x = covid.pivot_table('Total Confirmed cases', columns = 'Date', index = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
x[x.columns[-1]].sort_values(ascending = False)[:20].plot(kind = 'barh', figsize = (15,5), color = 'm')
plt.title('States with maximum number of cases', fontsize = 20)
x = covid.pivot_table('Total Confirmed cases', columns = 'Date', index = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
x = x[x.columns[-1]].sort_values(ascending = False)
z = x[10:].sum()
x = x[:10]
x.name = ''
x['Others'] = z
explode = []
for i in range(len(x.index)):
    explode.append(0.1)
explode = tuple(explode)
x.plot(kind = 'pie', figsize = (15,10), autopct='%1.1f%%', startangle=90, explode = explode)
plt.title('Coronavirus cases distribution over various states', fontsize = 20)
x = covid.pivot_table('Total Confirmed cases', index = 'Date', columns = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
prev = np.zeros(len(x.values[0]), dtype = 'int64')
for i in range(len(x.values)):
    t = x.values[i].copy()
    x.values[i] -= prev
    prev = t
plt.figure(figsize=(15,5))
plt.plot(x.index, x['Delhi'], marker = '.',label = 'Delhi')
plt.plot(x.index, x['Gujarat'], marker = '.', label = 'Gujarat')
plt.plot(x.index, x['Maharashtra'], marker = '.', label = 'Maharashtra')
plt.ylabel('Number of Daily Coronavirus cases')
plt.xlabel('Date')
plt.title('Delhi v/s Gujarat v/s Maharashtra - Daily confirmed cases', fontsize = 20)
plt.xticks([])
plt.legend()
x = covid.pivot_table('Total Confirmed cases', index = 'Date', columns = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
prev = np.zeros(len(x.values[0]), dtype = 'int64')
for i in range(len(x.values)):
    t = x.values[i].copy()
    x.values[i] -= prev
    prev = t

x  = x[-4:-1].T
print(x.columns)
x['Total'] = 0
for columns in x.columns:
    x['Total'] += x[columns]
x['Total'] //= 2
x.sort_values('Total', ascending = False)[:10].plot(kind = 'barh', figsize = (10,4))
plt.title('Statewise confirmed cases in last 3 days', fontsize = 20)
x = covid.groupby('Date')['Death'].sum()
x.plot(xticks = [], marker = '.', color = 'r', figsize = (10,4))
plt.title('Deaths due to coronavirus', fontsize = 20)
x = covid.groupby('Date')['Death'].sum()
x = x[:-2]
s = 0
for i in range(len(x.values)):
    t = x.values[i].copy()
    x.values[i] -= s
    s = t
    
x.plot(marker = '.',xticks=[], figsize = (10,4))
plt.title('Daily Deaths', fontsize = 20)
x = covid.pivot_table('Death', index = 'Date', columns = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
plt.figure(figsize=(15,5))
plt.plot(x.index, x['Delhi'], color = 'r', marker = '.',label = 'Delhi')
plt.plot(x.index, x['Kerala'], color = 'b', label = 'Kerala', marker = '.')
plt.plot(x.index, x['Maharashtra'], color = 'g', label = 'Maharashtra', marker = '.')
plt.plot(x.index, x['Gujarat'], color = 'k', label = 'Gujarat', marker = '.')
plt.xticks([])
plt.title('Statewise trends of death due to coronavirus', fontsize = 20)
plt.legend()
x = covid.pivot_table('Death', columns = 'Date', index = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
x[x.columns[-1]].sort_values(ascending = False)[:20].plot(kind = 'barh', figsize = (15,6), color = 'm')
plt.title('States with maximum number of deaths', fontsize = 20)
x = covid.pivot_table('Death', columns = 'Date', index = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
x = x[x.columns[-1]].sort_values(ascending = False)
z = x[10:].sum()
x = x[:10]
x.name = ''
x['Others'] = z
explode = []
for i in range(len(x.index)):
    explode.append(0.1)
explode = tuple(explode)
x.plot(kind = 'pie', figsize = (15,10), autopct='%1.1f%%', startangle=90, explode = explode)
plt.title('Death distribution over various states', fontsize = 20)
x = covid.pivot_table('Death', index = 'Date', columns = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
prev = np.zeros(35, dtype = 'int64')
for i in range(len(x.values)):
    t = x.values[i].copy()
    x.values[i] -= prev
    prev = t
plt.figure(figsize = (15,5))
plt.plot(x.index, x['Delhi'], label = 'Delhi', marker = '.')
plt.plot(x.index, x['Gujarat'], label = 'Gujarat', marker = '.')
plt.plot(x.index, x['Maharashtra'], label = 'Maharashtra', marker = '.')
plt.ylabel('Number of Daily Deaths')
plt.xlabel('Date')
plt.title('Delhi v/s Gujarat v/s Maharashtra - Daily Deaths', fontsize = 20)
plt.xticks([])
plt.legend()
x = covid.pivot_table('Death', index = 'Date', columns = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
prev = np.zeros(35, dtype = 'int64')
for i in range(len(x.values)):
    t = x.values[i].copy()
    x.values[i] -= prev
    prev = t

x  = x[-4:-1].T
print(x.columns)
x['Total'] = 0
for columns in x.columns:
    x['Total'] += x[columns]
x['Total'] //= 2
x.sort_values('Total', ascending = False)[:10].plot(kind = 'barh', figsize = (10,4))
plt.title('Statewise deaths in last 3 days', fontsize = 20)
x = covid.groupby('Date')['Cured/Discharged/Migrated'].sum()
x.plot(xticks = [], marker='.', color = 'g', figsize = (10,4))
plt.title('Cured/Discharged/Migrated', fontsize = 20)
x = covid.groupby('Date')['Cured/Discharged/Migrated'].sum()
s = 0
for i in range(len(x.values)):
    t = x.values[i].copy()
    x.values[i] -= s
    s = t
plt.figure(figsize=(10,4))   
x.plot(xticks=[], marker = '.', color = 'g')
plt.title('Daily Recoveries', fontsize = 20)
x = covid.pivot_table('Cured/Discharged/Migrated', index = 'Date', columns = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
plt.figure(figsize = (15,5))
plt.plot(x.index, x['Delhi'], color = 'r', label = 'Delhi', marker= '.')
plt.plot(x.index, x['Kerala'], color = 'b', label = 'Kerala', marker = '.')
plt.plot(x.index, x['Maharashtra'], color = 'g', label = 'Maharashtra', marker = '.')
plt.plot(x.index, x['Gujarat'], color = 'k', label = 'Gujarat', marker = '.')
plt.xticks([])
plt.title('Statewise trends of recovery', fontsize = 20)
plt.legend()
x = covid.pivot_table('Cured/Discharged/Migrated', columns = 'Date', index = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
x[x.columns[-1]].sort_values(ascending = False)[:15].plot(kind = 'barh', figsize = (15,5), color = 'm')
plt.title('States with maximum recoveries', fontsize = 20)
x = covid.pivot_table('Cured/Discharged/Migrated', columns = 'Date', index = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
x = x[x.columns[-1]].sort_values(ascending = False)
z = x[10:].sum()
x = x[:10]
x['Others'] = z
x.name = ''
explode = []
for i in range(len(x.index)):
    explode.append(0.1)
explode = tuple(explode)
x.plot(kind = 'pie', figsize = (15,10), autopct = '%1.1f%%', explode = explode, startangle = 90)
plt.title('Recoveries distribution over various states', fontsize = 20)
x = covid.pivot_table('Cured/Discharged/Migrated', index = 'Date', columns = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
prev = np.zeros(35, dtype = 'int64')
x = x[:-1]
for i in range(len(x.values)):
    t = x.values[i].copy()
    x.values[i] -= prev
    prev = t
plt.figure(figsize = (15,5))
plt.plot(x.index, x['Delhi'], label = 'Delhi', marker = '.')
plt.plot(x.index, x['Gujarat'], label = 'Gujarat', marker = '.')
plt.plot(x.index, x['Maharashtra'], label = 'Maharashtra', marker = '.')
plt.ylabel('Number of Daily Deaths')
plt.xlabel('Date')
plt.xticks([])
plt.title('Delhi v/s Gujarat v/s Maharashtra - Daily Recoveries', fontsize = 20)
plt.legend()
x = covid.pivot_table('Cured/Discharged/Migrated', index = 'Date', columns = 'Name of State / UT', fill_value = 0, aggfunc = 'sum')
prev = np.zeros(35, dtype = 'int64')
x = x[:-2]
for i in range(len(x.values)):
    t = x.values[i].copy()
    x.values[i] -= prev
    prev = t

x  = x[-3:].T
x['Total'] = 0
for columns in x.columns:
    x['Total'] += x[columns]
x['Total'] //= 2
x.sort_values('Total', ascending = False)[:10].plot(kind = 'barh', figsize = (10,5))
plt.title('Statewise recoveries in last 3 days', fontsize = 20)
x1 = covid.groupby(['Name of State / UT', 'Date'])['Total Confirmed cases'].sum()
x2 = covid.groupby(['Name of State / UT', 'Date'])['Death'].sum()
x3 = covid.groupby(['Name of State / UT', 'Date'])['Cured/Discharged/Migrated'].sum()
x1 -= (x2+x3)
fig = plt.figure(figsize = (15,6))
plt.suptitle('Active, Recovered, Deaths in various states in India as of April 26', fontsize = 20, y = 1.0)
ax = fig.add_subplot(2,2,1)
state = 'Maharashtra'
ax.bar(x1[state].index,x1[state], color = 'g', alpha = 0.3, label = 'Cases' )
ax.bar(x2[state].index,x2[state], color = 'r', alpha = 1.0 ,label = 'Death')
ax.bar(x3[state].index,x3[state], color = 'gray', alpha = 0.6, label = 'Recoveries')
ax.set_xticks([])
plt.title('Maharashtra')

ax = fig.add_subplot(2,2,2)
state = 'Delhi'
ax.bar(x1[state].index,x1[state], color = 'g', alpha = 0.3, label = 'Cases' )
ax.bar(x2[state].index,x2[state], color = 'r', alpha = 1.0 ,label = 'Death')
ax.bar(x3[state].index,x3[state], color = 'gray', alpha = 0.6, label = 'Recoveries')
ax.set_xticks([])
plt.title('Delhi')

ax = fig.add_subplot(2,2,3)
state = 'Tamil Nadu'
ax.bar(x1[state].index,x1[state], color = 'g', alpha = 0.3, label = 'Cases' )
ax.bar(x2[state].index,x2[state], color = 'r', alpha = 1.0 ,label = 'Death')
ax.bar(x3[state].index,x3[state], color = 'gray', alpha = 0.6, label = 'Recoveries')
ax.set_xticks([])
plt.title('Tamil Nadu')

ax = fig.add_subplot(2,2,4)
state = 'Gujarat'
ax.bar(x1[state].index,x1[state], color = 'g', alpha = 0.3, label = 'Cases' )
ax.bar(x2[state].index,x2[state], color = 'r', alpha = 1.0 ,label = 'Death')
ax.bar(x3[state].index,x3[state], color = 'gray', alpha = 0.6, label = 'Recoveries')
ax.set_xticks([])
plt.title('Gujarat')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left')
state_cases = covid.groupby('Name of State / UT')['Total Confirmed cases', 'Death', 'Cured/Discharged/Migrated'].max().reset_index()
state_cases = state_cases.rename(columns = {
    'Name of State / UT': 'State',
    'Total Confirmed cases': 'Confirmed',
    'Cured/Discharged/Migrated': 'Recovered'
})
state_cases['Active'] = state_cases['Confirmed'] - (state_cases['Death'] + state_cases['Recovered'])
state_cases['Death Rate'] = np.around(100 * state_cases['Death'] / state_cases['Confirmed'], 2)
state_cases['Recovery Rate'] = np.around(100 * state_cases['Recovered'] / state_cases['Confirmed'], 2)
state_cases.sort_values('Confirmed', ascending = False).fillna(0).style.background_gradient(cmap = 'Greens', subset = [
    'Recovered', 'Recovery Rate'
]).background_gradient(cmap = 'Reds', subset = ['Confirmed','Death','Active','Death Rate'])
x = labs_covid.groupby('state')['lab'].count()
x = x.sort_values(ascending = False)
x.plot(kind = 'barh', figsize = (15,8), color = 'c')
plt.title('Testing Labs - Statewise', fontsize = 20)
x = labs_covid.groupby('city')['lab'].count()
x = x.sort_values(ascending = False)
z = x[30:].sum()
x = x[:30]
x.plot(kind = 'barh',figsize = (15,8),color = 'c')
plt.title('Testing Labs - Citywise', fontsize = 20)