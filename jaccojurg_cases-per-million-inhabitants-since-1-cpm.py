import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import time
import io
import requests


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

#globaldata = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
globaldata = pd.read_csv('/kaggle/input/covid-19-local/complete_data.csv')

globaldata = globaldata[['Province_State', 'Country_Region', 'Latitude', 'Longitude', 'date', 'Confirmed','Deaths']]

population = train = pd.read_csv('/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv')
events = pd.read_excel('/kaggle/input/goverment-measures-by-country-and-date/events.xlsx')

## Allign headings to work with my previous work

globaldata.columns = ['Province.State', 'Country.Region', 'Lat', 'Long', 'date', 'total_cases','total_deaths']
corona = globaldata
keys = ['Province.State','Country.Region']

# Initial Analysis on total cases. This is biased by the testing regime. Needs look at the data set with number of tests done by day by country.

#metric = 'total_cases'
#metric = 'total_deaths'

corona['Province.State'] = corona['Province.State'].fillna(corona['Country.Region'])
population['Province.State'] = population['Province.State'].fillna(population['Country.Region'])
#population['Province.State'][population['Country.Region'] == population['Province.State']] = ""
corona = pd.merge(corona, population, on=keys,how='left')
corona.date = pd.to_datetime(corona.date)

corona['CasesPerM'] =  (1000000.0) * (corona['total_cases'] / corona['Population'])
corona['DeathsPerM'] =  (1000000.0) * (corona['total_deaths'] / corona['Population'])


corona['concatstate'] = corona['Province.State'].apply(lambda x: "" if x=="" else " - "+ x)
corona['location'] = corona['Country.Region']  + corona['concatstate']


## CPM is Cases Per Million - Variable to adjust the start allignment of the time serries
CPM = 1

corona['startdate'] = ""
countrystart = {}

maxdate = corona.date.max().date()

for country in corona.location.unique():
    try: 
        idmin = corona['CasesPerM'][ (corona.CasesPerM > CPM) & (corona.location == country) ].idxmin()
        startdate = corona['date'].iloc[idmin].date()
    except:
        startdate = maxdate # datetime.now().date()
        
    countrystart[country] =  startdate  
    corona['startdate'][corona.location == country] = startdate


corona['dayssince']=corona[['date','startdate']].apply(lambda x: (x[0]-x[1]).days,axis=1)
corona = corona.sort_values('date')

corona['location'][corona['Province.State'] == corona['Country.Region']] = corona['Country.Region']


#values = ['event']
#fields = ['location', 'date']
#events = events.groupby(fields)['event'].apply(lambda x: ' - '.join(x))

#events = events.reset_index()
#events['dayssince'] = events.apply(lambda x: (x[1].date() - countrystart[x[0]]).days , axis=1)
#info = pd.merge(events,corona,on=['location','dayssince'])




def chartcountries(Countries, maxdays=60, metric='CasesPerM'):

    plt.figure(figsize=(15,10))

    for country in Countries:

        subset = corona[ (corona.location == country) & (corona.dayssince >= 0) ]

        labelname  = str(subset.startdate.unique()[0].strftime("%d-%B-%Y")) + " " + country
        plt.plot(subset['dayssince'],subset[metric],label=labelname,linewidth=3)
        
    #for r in range(len(info)):
    #    plt.annotate(info.loc[r]['event'],(info.loc[r]['dayssince'],info.loc[r]['CasesPerM']))

    title = str(Countries)
    plt.grid()
    plt.title("Total " + metric + " per Million" + " as of " + str(maxdate)  )
    plt.xlabel('Days since cases per million inhabitants passed ' + str(CPM) )
    plt.ylabel(metric + ' per million inhabitants')
    plt.xlim(0,maxdays)
    plt.legend()
## Look at wide range of different countries

Countries = ['Italy', 'Netherlands','Sweden', 'China - Hubei', 'US - New York','Germany', 'Iran', 'China - Hubei','Belgium', 'Spain', 'US - California'] #, 'Finland','Spain','France','US - California']

chartcountries(Countries)
chartcountries(Countries, metric='DeathsPerM', maxdays=80)

## Look at wide range of different countries

Countries = ['Italy', 'Netherlands','China - Hubei','Germany', 'US - New York'] #, 'Finland','Spain','France','US - California']


chartcountries(Countries, metric='DeathsPerM', maxdays=80)


Countries = ['Italy','Netherlands', 'China - Hubei', 'US - New York','Finland','Denmark','Sweden', 'Norway']

chartcountries(Countries)
chartcountries(Countries, metric='DeathsPerM')
## Look at wide range of different countries

Countries = ['Japan', 'Singapore'] #, 'Finland','Spain','France','US - California']

chartcountries(Countries)
chartcountries(Countries, metric='DeathsPerM', maxdays=60)


subset = corona[['location','CasesPerM']][(corona['Country.Region'] == 'China') & 
                                    (corona.dayssince >= 0)]



fields = ['location']
subet = subset.groupby(fields)['CasesPerM'].max()
subset  = subet.reset_index()
subset = subset.sort_values('CasesPerM',ascending=False)
Countries = list(subset['location'][:5]) + list(subset['location'][-2:] )


chartcountries(Countries,80)
chartcountries(Countries, metric='DeathsPerM')
#Look at Chinae  excluding Hubei

subset = corona[['location','CasesPerM']][(corona['Country.Region'] == 'China') & 
                                    (corona.dayssince >= 0) &
                                    (corona['Province.State'] != 'Hubei')]
fields = ['location']
subet = subset.groupby(fields)['CasesPerM'].max()
subset  = subet.reset_index()
subset = subset.sort_values('CasesPerM',ascending=False)
Countries = list(subset['location'][:5]) + list(subset['location'][-2:] )


chartcountries(Countries,80)
chartcountries(Countries, metric='DeathsPerM')

corona
subet = corona[['location','CasesPerM','DeathsPerM','dayssince']][( (corona.dayssince >= 0) & (corona.Population > 1000000) &
                                     (corona['Province.State'] != 'Grand Princess') & 
                                    (corona['Province.State'] != 'Diamond Princess'))]

fields = ['location']
subet = subet.groupby(fields)['DeathsPerM','dayssince'].max()
subset  = subet.reset_index()
subset = subset.sort_values('DeathsPerM',ascending=False)
Countries = list(subset['location'][:10]) #+ list(subset['location'][-3:] )

chartcountries(Countries, metric='DeathsPerM')

print(subset[:10])
subet = corona[['location','CasesPerM','DeathsPerM','dayssince']][( (corona.dayssince >= 0) & (corona.Population > 10000) &
                                     (corona['Province.State'] != 'Grand Princess') & 
                                    (corona['Province.State'] != 'Diamond Princess'))]

fields = ['location']
subet = subet.groupby(fields)['DeathsPerM','dayssince'].max()
subset  = subet.reset_index()
subset = subset.sort_values('DeathsPerM',ascending=False)
Countries = list(subset['location'][:10]) #+ list(subset['location'][-3:] )


print(subset[:50])
subet = corona[['location','CasesPerM','DeathsPerM','dayssince']][( (corona.dayssince >= 0) & (corona.Population > 1000000) &
                                     (corona['Province.State'] != 'Grand Princess') & 
                                    (corona['Province.State'] != 'Diamond Princess'))]

fields = ['location']
subet = subet.groupby(fields)['DeathsPerM','dayssince'].max()
subset  = subet.reset_index()
subset = subset.sort_values('DeathsPerM',ascending=False)
Countries = list(subset['location'][10:20]) #+ list(subset['location'][-3:] )


chartcountries(Countries, metric='DeathsPerM')

print(subset[10:20])
import seaborn as sns; sns.set()

df = subset[0:10].reset_index()

plt.figure(figsize=(15,10))
ax = sns.scatterplot(x='dayssince',y='DeathsPerM', hue='location', data=df, marker='o')


for line in range(0,df.shape[0]):
     plt.text(df.dayssince[line]+0.2, df.DeathsPerM[line], df.location[line], horizontalalignment='left', size='medium', color='black', weight='semibold')


import seaborn as sns; sns.set()

df = subset[11:20].reset_index()

plt.figure(figsize=(15,10))
ax = sns.scatterplot(x='dayssince',y='DeathsPerM', hue='location', data=df, marker='o')


for line in range(0,df.shape[0]):
     plt.text(df.dayssince[line]+0.2, df.DeathsPerM[line], df.location[line], horizontalalignment='left', size='medium', color='black', weight='semibold')


subet = corona[['location','CasesPerM']][(corona['Country.Region'] == 'US') & 
                                    (corona.dayssince >= 0) &
                                     (corona['Province.State'] != 'Grand Princess') & 
                                    (corona['Province.State'] != 'Diamond Princess')]

fields = ['location']
subet = subet.groupby(fields)['CasesPerM'].max()
subset  = subet.reset_index()
subset = subset.sort_values('CasesPerM',ascending=False)
Countries = list(subset['location'][:6]) #+ list(subset['location'][-3:] )

chartcountries(Countries)
chartcountries(Countries, metric='DeathsPerM')


Countries = ['Italy', 'Netherlands', 'Finland', 'Sweden','Norway','Germany', 'Spain', 'Portugal','Denmark','Belgium', 'France', 'Spain', 'San Marino', 'Switzerland'] #, 'US - California']
chartcountries(Countries)
chartcountries(Countries, metric='DeathsPerM')

Countries = ['Spain', 'San Marino', 'Switzerland', 'Italy']

chartcountries(Countries,40)
chartcountries(Countries, metric='DeathsPerM')

Countries = ['United Kingdom','Netherlands','US - New York', 'Italy','Spain', 'Ireland'] #, 'US - California']
chartcountries(Countries,60)
chartcountries(Countries, metric='DeathsPerM')

Countries = ['Brazil', 'Argentina',"Chile",'China - Hubei', 'Suriname'] #, 'US - California']
chartcountries(Countries,40)
chartcountries(Countries, metric='DeathsPerM')

Countries = ['Tunisia',"Egypt",'China - Hubei'] #, 'US - California']
chartcountries(Countries,40)
chartcountries(Countries, metric='DeathsPerM')
plt.figure(figsize=(15,10))

Countries = ['Italy', 'Netherlands']
#Countries = ['Italy', 'Netherlands', 'France'] #, 'Belgium'] #, 'Belgium','France']
#,'Greece',"Spain","Belgium", "United Kingdom", "Finland", 'Denmark', 'US']
#Countries = ['Netherlands','Italy', "Singapore", 'Spain', "Belgium", "Germany"]



for country in Countries:
    
    subset = corona[ (corona.location == country) & (corona.dayssince >= 0) ]
    print(subset.startdate.unique())
    labelname  = str(subset.startdate.unique()[0].strftime("%d-%B-%Y")) + " " + country
    plt.plot(subset['dayssince'],subset['CasesPerM'],label=labelname,linewidth=3)
  

        
#plt.plot(info['dayssince'],info['CasesPerM'],marker='o', linestyle='', ms=5, label=info['event'])


for r in range(len(info)):
    plt.annotate(info.loc[r]['event'],(info.loc[r]['dayssince'],info.loc[r]['CasesPerM']),
                 xytext = (-10+info.loc[r]['dayssince'],info.loc[r]['CasesPerM']-1),
                 #color=c[info.loc[r]['location']],
                 arrowprops=dict(arrowstyle="->",connectionstyle="angle3"),
                 horizontalalignment='left', verticalalignment='center')



title = str(Countries)
plt.grid()
plt.title("Total " + metric + " per Million"  )
plt.xlabel('Days since cases per million inhabitants passed ' + str(CPM) )
plt.ylabel('Cases per million inhabitants')
plt.xlim(0,40)
plt.legend()