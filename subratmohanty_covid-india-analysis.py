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
#data loaded

census=pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')

covid=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

hospitals=pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')



#data cleaned

census.dropna()

census=census.drop_duplicates()

covid=covid.dropna()

covid=covid.drop_duplicates()

hospitals=hospitals.replace(np.nan,0)

hospitals=hospitals.drop_duplicates()

hospitals=hospitals.drop([36,37],axis=0)



census.head(2)

covid.head(2)

hospitals=hospitals.rename(columns={'State/UT':'State / Union Territory'})

census.head(2)

covid.head(2)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()





#print(covid[covid['Date']== '22/03/20'])

covid_ds=covid[270:]

covid_ds

covid_ds['Total cases']=covid_ds['ConfirmedIndianNational']+covid_ds['ConfirmedForeignNational']

covid_ds['% recovered']= round((covid_ds['Cured']/covid_ds['Total cases']*100),0)

covid_ds=covid_ds.sort_values(by='Total cases',ascending=False)
plt.figure(figsize=(15,15))

plt.subplot(3,1,1)

plt.bar(covid_ds['State/UnionTerritory'],covid_ds['ConfirmedIndianNational'], color='c',label='Indians')

plt.xticks(rotation=90)

plt.legend()

plt.title('COVID-19 case distribution: Statewise')

plt.show()

plt.figure(figsize=(15,15))

plt.subplot(3,1,2)

plt.bar(covid_ds['State/UnionTerritory'],covid_ds['ConfirmedForeignNational'], color='k',label='Foreigners')

plt.xticks(rotation=90)

plt.legend()

plt.show()

import seaborn as sns

covid_copy=covid_ds.copy()

covid_copy=covid_copy.replace({'State / Union Territory':'State/UnionTerritory'})

#print(covid_copy.columns[2])

#print(census.columns[1])

c=covid_copy.rename(columns={covid_copy.columns[2]:census.columns[1]})

covid_copy=c

combined_ds=pd.merge(census,covid_copy,on=census.columns[1],how='inner')

combined_ds=combined_ds.replace(np.nan,0)



print("Correlation between population and total cases:",(combined_ds['Population']).corr((combined_ds['Total cases'])))

print("Correlation between urban population and total cases:",(combined_ds['Urban population']).corr((combined_ds['Total cases'])))

print("Correlation between rural population and total cases:",(combined_ds['Rural population']).corr((combined_ds['Total cases'])))

print("Correlation between sex ratio and total cases:",(combined_ds['Sex Ratio']).corr((combined_ds['Total cases'])))



plt.figure(figsize=(15,15))

plt.subplot(3,1,3)

plt.scatter(combined_ds['Urban population'],combined_ds['Total cases'],color='c')

plt.title('Urban population Vs Total cases of infection')

plt.xticks(rotation=90)

plt.show()
combined_ds['State / Union Territory'].astype('str')

hospitals['State / Union Territory'].astype('str')



table=pd.merge(combined_ds,hospitals,on='State / Union Territory',how='inner')

table['Population'].astype('str').astype('int')

table['NumPublicBeds_HMIS']=table['NumPublicBeds_HMIS'].astype('int')

table['Specific bed count']=table['NumPublicBeds_HMIS']/table['Population']

table.sort_values(by='Population',ascending=False)





plt.figure(figsize=(15,15))

plt.subplot(3,1,3)

plt.bar(table['State / Union Territory'],table['Specific bed count'],color='r')

plt.title('Number of hospitals beds available per person')

plt.xticks(rotation=90)

plt.show()



plt.figure(figsize=(15,15))

plt.subplot(3,1,3)

plt.bar(table['State / Union Territory'],table['Total cases'],color='g')

plt.title('Total case count - state wise')

plt.xticks(rotation=90)

plt.show()



plt.figure(figsize=(15,15))

plt.subplot(3,1,3)

plt.bar(table['State / Union Territory'],table['% recovered'], color='b')

plt.xticks(rotation=90)

plt.title('Recovery percentage - state wise')

plt.legend()

plt.show()
import matplotlib.pyplot as plt

c=covid.copy()

c['Total Cases']=c['ConfirmedIndianNational']+c['ConfirmedForeignNational']

tab=pd.DataFrame(c.groupby('Date')['Total Cases'].sum())

tab.reset_index(inplace=True)

tab['Date']=pd.to_datetime(tab['Date'],dayfirst=True)



tab=tab.sort_values(by='Date',ascending=False)



plt.figure(figsize=(20,20))

plt.subplot(1,1,1)

plt.plot(tab['Date'],tab['Total Cases'])

plt.xticks(rotation=90)

plt.title('Growth of COVID-19 cases in India in last 3 months')

plt.show()
t=pd.DataFrame(c.groupby(['State/UnionTerritory','Date'])['Total Cases'].sum())

states_affected=list(set(c['State/UnionTerritory']))

t.reset_index(inplace=True)

t['Date']=pd.to_datetime(t['Date'],dayfirst=True)

t=t.sort_values(by='Date',ascending=True)

#plt.plot(t[t['State/UnionTerritory']=='Kerala']['Date'],t[t['State/UnionTerritory']=='Kerala']['Total Cases'])

fig=plt.figure(figsize=(16,16))

plt.subplot(2,1,1)

for i in states_affected[:12]:

    plt.scatter(t[t['State/UnionTerritory']==i]['Date'],t[t['State/UnionTerritory']==i]['Total Cases'],label=i)

    plt.legend()

plt.xticks(rotation=90)

plt.title('COVID-19 spread in India: Statewise')



plt.subplot(2,1,2)

for i in states_affected[12:]:

    plt.scatter(t[t['State/UnionTerritory']==i]['Date'],t[t['State/UnionTerritory']==i]['Total Cases'],label=i)

    plt.legend()

plt.xticks(rotation=90)

plt.show()

fig.savefig('COVID.jpeg')