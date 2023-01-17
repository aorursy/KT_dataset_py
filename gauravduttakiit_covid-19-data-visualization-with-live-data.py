from datetime import datetime



now = datetime.now()



date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

print("Last Updated at :",date_time)
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

%matplotlib inline

plt.style.use('ggplot')

covid19=pd.read_csv('/kaggle/input/covid-19/data/countries-aggregated.csv',parse_dates=['Date'])
print('The Data is available till :' , covid19['Date'].iloc[-1])
covid19.head()
covid19['Total Cases']= covid19[['Confirmed', 'Recovered', 'Deaths']].sum(axis=1)
covid19.head()
worldwide_covid19=covid19.groupby(['Date']).sum()

ax=worldwide_covid19.plot(figsize=(8,8))

ax.set_xlabel('Date')

ax.set_ylabel('# of Cases Worldwide')

ax.title.set_text('Worldwide Covid 19 Insights')

plt.show()
India_covid19=covid19[covid19['Country']== 'India'].groupby(['Date']).sum()

US_covid19=covid19[covid19['Country']== 'US'].groupby(['Date']).sum()

Brazil_covid19=covid19[covid19['Country']== 'Brazil'].groupby(['Date']).sum()

Russia_covid19=covid19[covid19['Country']== 'Russia'].groupby(['Date']).sum()

SA_covid19=covid19[covid19['Country']== 'South Africa'].groupby(['Date']).sum()

fig=plt.figure(figsize=(12,5))

ax=fig.add_subplot(111)

ax.plot(US_covid19[['Total Cases']],label='US')

ax.plot(India_covid19[['Total Cases']],label='India')

ax.plot(Brazil_covid19[['Total Cases']],label='Brazil')

ax.plot(Russia_covid19[['Total Cases']],label='Russia')

ax.plot(SA_covid19[['Total Cases']],label='South Africa')

ax.set_xlabel('Date')

ax.set_ylabel('# of Total Cases')

ax.title.set_text('Top 5 Covid 19 Cases by Country')

plt.legend(loc='upper left')



plt.show()
fig=plt.figure(figsize=(12,5))

ax=fig.add_subplot(111)

ax.plot(worldwide_covid19[['Total Cases']],label='World')

ax.plot(India_covid19[['Total Cases']],label='India')

ax.set_xlabel('Date')

ax.set_ylabel('# of Total Cases')

ax.title.set_text('World v/s India')

plt.legend(loc='upper left')

plt.show()

India_covid19=India_covid19.reset_index()

India_covid19['Daily Cases']=India_covid19['Total Cases'].sub(India_covid19['Total Cases'].shift())

India_covid19['Daily Deaths']=India_covid19['Deaths'].sub(India_covid19['Deaths'].shift())

India_covid19['Daily Confirmed']=India_covid19['Confirmed'].sub(India_covid19['Confirmed'].shift())

India_covid19['Daily Recovered']=India_covid19['Recovered'].sub(India_covid19['Recovered'].shift())

fig=plt.figure(figsize=(20,8))

ax=fig.add_subplot(111)

ax.bar(India_covid19['Date'],India_covid19['Daily Cases'], color='r',label='India Daily Cases')

ax.bar(India_covid19['Date'],India_covid19['Daily Confirmed'], color='b',label='India Daily  Confirmed Cases')

ax.bar(India_covid19['Date'],India_covid19['Daily Recovered'], color='m',label='India Daily Recovered Cases')

ax.bar(India_covid19['Date'],India_covid19['Daily Deaths'], color='g',label='India Daily Death Cases')

ax.set_xlabel('Date')

ax.set_ylabel('# of Affected people')

ax.title.set_text('India Daily Cases,Death Cases,Daily Confirmed,Daily Recovered ')

plt.legend(loc='upper left')

plt.show()
fig=plt.figure(figsize=(20,8))

ax=fig.add_subplot(111)

plt.yscale("log")

ax.bar(India_covid19['Date'],India_covid19['Daily Recovered'], color='r',label='India Daily Recovered Cases')

ax.bar(India_covid19['Date'],India_covid19['Daily Deaths'], color='g',label='India Daily Death Cases')

ax.set_xlabel('Date')

ax.set_ylabel('# of Affected people')

ax.title.set_text('India Daily Recovered Cases & Death Cases')

plt.legend(loc='upper left')

plt.show()
from datetime import date, timedelta

yesterday=date.today() - timedelta(days=3)

yesterday=yesterday.strftime('%Y-%m-%d')

today_covid=covid19[covid19['Date']==yesterday ]

top_10=today_covid.sort_values(['Confirmed'],ascending=False)[:10]

top_10

top_10.loc['rest-of-world']=today_covid.sort_values(['Confirmed'],ascending=False)[10:].sum()

top_10.loc['rest-of-world','Country']= "Rest of world"

fig=plt.figure(figsize=(10,10))

ax=fig.add_subplot(111)

ax.pie(top_10['Confirmed'],labels=top_10['Country'],autopct='%1.1f%%')

ax.title.set_text('Hardest Hit Coutried by Covid 19')

print("Displaying Data for",yesterday)

print ('on', date.today() )

plt.legend(loc='upper left')

plt.show()