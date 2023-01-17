import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings 

warnings.filterwarnings("ignore")





# for kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)



df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")



df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)

df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)

df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)
df.columns
df.head()
df.tail()
df.describe()
df2 = df.groupby(["Date", "Country", "Province/State"])[['SNo', 'Date', 'Province/State', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df2
df.query('Country=="Mainland China"').groupby("Last Update")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df.groupby("Country")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df.groupby('Date').sum()
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
f,ax = plt.subplots(figsize = (20,15))

sns.barplot(x=confirmed['Confirmed'],y=confirmed['Date'],color='yellow',alpha = 0.5,label='confirmed' )

sns.barplot(x=deaths['Deaths'],y=deaths['Date'],color='purple',alpha = 0.7,label='deaths')

sns.barplot(x=recovered['Recovered'],y=recovered['Date'],color='green',alpha = 0.6,label='recovered')





ax.legend(loc='lower right',frameon = True)     

ax.set(xlabel='confirmed, deaths and recovered', ylabel='Date',title = "confirmed, deaths and recovered rate by dates")

plt.show()

dfForVisualization = pd.DataFrame({'Date': confirmed.Date,'Confirmed': confirmed.Confirmed, 'Deaths': deaths.Deaths, 'Recovered': recovered.Recovered})
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='Date',y='Confirmed',data=dfForVisualization,color='orange',alpha=0.8)

sns.pointplot(x='Date',y='Deaths',data=dfForVisualization,color='red',alpha=0.8)

sns.pointplot(x='Date',y='Recovered',data=dfForVisualization,color='lime',alpha=0.8)



plt.text(20,130000,'number of Confirmed',color='orange',fontsize = 17,style = 'italic')

plt.text(20,120000,'number of Deaths',color='red',fontsize = 18,style = 'italic')

plt.text(20,110000,'number of Recovered',color='lime',fontsize = 18,style = 'italic')







plt.xlabel('Dates',fontsize = 15,color='blue')

plt.xticks(rotation=45,size=10)

plt.ylabel('Numbers',fontsize = 15,color='blue')

plt.title('number of Confirmed - Deaths - Recovered',fontsize = 20,color='blue')

plt.grid()