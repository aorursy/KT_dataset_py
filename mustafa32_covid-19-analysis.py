import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import folium
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv",parse_dates=['Date'])
df.head()
df['Country'] = df['Country'].replace('Mainland China', 'China')

df[['Province/State']] = df[['Province/State']].fillna('NA')
df.head()
df.groupby('Country')['Recovered'].mean().sort_values(ascending=False)
df.groupby('Country')['Confirmed'].mean().sort_values( ascending=False)
plt.figure(figsize=(15,7))

plt.xlabel("Deaths")

df.groupby('Date')['Deaths'].plot()
df.groupby(['Country','Province/State'])['Confirmed','Deaths','Recovered'].max()
df.groupby(['Country'])['Confirmed','Deaths','Recovered'].sum().reset_index().sort_values(by='Deaths',ascending=False)
df[df['Recovered']==0][['Country','Confirmed', 'Deaths', 'Recovered']].sort_values(by='Confirmed',ascending=False)
df[df['Confirmed']==df['Deaths']+df['Recovered']].sort_values(by='Confirmed',ascending=False)