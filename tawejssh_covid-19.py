import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import datetime
#Show more dataframe columns

pd.set_option("display.max_columns", 1000)
#Plot style

plt.style.use('fivethirtyeight')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
data.info()
#Let's use "China" instead of "Mainland China" and "Macao" instead of "Macau"

data['Country/Region'] = data['Country/Region'].replace('Mainland China', 'China')

#And replace nan values in province by empty string

data['Province/State'] = data['Province/State'].fillna('')
#Now we'll transform the strings to python datetime objects and extract DOW, MONTH, etc..

data['Date'] = data['ObservationDate'].apply(lambda x : datetime.datetime.strptime(x, '%m/%d/%Y'))

data['ObservationDOW'] = data['Date'].apply(lambda x: x.day_name())

data['ObservationMonth'] = data['Date'].apply(lambda x: x.month_name())

#Add Active Cases

data['Active'] = data['Confirmed'] - data['Deaths'] - data['Recovered']
#We'll drop some columns 

data.drop(columns=['SNo', 'Last Update'], inplace=True)
data.head()
last_available_date = data['ObservationDate'].iloc[-1]

latest_data = data.loc[data.ObservationDate == last_available_date].groupby('Country/Region').sum()[['Confirmed', 'Deaths', 

                                                                                        'Recovered', 'Active']]
evolution = data.groupby('ObservationDate').sum()[['Confirmed', 'Deaths','Recovered', 'Active']]

evolution['Death Rate'] = (evolution['Deaths'] / evolution['Confirmed']) * 100

evolution['Recovery Rate'] = (evolution['Recovered'] / evolution['Confirmed']) * 100
print(f'Last Update: {last_available_date}')

plt.figure(figsize=(11,6)) 

plt.pie(latest_data.sum().iloc[1:], labels=['Deaths', 'Recovered', 'Active'], autopct='%1.1f%%', explode=(1,0,0), shadow=True)

plt.title('COVID-19 Global Stats')

plt.show()
plt.figure(figsize=(10,7))

sns.boxplot(data=evolution[['Deaths', 'Recovered', 'Active']])

plt.title('Cases Dsitributions Boxplots')

plt.show()
plt.figure(figsize=(20,7))

plt.plot(evolution['Active'], label='Active')

plt.plot(evolution['Recovered'], label='Recovered')

plt.xticks(evolution.index[::7],rotation=45)

plt.title('Evolution of COVID-19 Results (worldwide)')

plt.xlabel('Date')

plt.ylabel('Number of Cases')

plt.legend()

plt.show()
#Due to values' ranges, we decided to plot deaths evolution separately (to be clearer)

plt.figure(figsize=(20,7))

plt.plot(evolution['Deaths'], label='Deaths')

plt.xticks(evolution.index[::7],rotation=45)

plt.title('Evolution of COVID-19 Results (worldwide)')

plt.xlabel('Date')

plt.ylabel('Number of Cases')

plt.legend()

plt.show()
#Let's visualise both death and recovery rates now

plt.figure(figsize=(20,7))

plt.plot(evolution['Recovery Rate'], label='Recovery Rate')

plt.title('Evolution of COVID-19 Recovery Rate (worldwide)')

plt.xticks(evolution.index[::7],rotation=45)

plt.ylabel('Rate %')

plt.legend()

plt.show()
#What about the evolution of death rate ?

plt.figure(figsize=(20,7))

plt.plot(evolution['Death Rate'], label='Death Rate')

plt.title('Evolution of COVID-19 Death Rate (worldwide)')

plt.xticks(evolution.index[::7],rotation=45)

plt.ylabel('Rate %')

plt.legend()

plt.show()
diff_evolution = evolution.diff().iloc[1:]
plt.figure(figsize=(20,7))

plt.plot(diff_evolution['Confirmed'])

plt.title('Evolution of COVID-19 New Confirmed Cases (worldwide)')

plt.xticks(diff_evolution.index[::7],rotation=45)

plt.ylabel('New Confirmed Cases')

plt.show()
plt.figure(figsize=(20,7))

plt.plot(diff_evolution['Deaths'])

plt.title('Evolution of COVID-19 New Deaths Cases (worldwide)')

plt.xticks(diff_evolution.index[::7],rotation=45)

plt.ylabel('New Deaths Cases')

plt.show()
#If you see a negative value in the plot above, uncomment the last line.

#As we can see in the deaths curve, there's a potential anomaly: a decrease in total deaths number.

#evolution.loc[['08/16/2020', '08/17/2020', '08/18/2020']]
plt.figure(figsize=(20,7))

plt.plot(diff_evolution['Recovered'])

plt.title('Evolution of COVID-19 New Recovery Cases (worldwide)')

plt.xticks(diff_evolution.index[::7],rotation=45)

plt.ylabel('New Recovery Cases')

plt.show()
latest_data = latest_data.sort_values(by='Confirmed', ascending=False)

latest_data.head(10).style.background_gradient(cmap='Oranges')
k = 10 # we'll show the Top k affected countries

conf_max = latest_data['Confirmed'][:k] 

conf_max.loc['Others'] = latest_data['Confirmed'][k:].sum()

plt.figure(figsize=(20,10))

plt.pie(conf_max, labels=conf_max.index, autopct='%1.1f%%', shadow=True, pctdistance=0.8)

plt.title('COVID-19 Confirmed Cases Partition')

plt.show()
px.bar(latest_data.reset_index()[0:10].sort_values('Confirmed', ascending=True),

       y="Confirmed", 

       x="Country/Region", 

       title="COVID-19  Top 10 Affected Countries",

       hover_data=['Deaths'],

       color='Deaths',

       orientation='v')
d = data.groupby(['Country/Region', 'ObservationDate']).sum().reset_index().sort_values('ObservationDate')

#US not included for better visualization since values are very high comparing to other countries

top_30_countries = latest_data.index[1:31].tolist()

d =d.loc[d['Country/Region'].isin(top_30_countries)]
df = d.copy()

for date in d['ObservationDate'].unique():

    for country in top_30_countries:

        if country not in d.loc[d['ObservationDate'] == date]['Country/Region'].unique():

            dff = pd.DataFrame({'Country/Region':[country], 'ObservationDate':[date], 'Confirmed':[0],

                               'Deaths':[0], 'Recovered':[0], 'Active':[0]})

            df = pd.concat([df,dff])
#Sorting dataframe by Date and the ranking of top 30 countries 

sorterIndex = dict(zip(top_30_countries, range(len(top_30_countries))))

df['Country/Region_X'] = df['Country/Region'].map(sorterIndex)

df.sort_values(['ObservationDate', 'Country/Region_X'], ascending=[True, True], inplace=True)

df.drop(columns='Country/Region_X', inplace = True)
df.reset_index(drop=True,inplace=True)
fig = px.bar(df, x="Confirmed",

             y="Country/Region", 

             color="Deaths", 

             orientation='h',

             title="Evolution of Confirmed Cases: Top 30 US not included",

             animation_frame='ObservationDate')

fig.update_layout(autosize=False, width=700, height=900)

fig.show()
sns.set()

fctgrid = sns.FacetGrid(data=df,

                        col='Country/Region',

                        hue='Country/Region',

                        col_wrap=4,

                        sharey=False)

fctgrid.map(plt.plot, 'ObservationDate', 'Confirmed')

fctgrid.set(xticks=df['ObservationDate'].unique()[10::90])

fctgrid.set_xticklabels(rotation=45)

fctgrid.fig.suptitle('Evolution Curves')

plt.subplots_adjust(top=0.9)

plt.show()
plt.style.use('fivethirtyeight') #using this style with the previous facetgrid may affect the clearness of the plot
plt.figure(figsize=(15,7))

sns.barplot(data=latest_data.reset_index()[:20], x='Confirmed', y='Country/Region', label='Confirmed', color='k')

sns.barplot(data=latest_data.reset_index()[:20], x='Recovered', y='Country/Region', label='Recovered', color='r')

plt.title('Confirmed vs Recovered: Top 20')

plt.legend()

plt.show()
print('Countries With No Deaths Registered')

no_deaths = latest_data.loc[latest_data['Deaths'] == 0]

no_deaths.style.background_gradient(cmap='Blues')
print('Countries With No Active Cases')

no_deaths = latest_data.loc[latest_data['Active'] == 0]

no_deaths.style.background_gradient(cmap='Purples')
print('Countries With No Recovery Registered')

no_recovery = latest_data.loc[latest_data['Recovered'] == 0]

no_recovery.style.background_gradient(cmap='Greens')
map_data = data.groupby(['Country/Region', 'ObservationDate']).sum().reset_index()
fig = px.choropleth(map_data, 

                    locations ="Country/Region", 

                    color ="Confirmed", 

                    hover_name='Country/Region',

                    locationmode="country names",

                    color_continuous_scale = px.colors.sequential.Plasma, 

                    scope ="world",

                    animation_frame ="ObservationDate",

                   title="Evolution of Confirmed Cases") 

fig.show()
fig = px.choropleth(map_data, 

                    locations ="Country/Region", 

                    color ="Deaths", 

                    hover_name='Country/Region',

                    locationmode="country names",

                    color_continuous_scale = px.colors.sequential.Plasma, 

                    scope ="world",

                    animation_frame ="ObservationDate",

                   title="Evolution of Death Cases") 

fig.show()
fig = px.choropleth(map_data, 

                    locations ="Country/Region", 

                    color ="Recovered", 

                    hover_name='Country/Region',

                    locationmode="country names",

                    color_continuous_scale = px.colors.sequential.Plasma, 

                    scope ="world",

                    animation_frame ="ObservationDate",

                   title="Evolution of Recovered Cases") 

fig.show()
tunisia = data.loc[data['Country/Region'] == 'Tunisia']

tunisia.drop(columns=['Province/State', 'Country/Region', 'Date'], inplace=True)

tunisia.set_index('ObservationDate', inplace=True)
tunisia.head()
print(f'Last Update: {last_available_date}')

plt.figure(figsize=(12,7))

plt.pie(tunisia.loc['08/25/2020'][['Deaths', 'Recovered', 'Active']], labels=['Deaths', 'Recovered', 'Active'], 

        autopct='%1.1f%%', explode=(1,0,0), shadow=True)

plt.title('COVID-19 Tunisia')

plt.show()
plt.figure(figsize=(17,5))

plt.plot(tunisia['Confirmed'])

plt.xticks(tunisia.index[::7],rotation=45)

plt.axvline(x='06/27/2020', label="Borders' opening", color='red', linestyle='--', linewidth='1.7')

plt.title('Evolution of COVID-19 Confirmed Cases (Tunisia)')

plt.xlabel('Date')

plt.ylabel('Confirmed Cases')

plt.legend()

plt.show()
plt.figure(figsize=(17,5))

plt.plot(tunisia['Active'], label='Active')

plt.plot(tunisia['Recovered'], label='Recovered')

plt.xticks(tunisia.index[::7],rotation=45)

plt.axvline(x='06/27/2020', label="Borders' opening", color='black', linestyle='--', linewidth='1.7')

plt.title('Evolution of COVID-19 Active and Recovered Cases (Tunisia)')

plt.xlabel('Date')

plt.legend()

plt.show()
plt.figure(figsize=(17,5))

plt.plot(tunisia['Deaths'])

plt.xticks(tunisia.index[::7],rotation=45)

plt.axvline(x='06/27/2020', label="Borders' opening", color='red', linestyle='--', linewidth='1.7')

plt.title('Evolution of COVID-19 Death Cases (Tunisia)')

plt.xlabel('Date')

plt.ylabel('Death Cases')

plt.legend()

plt.show()
diff_evolution_tunisia = tunisia.copy()

for col in ['Confirmed', 'Recovered', 'Deaths']:

    diff_evolution_tunisia[col] = diff_evolution_tunisia[col].diff()

diff_evolution_tunisia.drop(diff_evolution_tunisia.index[0], inplace=True)
plt.figure(figsize=(20,7))

plt.plot(diff_evolution_tunisia['Confirmed'], label='New Confirmed Cases')

plt.plot(diff_evolution_tunisia['Confirmed'].rolling(window=7).mean(), label='Moving Average, window = 7 days')

plt.plot(diff_evolution_tunisia['Confirmed'].rolling(window=14).mean(), label='Moving Average, window = 14 days')

plt.axvline(x='06/27/2020', label="Borders' opening", color='black', linestyle='--', linewidth='1.7')

plt.xticks(diff_evolution_tunisia.index[::7],rotation=45)

plt.title('Evolution of COVID-19 New Confirmed Cases (Tunisia)')

plt.ylabel('New Confirmed Cases')

plt.legend()

plt.show()
plt.figure(figsize=(20,7))

plt.plot(diff_evolution_tunisia['Deaths'])

plt.axvline(x='06/27/2020', label="Borders' opening", color='red', linestyle='--', linewidth='1.7')

plt.xticks(diff_evolution_tunisia.index[::7],rotation=45)

plt.title('Evolution of COVID-19 New Death Cases (Tunisia)')

plt.ylabel('New Death Cases')

plt.legend()

plt.show()
plt.figure(figsize=(20,7))

for n in [50, 100, 500, 1000, 1500, 2000]:

    tunisia_n = tunisia.loc[tunisia['Confirmed'] >= n]

    plt.plot(np.arange(len(tunisia_n)), tunisia_n['Confirmed'], label=f'Since {n} Cases')

plt.xlabel('Days since n cases')

plt.ylabel('Confirmed Cases')

plt.title('Evolution of COVID-19 Confirmed Cases: Tunisia')

plt.legend()

plt.show()