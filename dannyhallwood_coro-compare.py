# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import plotly.express as px



from IPython.display import Javascript

from IPython.core.display import display, HTML



# color pallette

cdr = ['#393e46', '#ff2e63', '#30e3ca'] # grey - red - blue

idr = ['#f8b400', '#ff2e63', '#30e3ca'] # yellow - red - blue



s = '#f0134d'

h = '#12cad6'

e = '#4a47a3'

m = '#42e6a4'

c = '#333333'



shemc = [s, h, e, m, c]

sec = [s, e, c]
# COVID-19

# --------



# covid_19 dataset

covid_19 = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                       parse_dates=['Date'])



# selecting important columns only

covid_19 = covid_19[['Date', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']]



# replacing Mainland china with just China

covid_19['Country/Region'] = covid_19['Country/Region'].replace('Mainland China', 'China')



# renaming columns

covid_19.columns = ['Date', 'Country', 'Cases', 'Deaths', 'Recovered']



# group by date and country

covid_19 = covid_19.groupby(['Date', 'Country'])['Cases', 'Deaths', 'Recovered']

covid_19 = covid_19.sum().reset_index()



# latest

c_lat = covid_19[covid_19['Date'] == max(covid_19['Date'])].reset_index()



# latest grouped by country

c_lat_grp = c_lat.groupby('Country')['Cases', 'Deaths', 'Recovered'].sum().reset_index()



# nth day

covid_19['nth_day'] = (covid_19['Date'] - min(covid_19['Date'])).dt.days



# day by day

c_dbd = covid_19.groupby('Date')['Cases', 'Deaths', 'Recovered'].sum().reset_index()



# nth day

c_dbd['nth_day'] = covid_19.groupby('Date')['nth_day'].max().values



# no. of countries

temp = covid_19[covid_19['Cases']>0]

c_dbd['n_countries'] = temp.groupby('Date')['Country'].apply(len).values



c_dbd['new_cases'] = c_dbd['Cases'].diff()

c_dbd['new_deaths'] = c_dbd['Deaths'].diff()

c_dbd['epidemic'] = 'COVID-19'

# EBOLA

# ------



# ebola dataset

ebola_14 = pd.read_csv("../input/ebola-outbreak-20142016-complete-dataset/ebola_2014_2016_clean.csv", 

                       parse_dates=['Date'])



# selecting important columns only

ebola_14 = ebola_14[['Date', 'Country', 'No. of confirmed, probable and suspected cases',

                     'No. of confirmed, probable and suspected deaths']]



# renaming columns

ebola_14.columns = ['Date', 'Country', 'Cases', 'Deaths']

ebola_14.head()



# group by date and country

ebola_14 = ebola_14.groupby(['Date', 'Country'])['Cases', 'Deaths']

ebola_14 = ebola_14.sum().reset_index()



# filling missing values

ebola_14['Cases'] = ebola_14['Cases'].fillna(0)

ebola_14['Deaths'] = ebola_14['Deaths'].fillna(0)



# converting datatypes

ebola_14['Cases'] = ebola_14['Cases'].astype('int')

ebola_14['Deaths'] = ebola_14['Deaths'].astype('int')



# latest

e_lat = ebola_14[ebola_14['Date'] == max(ebola_14['Date'])].reset_index()



# latest grouped by country

e_lat_grp = e_lat.groupby('Country')['Cases', 'Deaths'].sum().reset_index()



# nth day

ebola_14['nth_day'] = (ebola_14['Date'] - min(ebola_14['Date'])).dt.days



# day by day

e_dbd = ebola_14.groupby('Date')['Cases', 'Deaths'].sum().reset_index()



# nth day

e_dbd['nth_day'] = ebola_14.groupby('Date')['nth_day'].max().values



# no. of countries

temp = ebola_14[ebola_14['Cases']>0]

e_dbd['n_countries'] = temp.groupby('Date')['Country'].apply(len).values



e_dbd['new_cases'] = e_dbd['Cases'].diff()

e_dbd['new_deaths'] = e_dbd['Deaths'].diff()

e_dbd['epidemic'] = 'EBOLA'
# SARS

# ----



# sars dataset

sars_03 = pd.read_csv("../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv", 

                       parse_dates=['Date'])



# selecting important columns only

sars_03 = sars_03[['Date', 'Country', 'Cumulative number of case(s)', 

                   'Number of deaths', 'Number recovered']]



# renaming columns

sars_03.columns = ['Date', 'Country', 'Cases', 'Deaths', 'Recovered']



# group by date and country

sars_03 = sars_03.groupby(['Date', 'Country'])['Cases', 'Deaths', 'Recovered']

sars_03 = sars_03.sum().reset_index()



# latest

s_lat = sars_03[sars_03['Date'] == max(sars_03['Date'])].reset_index()



# latest grouped by country

s_lat_grp = s_lat.groupby('Country')['Cases', 'Deaths', 'Recovered'].sum().reset_index()



# nth day

sars_03['nth_day'] = (sars_03['Date'] - min(sars_03['Date'])).dt.days



# day by day

s_dbd = sars_03.groupby('Date')['Cases', 'Deaths', 'Recovered'].sum().reset_index()



# nth day

s_dbd['nth_day'] = sars_03.groupby('Date')['nth_day'].max().values



# no. of countries

temp = sars_03[sars_03['Cases']>0]

s_dbd['n_countries'] = temp.groupby('Date')['Country'].apply(len).values





s_dbd['new_cases'] = s_dbd['Cases'].diff()

s_dbd['new_deaths'] = s_dbd['Deaths'].diff()

s_dbd['epidemic'] = 'SARS'
# MERS

mers_cntry = pd.read_csv("../input/mers-outbreak-dataset-20122019/country_count_latest.csv")

mers_weekly = pd.read_csv("../input/mers-outbreak-dataset-20122019/weekly_clean.csv")
# sum of cases

# -----------



c_cases = sum(c_lat_grp['Cases'])

c_deaths = sum(c_lat_grp['Deaths'])

c_no_countries = len(c_lat_grp['Country'].value_counts())



s_cases = sum(s_lat_grp['Cases'])

s_deaths = sum(s_lat_grp['Deaths'])

s_no_countries = len(s_lat_grp['Country'].value_counts())



e_cases = sum(e_lat_grp['Cases'])

e_deaths = sum(e_lat_grp['Deaths'])

e_no_countries = len(e_lat_grp['Country'].value_counts())
epidemics = pd.DataFrame({

    'epidemic' : ['COVID-19', 'SARS', 'EBOLA', 'MERS', 'H1N1'],

    'start_year' : [2019, 2003, 2014, 2012, 2009],

    'end_year' : [2020, 2004, 2016, 2017, 2010],

    'confirmed' : [c_cases, s_cases, e_cases, 2494, 6724149],

    'deaths' : [c_deaths, s_deaths, e_deaths, 858, 19654],

    'no_of_countries' : [c_no_countries, s_no_countries, e_no_countries, 27, 178]

})



epidemics['mortality'] = round((epidemics['deaths']/epidemics['confirmed'])*100, 2)

epidemics = epidemics.sort_values('end_year').reset_index(drop=True)

epidemics.head()
fig = px.bar(epidemics.sort_values('confirmed',ascending=False), 

             x="confirmed", y="epidemic", color='epidemic', 

             text='confirmed', orientation='h', title='No. of Cases', 

             range_x=[0,7500000],

             color_discrete_sequence = [h, c, e, s, m])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.bar(epidemics.sort_values('deaths',ascending=False), 

             x="deaths", y="epidemic", color='epidemic', 

             text='deaths', orientation='h', title='No. of Deaths',

             range_x=[0,25000],

             color_discrete_sequence = [h, e, c, m, s])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.bar(epidemics.sort_values('mortality',ascending=False),

             x="mortality", y="epidemic", color='epidemic', 

             text='mortality', orientation='h', title='Moratlity rate', 

             range_x=[0,100],

             color_discrete_sequence = [e, m, s, c, h])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.bar(epidemics.sort_values('no_of_countries', ascending=False),

             x="no_of_countries", y="epidemic", color='epidemic', 

             text='no_of_countries', orientation='h', title='No. of Countries', 

             range_x=[0,200],

             color_discrete_sequence = [h, c, s, m, e])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
temp = pd.concat([s_dbd, e_dbd, c_dbd], axis=0, sort=True)

                

fig = px.line(temp, x="Date", y="Cases", color='epidemic', 

             title='No. of new cases',

             color_discrete_sequence = sec)

fig.update_layout(xaxis_rangeslider_visible=True)





fig = px.line(temp, x="Date", y="Deaths", color='epidemic', 

             title='No. of new deaths',

             color_discrete_sequence = sec)

fig.update_layout(xaxis_rangeslider_visible=True)

fig = px.line(temp, x="nth_day", y="Cases", color='epidemic', 

             title='Cases', color_discrete_sequence = sec)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()



fig = px.line(temp, x="nth_day", y="Deaths", color='epidemic', 

             title='Deaths', color_discrete_sequence = sec)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()



fig = px.line(temp, x="nth_day", y="n_countries", color='epidemic', 

             title='No. of Countries', color_discrete_sequence = sec)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
temp = covid_19.groupby('Date')['Cases'].sum().reset_index()



covid = temp['Cases']

sars = [8096 for i in range(len(temp))]

ebola = [28646 for i in range(len(temp))]

mers = [2494 for i in range(len(temp))]

h1n1 = [6724149 for i in range(len(temp))]



plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(20, 8))

ax = plt.plot(temp['Date'], covid, label='COVID-19 (2019-2020)', c='#555555', alpha=0.8)

ax = plt.plot(temp['Date'], sars, label='SARS (2003-2004)', c='#E71D36', ls='--', alpha=0.8)

ax = plt.plot(temp['Date'], ebola, label='EBOLA (2014-2016)', c='#FF9F1C', ls='--', alpha=0.8)

ax = plt.plot(temp['Date'], mers, label='MERS', c='#2EC4B6', ls='--', alpha=0.8)

plt.title('Number of Cases')

plt.legend()

plt.show()
temp = covid_19.groupby('Date')['Deaths'].sum().reset_index()



covid = temp['Deaths']

sars = [774 for i in range(len(temp))]

ebola = [11323 for i in range(len(temp))]

mers = [858 for i in range(len(temp))]

h1n1 = [19654 for i in range(len(temp))]



plt.figure(figsize=(20, 8))

ax = plt.plot(temp['Date'], covid, label='COVID-19 (2019-2020)', c='#555555', alpha=0.8)

ax = plt.plot(temp['Date'], sars, label='SARS (2003-2004)', c='#E71D36', ls='--', alpha=0.8)

ax = plt.plot(temp['Date'], ebola, label='EBOLA (2014-2016)', c='#FF9F1C', ls='--', alpha=0.8)

ax = plt.plot(temp['Date'], mers, label='MERS', c='#2EC4B6', ls='--', alpha=0.8)

ax = plt.plot(temp['Date'], h1n1, label='H1N1', c='#2345BA', ls='--', alpha=0.8)

plt.title('Number of Deaths')

plt.legend()

plt.show()