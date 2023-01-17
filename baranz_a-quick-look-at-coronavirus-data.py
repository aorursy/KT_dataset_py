# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import folium



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cov_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid_complete = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

cov_data.head()
cov_data.info()
covid_complete.info()
cov_data.isna().sum()

cov_data['Country/Region'] = cov_data['Country/Region'].replace('Mainland China', 'China')

cov_data['Province/State'] = cov_data['Province/State'].fillna('NA')



covid_complete.isna().sum()

covid_complete['Country/Region'] = covid_complete['Country/Region'].replace('Mainland China', 'China')

covid_complete['Province/State'] = covid_complete['Province/State'].fillna('NA')
countries = covid_complete['Country/Region'].unique().tolist()

print(countries)



# Converting foramt of Date column to datetime

covid_complete['Date'] = covid_complete['Date'].apply(pd.to_datetime)



print('\n {} countries affected untill {} '.format(len(countries), max(covid_complete['Date'])))
print('covid_19_clean_complete.csv\n', covid_complete['Country/Region'].value_counts())

print('covid_19_data.csv\n\n', cov_data['Country/Region'].value_counts())

conf = pd.read_csv('/kaggle/input/httpsgithubcomcssegisanddatacovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

conf.head()
recov = pd.read_csv('/kaggle/input/httpsgithubcomcssegisanddatacovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')

recov.head()
death = pd.read_csv('/kaggle/input/httpsgithubcomcssegisanddatacovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

death.head()
print('Around the world: (Last Update on {}) \n\n {}  Confirmed \n {} Recovered \n {} Dead'.format(max(covid_complete['Date']), covid_complete['Confirmed'].sum(), covid_complete['Recovered'].sum(), covid_complete['Deaths'].sum()))
nutshell = covid_complete.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered'].max()

nutshell.style.background_gradient(cmap='PuBu')
world_map = folium.Map(location=[10, -20], tiles='cartodbpositron', 

               min_zoom=1, max_zoom=4, zoom_start=1.5)



for i in range (0, len(covid_complete)):

    

    folium.CircleMarker(location=[covid_complete.iloc[i]['Lat'], covid_complete.iloc[i]['Long']], 

                      popup='<strong>Country: </strong>'+ str(covid_complete['Country/Region'][i])+

                      '\n<strong>Confirmed: </strong>'+ str(covid_complete['Confirmed'][i])+

                      '\n<strong>Deaths: </strong>'+ str(covid_complete['Deaths'][i])+

                      '\n<strong>Recovered: </strong>'+ str(covid_complete['Recovered'][i]),

                       radius=1,

                       color='red',

                       fill=True,

                       fill_color='red').add_to(world_map)

    





world_map