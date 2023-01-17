# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# libraries for plotting
import matplotlib.pyplot as plt 
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
confirmed_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
death_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
weather_data_cases = pd.read_csv('/kaggle/input/weather-data-countries-covid19/weather_data_countries_covid19.csv')
weather_data_cases.head()
# we are going to look at country wise data, hence sum up all the cities of that country into one row
confirmed_cases = confirmed_cases.groupby(['Country/Region'],as_index = False).sum().drop(["Lat","Long"],axis=1)
death_cases = death_cases.groupby(['Country/Region'],as_index = False).sum().drop(["Lat","Long"],axis=1)
confirmed_cases.head()
def get_unique_countries(df):
    locations_master = df['Country/Region'].sort_values().unique()
    return list(locations_master)

# print("unique countries list of confirmed cases: {}\nlength: {} ".format(get_unique_countries(confirmed_cases), len(get_unique_countries(confirmed_cases))))
# check if any country is missed in weather and remove that country for confirmed cases
# print("missing countries in cases data: {} ".format(confirmed_cases[~confirmed_cases['Country/Region'].isin(weather_data_cases['Country/Region'])]['Country/Region'].to_list()))
confirmed_cases = confirmed_cases[confirmed_cases['Country/Region'].isin(weather_data_cases['Country/Region'])]

death_cases = death_cases[death_cases['Country/Region'].isin(weather_data_cases['Country/Region'])]
# print("list of confirmed cases: {}\nshape {} ".format(confirmed_cases['Country/Region'].to_list(), confirmed_cases.shape))
# print("unique list of confirmed cases: {}\nlength: {} ".format(get_unique_countries(confirmed_cases), len(get_unique_countries(confirmed_cases))))
weather_data_df_maxtemp = weather_data_cases.loc[weather_data_cases['weather_param'] == 'maxtempC']
weather_data_df_mintemp = weather_data_cases.loc[weather_data_cases['weather_param'] == 'mintempC']
weather_data_df_humidity = weather_data_cases.loc[weather_data_cases['weather_param'] == 'humidity']

weather_data_df_maxtemp = weather_data_df_maxtemp.groupby(['Country/Region']).sum()
weather_data_df_mintemp = weather_data_df_mintemp.groupby(['Country/Region']).sum()
weather_data_df_humidity = weather_data_df_humidity.groupby(['Country/Region']).sum()
cols = confirmed_cases.keys()
confirmed = confirmed_cases.loc[:, cols[1:-1]]
deaths = death_cases.loc[:, cols[1:-1]]
confirmed = confirmed.set_index(confirmed_cases['Country/Region'])
deaths = deaths.set_index(death_cases['Country/Region'])
confirmed_cases1 = confirmed_cases.groupby(['Country/Region']).sum()
death_cases1 = death_cases.groupby(['Country/Region']).sum()
total_cases = pd.merge(confirmed_cases1.sum(axis=1).reset_index(name ='Total_Confirmed_Cases'), death_cases1.sum(axis=1).reset_index(name ='Total_Death_Cases'), on="Country/Region")
total_cases.style.background_gradient(cmap='Wistia')
china_cases = confirmed_cases1.loc['China'].to_numpy()
italy_cases = confirmed_cases1.loc['Italy'].to_numpy()
us_cases = confirmed_cases1.loc['US'].to_numpy()
india_cases = confirmed_cases1.loc['India'].to_numpy()

china_death_cases = death_cases1.loc['China'].to_numpy()
italy_death_cases = death_cases1.loc['Italy'].to_numpy()
us_death_cases = death_cases1.loc['US'].to_numpy()
india_death_cases = death_cases1.loc['India'].to_numpy()

china_maxtemp = weather_data_df_maxtemp.loc['China'].to_numpy()
italy_maxtemp = weather_data_df_maxtemp.loc['Italy'].to_numpy()
us_maxtemp = weather_data_df_maxtemp.loc['US'].to_numpy()
india_maxtemp = weather_data_df_maxtemp.loc['India'].to_numpy()

china_humidity = weather_data_df_humidity.loc['China'].to_numpy()
italy_humidity = weather_data_df_humidity.loc['Italy'].to_numpy()
us_humidity = weather_data_df_humidity.loc['US'].to_numpy()
india_humidity = weather_data_df_humidity.loc['India'].to_numpy()

china_mintemp = weather_data_df_mintemp.loc['China'].to_numpy()
italy_mintemp = weather_data_df_mintemp.loc['Italy'].to_numpy()
us_mintemp = weather_data_df_mintemp.loc['US'].to_numpy()
india_mintemp = weather_data_df_mintemp.loc['India'].to_numpy()

days = np.array([i for i in range(len(confirmed.keys()))]).reshape(-1, 1)
limit_start = 0
limit_end = len(days)
plt.figure(figsize=(15, 12))

plt.plot(days[limit_start:limit_end], us_cases[limit_start:limit_end], 'r-o')
plt.plot(days[limit_start:limit_end], china_cases[limit_start:limit_end], 'g-o')
plt.plot(days[limit_start:limit_end], italy_cases[limit_start:limit_end], 'b-o')
plt.plot(days[limit_start:limit_end], india_cases[limit_start:limit_end], 'm-o')

#Add a variation of the original cases with weather data, this is done to show a parallel curve along the original recorded cases
plt.plot(days[limit_start:limit_end], us_cases[limit_start:limit_end]+500+500*us_maxtemp[limit_start:limit_end], 'r--o')
plt.plot(days[limit_start:limit_end], china_cases[limit_start:limit_end]+500+500*china_maxtemp[limit_start:limit_end], 'g--o')
plt.plot(days[limit_start:limit_end], italy_cases[limit_start:limit_end]+500+500*italy_maxtemp[limit_start:limit_end], 'b--o')
plt.plot(days[limit_start:limit_end], india_cases[limit_start:limit_end]+500+500*india_maxtemp[limit_start:limit_end], 'm--o')

plt.title('Coronavirus confirmed cases w.r.t max temperature', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Number of Cases', size=15)
plt.legend(['US_cases', 'China_cases', 'Italy_cases','India_cases', 'US_maxt', 'China_maxt', 'Italy_maxt', 'India_maxt'], prop={'size': 15})
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(15, 12))

plt.plot(days[limit_start:limit_end], us_cases[limit_start:limit_end], 'r-o')
plt.plot(days[limit_start:limit_end], china_cases[limit_start:limit_end], 'g-o')
plt.plot(days[limit_start:limit_end], italy_cases[limit_start:limit_end], 'b-o')
plt.plot(days[limit_start:limit_end], india_cases[limit_start:limit_end], 'm-o')

#Add a variation of the original cases with weather data, this is done to show a parallel curve along the original recorded cases
plt.plot(days[limit_start:limit_end], us_cases[limit_start:limit_end]+500+500*us_humidity[limit_start:limit_end], 'r--o')
plt.plot(days[limit_start:limit_end], china_cases[limit_start:limit_end]+500+500*china_humidity[limit_start:limit_end], 'g--o')
plt.plot(days[limit_start:limit_end], italy_cases[limit_start:limit_end]+500+500*italy_humidity[limit_start:limit_end], 'b--o')
plt.plot(days[limit_start:limit_end], india_cases[limit_start:limit_end]+500+500*india_humidity[limit_start:limit_end], 'm--o')

plt.title('Coronavirus confirmed cases w.r.t humidity', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Number of Cases', size=15)
plt.legend(['US_cases', 'China_cases', 'Italy_cases','India_cases', 'US_humidity', 'China_humidity', 'Italy_humidity','India_humidity'], prop={'size': 15})
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(15, 12))

plt.plot(days[limit_start:limit_end], us_cases[limit_start:limit_end], 'r-o')
plt.plot(days[limit_start:limit_end], china_cases[limit_start:limit_end], 'g-o')
plt.plot(days[limit_start:limit_end], italy_cases[limit_start:limit_end], 'b-o')
plt.plot(days[limit_start:limit_end], india_cases[limit_start:limit_end], 'm-o')

#Add a variation of the original cases with weather data, this is done to show a parallel curve along the original recorded cases
plt.plot(days[limit_start:limit_end], us_cases[limit_start:limit_end]+500+500*us_mintemp[limit_start:limit_end], 'r--o')
plt.plot(days[limit_start:limit_end], china_cases[limit_start:limit_end]+500+500*china_mintemp[limit_start:limit_end], 'g--o')
plt.plot(days[limit_start:limit_end], italy_cases[limit_start:limit_end]+500+500*italy_mintemp[limit_start:limit_end], 'b--o')
plt.plot(days[limit_start:limit_end], india_cases[limit_start:limit_end]+500+500*india_mintemp[limit_start:limit_end], 'm--o')

plt.title('Coronavirus confirmed cases w.r.t minTemp', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Number of Cases', size=15)
plt.legend(['US_cases', 'China_cases', 'Italy_cases','India_cases', 'US_mintemp', 'China_mintemp', 'Italy_mintemp','India_mintemp'], prop={'size': 15})
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(15, 12))

plt.plot(days[limit_start:limit_end], us_death_cases[limit_start:limit_end], 'r-o')
plt.plot(days[limit_start:limit_end], china_death_cases[limit_start:limit_end], 'g-o')
plt.plot(days[limit_start:limit_end], italy_death_cases[limit_start:limit_end], 'b-o')
plt.plot(days[limit_start:limit_end], india_death_cases[limit_start:limit_end], 'm-o')

#Add a variation of the original cases with weather data, this is done to show a parallel curve along the original recorded cases
plt.plot(days[limit_start:limit_end], us_death_cases[limit_start:limit_end]+50+100*us_maxtemp[limit_start:limit_end], 'r--o')
plt.plot(days[limit_start:limit_end], china_death_cases[limit_start:limit_end]+100+100*china_maxtemp[limit_start:limit_end], 'g--o')
plt.plot(days[limit_start:limit_end], italy_death_cases[limit_start:limit_end]+100+100*italy_maxtemp[limit_start:limit_end], 'b--o')
plt.plot(days[limit_start:limit_end], india_death_cases[limit_start:limit_end]+50+100*india_maxtemp[limit_start:limit_end], 'm--o')

plt.title('Coronavirus death cases w.r.t max temperature', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Number of Cases', size=15)
plt.legend(['US_cases', 'China_cases', 'Italy_cases','India_cases', 'US_maxt', 'China_maxt', 'Italy_maxt', 'India_maxt'], prop={'size': 15})
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(15, 12))

plt.plot(days[limit_start:limit_end], us_death_cases[limit_start:limit_end], 'r-o')
plt.plot(days[limit_start:limit_end], china_death_cases[limit_start:limit_end], 'g-o')
plt.plot(days[limit_start:limit_end], italy_death_cases[limit_start:limit_end], 'b-o')
plt.plot(days[limit_start:limit_end], india_death_cases[limit_start:limit_end], 'm-o')

#Add a variation of the original cases with weather data, this is done to show a parallel curve along the original recorded cases
plt.plot(days[limit_start:limit_end], us_death_cases[limit_start:limit_end]+50+50*us_humidity[limit_start:limit_end], 'r--o')
plt.plot(days[limit_start:limit_end], china_death_cases[limit_start:limit_end]+50+50*china_humidity[limit_start:limit_end], 'g--o')
plt.plot(days[limit_start:limit_end], italy_death_cases[limit_start:limit_end]+50+50*italy_humidity[limit_start:limit_end], 'b--o')
plt.plot(days[limit_start:limit_end], india_death_cases[limit_start:limit_end]+50+50*india_humidity[limit_start:limit_end], 'm--o')

plt.title('Coronavirus confirmed cases w.r.t humidity', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Number of Cases', size=15)
plt.legend(['US_cases', 'China_cases', 'Italy_cases','India_cases', 'US_humidity', 'China_humidity', 'Italy_humidity','India_humidity'], prop={'size': 15})
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(15, 12))

plt.plot(days[limit_start:limit_end], us_death_cases[limit_start:limit_end], 'r-o')
plt.plot(days[limit_start:limit_end], china_death_cases[limit_start:limit_end], 'g-o')
plt.plot(days[limit_start:limit_end], italy_death_cases[limit_start:limit_end], 'b-o')
plt.plot(days[limit_start:limit_end], india_death_cases[limit_start:limit_end], 'm-o')

#Add a variation of the original cases with weather data, this is done to show a parallel curve along the original recorded cases
plt.plot(days[limit_start:limit_end], us_death_cases[limit_start:limit_end]+100+100*us_mintemp[limit_start:limit_end], 'r--o')
plt.plot(days[limit_start:limit_end], china_death_cases[limit_start:limit_end]+100+100*china_mintemp[limit_start:limit_end], 'g--o')
plt.plot(days[limit_start:limit_end], italy_death_cases[limit_start:limit_end]+100+100*italy_mintemp[limit_start:limit_end], 'b--o')
plt.plot(days[limit_start:limit_end], india_death_cases[limit_start:limit_end]+100+100*india_mintemp[limit_start:limit_end], 'm--o')

plt.title('Coronavirus confirmed cases w.r.t minTemp', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Number of Cases', size=15)
plt.legend(['US_cases', 'China_cases', 'Italy_cases','India_cases', 'US_mintemp', 'China_mintemp', 'Italy_mintemp','India_mintemp'], prop={'size': 15})
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()