import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import seaborn as sns

import pandas as pd 

import random



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline 

confirmed_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

#confirmed_df.head()
deaths_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

#deaths_df.head()
recoveries_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

#recoveries_df.head()
dates = confirmed_df.columns[4:]

dataframe_list = list()

for date in dates:

  ObservationDate = [date for _ in range(confirmed_df.shape[0])]

  Province_State = list(confirmed_df['Province/State'])

  Country_Region = list(confirmed_df['Country/Region'])

  Confirmed = list(confirmed_df[date])

  Recoveries = list(recoveries_df[date])

  Deaths = list(deaths_df[date])

  dict_temp = {'ObservationDate': ObservationDate, 'Province/State': Province_State, 'Country/Region': Country_Region, 'Confirmed': Confirmed, 'Recoveries': Recoveries, 'Deaths': Deaths}

  df_temp = pd.DataFrame(dict_temp)

  dataframe_list.append(df_temp)

df = pd.concat(dataframe_list) 

df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])

#df.head()
countries = df['Country/Region'].unique()

dates = df.ObservationDate.unique()

dataframe_countries = dict()



for countryName in countries:

  df_country = df.loc[df['Country/Region'] == countryName]

  df_country_dates = pd.DataFrame(columns = ['ObservationDate', 'Confirmed',	'Deaths',	'Recoveries'])

  dict_df = dict()

  prev_Confirmed = 0

  prev_Deaths = 0

  prev_Recoveries = 0

  first = True

  for date in dates:

    dict_df[date] = df_country.loc[df_country['ObservationDate'] == date, ['Province/State',	'Country/Region', 'Confirmed',	'Deaths',	'Recoveries']]

    if first:

      first = False

      prev_Confirmed = df_country.loc[df_country['ObservationDate'] == date, 'Confirmed'].sum()

      prev_Deaths = df_country.loc[df_country['ObservationDate'] == date, 'Deaths'].sum()

      prev_Recoveries = df_country.loc[df_country['ObservationDate'] == date, 'Recoveries'].sum()

    else:

      curr_Confirmed = df_country.loc[df_country['ObservationDate'] == date, 'Confirmed'].sum()

      curr_Deaths = df_country.loc[df_country['ObservationDate'] == date, 'Deaths'].sum()

      curr_Recoveries = df_country.loc[df_country['ObservationDate'] == date, 'Recoveries'].sum()

      Confirmed = curr_Confirmed - prev_Confirmed

      Deaths = curr_Deaths - prev_Deaths

      Recoveries = curr_Recoveries - prev_Recoveries

      prev_Confirmed = curr_Confirmed

      prev_Deaths = curr_Deaths

      prev_Recoveries = curr_Recoveries  

      data = {'ObservationDate': date, 'Confirmed': Confirmed,	'Deaths': Deaths,	'Recoveries': Recoveries}

      df_country_dates = df_country_dates.append(data, ignore_index=True)

  #df_country_dates.sort_values("Confirmed", axis = 0, ascending = False, inplace = True)

  dataframe_countries[countryName] = df_country_dates
top_countries_dict = dict()

last_updated_date = confirmed_df.columns[-1]

for countryName in countries:

  top_countries_dict[countryName] = confirmed_df.loc[confirmed_df['Country/Region'] == countryName, last_updated_date].sum()

top_countries_Confirmed_list = sorted(top_countries_dict.items(), key = lambda kv:kv[1], reverse = True)[:10]

top_countries_Confirmed_list = [var[0] for var in top_countries_Confirmed_list]

#top_countries_Confirmed_list
last_days = 15

plt.figure(figsize=(15, 10))

myCountryList = top_countries_Confirmed_list

count = 1

for countryName in myCountryList:

  x = dataframe_countries[countryName].ObservationDate

  y = dataframe_countries[countryName].Confirmed

  #print(x[len(x) - last_days: ], y[len(y) - last_days: ])

  plt.plot(x[len(x) - last_days: ], y[len(y) - last_days: ], linestyle='dashdot')

  count += 1

  if count > 4:

      break

plt.xticks(rotation=50, size=15)

plt.title("No of Coronavirus Confirmed cases (Top 10 Countries) in last '{}' days".format(last_days), size=30)

plt.xlabel('Date', size=30)

plt.ylabel('No of Cases', size=30)

plt.legend(myCountryList)

plt.show()
plt.figure(figsize=(15, 10))

myCountryList = top_countries_Confirmed_list[1:]

for countryName in myCountryList:

  x = dataframe_countries[countryName].ObservationDate

  y = dataframe_countries[countryName].Recoveries

  plt.plot(x[len(x) - last_days: ], y[len(y) - last_days: ], linestyle='dashdot')

plt.xticks(rotation=50, size=15)

plt.title("No of Coronavirus Recoveries cases (Top 10 Countries) in last '{}' days".format(last_days), size=30)

plt.xlabel('Date', size=30)

plt.ylabel('No of Recoveries cases', size=30)

plt.legend(myCountryList)

plt.show()

plt.figure(figsize=(15, 10))

myCountryList = top_countries_Confirmed_list

for countryName in myCountryList:

  x = dataframe_countries[countryName].ObservationDate

  y = dataframe_countries[countryName].Deaths

  plt.plot(x[len(x) - last_days: ], y[len(y) - last_days: ], linestyle='dashdot')

plt.xticks(rotation=50, size=15)

plt.title("No of Coronavirus Death cases (Top 10 Countries) in last '{}' days".format(last_days), size=30)

plt.xlabel('Date', size=30)

plt.ylabel('No of Deaths cases', size=30)

plt.legend(myCountryList)

plt.show()

print('Recovery rate Country wise')

top_countries_dict = dict()

recovery_rate = dict()

last_updated_date = recoveries_df.columns[-1]

for countryName in countries:

  recovery = recoveries_df.loc[recoveries_df['Country/Region'] == countryName, last_updated_date].sum()

  confirmed = confirmed_df.loc[confirmed_df['Country/Region'] == countryName, last_updated_date].sum()

  rate = float('%.2f' % (recovery / confirmed))

  recovery_rate[countryName] = rate

recovery_rate_list = sorted(recovery_rate.items(), key = lambda kv:kv[1], reverse = True)

recovery_rate_list = [(var[0], var[1]) for var in recovery_rate_list if var[1] > 0]

recovery_rate_list
print('Death rate Country wise')

top_countries_dict = dict()

death_rate = dict()

last_updated_date = deaths_df.columns[-1]

for countryName in countries:

  death = deaths_df.loc[deaths_df['Country/Region'] == countryName, last_updated_date].sum()

  confirmed = confirmed_df.loc[confirmed_df['Country/Region'] == countryName, last_updated_date].sum()

  rate = float('%.2f' % (death / confirmed))

  death_rate[countryName] = rate

death_rate_list = sorted(death_rate.items(), key = lambda kv:kv[1], reverse = True)

death_rate_list = [(var[0], var[1]) for var in death_rate_list if var[1] > 0]

death_rate_list
last_updated_date = deaths_df.columns[-1]

df_lastUpdate = pd.DataFrame(columns = ['Country/Region', 'Confirmed',	'Deaths',	'Recoveries'])

for countryName in countries:

  death = deaths_df.loc[deaths_df['Country/Region'] == countryName, last_updated_date].sum()

  confirmed = confirmed_df.loc[confirmed_df['Country/Region'] == countryName, last_updated_date].sum()

  recoveries = recoveries_df.loc[recoveries_df['Country/Region'] == countryName, last_updated_date].sum()

  temp_dict = {'Country/Region': countryName, 'Confirmed': confirmed,	'Deaths': death,	'Recoveries': recoveries}

  df_lastUpdate = df_lastUpdate.append(temp_dict, ignore_index=True)

#df_lastUpdate.head()
df_lastUpdate.sort_values("Confirmed", axis = 0, ascending = False, inplace = True)



total_countries = 10

displayFirst = False

if displayFirst:

  lower = 0

else: 

  lower = 1

upper = total_countries + lower



c = random.choices(list(mcolors.CSS4_COLORS.values()),k = total_countries)

plt.figure(figsize=(15,15))

plt.title('Covid-19 Confirmed Cases per Country (Top 10, except China)')

plt.pie(df_lastUpdate['Confirmed'][lower:upper], colors=c)

plt.legend(df_lastUpdate['Country/Region'][lower:upper], loc=1)

plt.show()
df_lastUpdate.sort_values("Deaths", axis = 0, ascending = False, inplace = True)



total_countries = 10

displayFirst = False

if displayFirst:

  lower = 0

else: 

  lower = 1

upper = total_countries + lower



c = random.choices(list(mcolors.CSS4_COLORS.values()),k = total_countries)

plt.figure(figsize=(15,15))

plt.title('Covid-19 Death Cases per Country (Top 10, except China)')

plt.pie(df_lastUpdate['Deaths'][lower:upper], colors=c)

plt.legend(df_lastUpdate['Country/Region'][lower:upper], loc=1)

plt.show()
df_lastUpdate.sort_values("Recoveries", axis = 0, ascending = False, inplace = True)



total_countries = 10

displayFirst = False

if displayFirst:

  lower = 0

else: 

  lower = 1

upper = total_countries + lower



c = random.choices(list(mcolors.CSS4_COLORS.values()),k = total_countries)

plt.figure(figsize=(15,15))

plt.title('Covid-19 Recoveries Cases per Country (Top 10, except China)')

plt.pie(df_lastUpdate['Recoveries'][lower:upper], colors=c)

plt.legend(df_lastUpdate['Country/Region'][lower:upper], loc=1)

plt.show()
last_updated_date = deaths_df.columns[-1]

prev_updated_date = deaths_df.columns[-2]

df_lastDay = pd.DataFrame(columns = ['Country/Region', 'Confirmed',	'Deaths',	'Recoveries'])

for countryName in countries:

  total_death = deaths_df.loc[deaths_df['Country/Region'] == countryName, last_updated_date].sum()

  total_confirmed = confirmed_df.loc[confirmed_df['Country/Region'] == countryName, last_updated_date].sum()

  total_recoveries = recoveries_df.loc[recoveries_df['Country/Region'] == countryName, last_updated_date].sum()

  prev_death = deaths_df.loc[deaths_df['Country/Region'] == countryName, prev_updated_date].sum()

  prev_confirmed = confirmed_df.loc[confirmed_df['Country/Region'] == countryName, prev_updated_date].sum()

  prev_recoveries = recoveries_df.loc[recoveries_df['Country/Region'] == countryName, prev_updated_date].sum()

  confirmed = total_confirmed - prev_confirmed

  death = total_death - prev_death

  recoveries = total_recoveries - prev_recoveries

  temp_dict = {'Country/Region': countryName, 'Confirmed': confirmed,	'Deaths': death,	'Recoveries': recoveries}

  df_lastDay = df_lastDay.append(temp_dict, ignore_index=True)

#df_lastDay.head()
df_lastDay.sort_values("Confirmed", axis = 0, ascending = False, inplace = True)



total_countries = 10

displayFirst = True

if displayFirst:

  lower = 0

else: 

  lower = 1

upper = total_countries + lower



c = random.choices(list(mcolors.CSS4_COLORS.values()),k = total_countries)

plt.figure(figsize=(15,15))

plt.title('Covid-19 Confirmed Cases per Country (Top 15, last day)')

plt.pie(df_lastDay['Confirmed'][lower:upper], colors=c)

plt.legend(df_lastDay['Country/Region'][lower:upper], loc=1)

plt.show()
df_lastDay.sort_values("Deaths", axis = 0, ascending = False, inplace = True)



total_countries = 10

displayFirst = True

if displayFirst:

  lower = 0

else: 

  lower = 1

upper = total_countries + lower



c = random.choices(list(mcolors.CSS4_COLORS.values()),k = total_countries)

plt.figure(figsize=(15,15))

plt.title('Covid-19 Deaths Cases per Country (Top 15, last day)')

plt.pie(df_lastDay['Deaths'][lower:upper], colors=c)

plt.legend(df_lastDay['Country/Region'][lower:upper], loc=1)

plt.show()
df_lastDay.sort_values("Recoveries", axis = 0, ascending = False, inplace = True)



total_countries = 10

displayFirst = True

if displayFirst:

  lower = 0

else: 

  lower = 1

upper = total_countries + lower



c = random.choices(list(mcolors.CSS4_COLORS.values()),k = total_countries)

plt.figure(figsize=(15,15))

plt.title('Covid-19 Recoveries Cases per Country (Top 15, last day)')

plt.pie(df_lastDay['Recoveries'][lower:upper], colors=c)

plt.legend(df_lastDay['Country/Region'][lower:upper], loc=1)

plt.show()
last_updated_date = deaths_df.columns[-1]



total_death = deaths_df[last_updated_date].sum()

total_confirmed = confirmed_df[last_updated_date].sum()

total_recoveries = recoveries_df[last_updated_date].sum()

total_ongoingTreatement = total_confirmed - (total_death + total_recoveries)



c = random.choices(list(mcolors.CSS4_COLORS.values()),k = 3)

plt.figure(figsize=(10,10))

plt.title('Covid-19 summary')

plt.pie([total_ongoingTreatement, total_death, total_recoveries], colors=c)

plt.legend(['Being Treated', 'Death cases', 'Recoveries cases'], loc=1)

plt.show()