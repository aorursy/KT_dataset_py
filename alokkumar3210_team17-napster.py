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
import plotly as py

import plotly.express as px

import plotly.graph_objs as go

import matplotlib.pyplot as plt 
#Loading the dataset of COVID-19 in dataframe

df1 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
#Top 5 rows 

df1.head(5)
#Checking the datatype and null values

df1.info()
#df2 = pd.read_excel("../input/testsector/Sample Dataset of Sector Sales.xlsx")
#Rename columns in DataFrame

df1.rename(columns={'ObservationDate':'date','Province/State':'state','Country/Region':'country','Last Update':'last_updated','Confirmed':'confirmed','Deaths':'deaths','Recovered':'recovered'},inplace=True)
#Filling the missing value in state with empty space

df1[['state']]=df1[['state']].fillna('')
df_countries = df1.groupby(['country', 'date']).sum().reset_index().sort_values('date', ascending=False)

df_countries = df_countries.drop_duplicates(subset = ['country'])

df_countries = df_countries[df_countries['confirmed']>0] 

df_countries.head()
#World map of Corona Virus outbreak

fig = go.Figure(data=go.Choropleth(

    locations = df_countries['country'],

    locationmode = 'country names',

    z = df_countries['confirmed'],

    colorscale = 'Reds',

    marker_line_color = 'black',

    marker_line_width = 0.5,

))

fig.update_layout(

    title_text = 'Confirmed Cases as of April 20th, 2020',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

        projection_type = 'equirectangular'

    )

) 

 
#Reading the dataset

confirmed_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

deaths_df= pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

recoveries_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
#Storing the columns of the dataset

cols = confirmed_df.keys()

cols
#Day wise confirmed , deaths and recoveries data

confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]

recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()

world_cases = []

total_deaths = [] 

mortality_rate = []

recovery_rate = [] 

total_recovered = [] 

total_active = [] 



china_cases = [] 

italy_cases = []

us_cases = [] 

spain_cases = [] 

france_cases = [] 

germany_cases = [] 

uk_cases = [] 



china_deaths = [] 

italy_deaths = []

us_deaths = [] 

spain_deaths = [] 

france_deaths = [] 

germany_deaths = [] 

uk_deaths = [] 



china_recoveries = [] 

italy_recoveries = []

us_recoveries = [] 

spain_recoveries = [] 

france_recoveries = [] 

germany_recoveries = [] 

uk_recoveries = [] 



for i in dates:

    confirmed_sum = confirmed[i].sum()

    death_sum = deaths[i].sum()

    recovered_sum = recoveries[i].sum()

    

    # confirmed, deaths, recovered, and active

    world_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    total_recovered.append(recovered_sum)

    total_active.append(confirmed_sum-death_sum-recovered_sum)

    

    # calculate rates

    mortality_rate.append(death_sum/confirmed_sum)

    recovery_rate.append(recovered_sum/confirmed_sum)



    # case studies 



    us_cases.append(confirmed_df[confirmed_df['Country/Region']=='US'][i].sum())

    us_deaths.append(deaths_df[deaths_df['Country/Region']=='US'][i].sum())

    us_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='US'][i].sum())

def daily_increase(data):

    d = [] 

    for i in range(len(data)):

        if i == 0:

            d.append(data[0])

        else:

            d.append(data[i]-data[i-1])

    return d 



# confirmed cases

world_daily_increase = daily_increase(world_cases)

us_daily_increase = daily_increase(us_cases)



# deaths

world_daily_death = daily_increase(total_deaths)

us_daily_death = daily_increase(us_deaths)



# recoveries

world_daily_recovery = daily_increase(total_recovered)

us_daily_recovery = daily_increase(us_recoveries)
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)

total_deaths = np.array(total_deaths).reshape(-1, 1)

total_recovered = np.array(total_recovered).reshape(-1, 1)
days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-10]
import datetime

start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
adjusted_dates = adjusted_dates.reshape(1, -1)[0]

plt.figure(figsize=(16, 9))

plt.bar(adjusted_dates, world_daily_increase)

plt.title('World Daily Increases in Confirmed Cases', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.bar(adjusted_dates, world_daily_death)

plt.title('World Daily Increases in Confirmed Deaths', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.bar(adjusted_dates, world_daily_recovery)

plt.title('World Daily Increases in Confirmed Recoveries', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, np.log10(world_cases))

plt.title('Log of # of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, np.log10(total_deaths))

plt.title('Log of # of Coronavirus Deaths Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, np.log10(total_recovered))

plt.title('Log of # of Coronavirus Recoveries Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
def country_plot(x, y1, y2, y3, y4, country):

    plt.figure(figsize=(16, 9))

    plt.plot(x, y1)

    plt.title('{} Confirmed Cases'.format(country), size=30)

    plt.xlabel('Days Since 1/22/2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()



    plt.figure(figsize=(16, 9))

    plt.bar(x, y2)

    plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)

    plt.xlabel('Days Since 1/22/2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()



    plt.figure(figsize=(16, 9))

    plt.bar(x, y3)

    plt.title('{} Daily Increases in Deaths'.format(country), size=30)

    plt.xlabel('Days Since 1/22/2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()

    

    plt.figure(figsize=(16, 9))

    plt.bar(x, y4)

    plt.title('{} Daily Increases in Recoveries'.format(country), size=30)

    plt.xlabel('Days Since 1/22/2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()
country_plot(adjusted_dates, us_cases, us_daily_increase, us_daily_death, us_daily_recovery, 'United States')
mean_mortality_rate = np.mean(mortality_rate)

plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, mortality_rate, color='orange')

plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')

plt.title('Mortality Rate of Coronavirus Over Time', size=30)

plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)], prop={'size': 20})

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Mortality Rate', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
mean_recovery_rate = np.mean(recovery_rate)

plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, recovery_rate, color='blue')

plt.axhline(y = mean_recovery_rate,linestyle='--', color='black')

plt.title('Recovery Rate of Coronavirus Over Time', size=30)

plt.legend(['recovery rate', 'y='+str(mean_recovery_rate)], prop={'size': 20})

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Recovery Rate', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
# plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, total_deaths, color='r')

plt.plot(adjusted_dates, total_recovered, color='green')

plt.legend(['death', 'recoveries'], loc='best', fontsize=20)

plt.title('# of Coronavirus Cases', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
unique_countries =  list(df1['Country_Region'].unique())

country_df = pd.DataFrame({'Country Name': unique_countries, 'Number of Confirmed Cases': country_confirmed_cases,

                          'Number of Deaths': country_death_cases, 'Number of Recoveries' : country_recovery_cases, 

                          'Number of Active Cases' : country_active_cases,

                          'Mortality Rate': country_mortality_rate})

# number of cases per country/region



country_df.style.background_gradient(cmap='Greens')
