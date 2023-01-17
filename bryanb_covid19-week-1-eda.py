import collections

import numpy as np

import pandas as pd

import seaborn as sns

from datetime import datetime

from datetime import timedelta

import matplotlib.pyplot as plt



import pycountry

import plotly

import plotly.io as pio

import plotly.express as px



from ipywidgets import interact
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
train.head()
print(f'{train.shape[0]} observations and {train.shape[1]} columns')
train.info()
# Convert Date column as datetime

train['Date_plotly'] = train['Date']

train['Date'] = pd.to_datetime(train['Date'])



print(f"from {min(train['Date'])} to {max(train['Date'])}")
# Rename column names

train = train.rename(columns = {'Province/State':'Province_State',

                                'Country/Region':'Country_Region', 

                                'Lat': 'Latitude', 

                                'Long':'Longitude', 

                                'ConfirmedCases': 'Confirmed_Cases'})
train.isnull().sum()
# Fill NAs with Country_Region

train['Province_State'] = train['Province_State'].fillna(train['Country_Region'])
# Create aggregated dataframe

train_agg = train[['Country_Region', 

                   'Date', 

                   'Confirmed_Cases', 

                   'Fatalities']].groupby(['Country_Region', 'Date'], as_index = False).agg({'Confirmed_Cases':'sum', 'Fatalities':'sum'})
fig = plt.figure(figsize=(15,10))

plt.title('COVID cases evolution in China', fontsize=20)



ax1 = fig.add_subplot(111)

sns.lineplot(x = 'Date', y='Confirmed_Cases', data = train_agg[train_agg['Country_Region'] == 'China'], ax=ax1)

ax1.axvline('2020-02-02', color='red', linestyle='--')



ax2 = ax1.twinx()

sns.lineplot(x = 'Date', y='Fatalities', data = train_agg[train_agg['Country_Region'] == 'China'], color='yellow', ax=ax2)



fig.legend(('Cumulated confirmed cases','Lockdown start date','Cumulated fatalities'),loc="upper right")

ax1.grid()

plt.show()
fig = plt.figure(figsize=(15,10))

plt.title('COVID cases evolution in South Korea', fontsize=20)



ax1 = fig.add_subplot(111)

sns.lineplot(x = 'Date', y='Confirmed_Cases', data = train_agg[train_agg['Country_Region'] == 'Korea, South'], ax=ax1)



ax2 = ax1.twinx()

sns.lineplot(x = 'Date', y='Fatalities', data = train_agg[train_agg['Country_Region'] == 'Korea, South'], color='yellow', ax=ax2)



fig.legend(('Cumulated confirmed cases','Cumulated fatalities'),loc="upper right")

ax1.grid()

plt.show()
fig = plt.figure(figsize=(15,10))

plt.title('COVID cases evolution in Singapore', fontsize=20)



ax1 = fig.add_subplot(111)

sns.lineplot(x = 'Date', y='Confirmed_Cases', data = train_agg[train_agg['Country_Region'] == 'Singapore'], ax=ax1)

ax1.axvline('2020-03-20', color='red', linestyle='--')



ax2 = ax1.twinx()

sns.lineplot(x = 'Date', y='Fatalities', data = train_agg[train_agg['Country_Region'] == 'Singapore'], color='yellow', ax=ax2)



fig.legend(('Cumulated confirmed cases','Lockdown start date','Cumulated fatalities'),loc="upper right")

ax1.grid()

plt.show()
fig = plt.figure(figsize=(15,10))

plt.title('COVID cases evolution in France', fontsize=20)



ax1 = fig.add_subplot(111)

sns.lineplot(x = 'Date', y='Confirmed_Cases', data = train_agg[train_agg['Country_Region'] == 'France'], ax=ax1)



ax2 = ax1.twinx()

sns.lineplot(x = 'Date', y='Fatalities', data = train_agg[train_agg['Country_Region'] == 'France'], color='yellow', ax=ax2)



fig.legend(('Cumulated confirmed cases','Cumulated fatalities'),loc="upper right")

ax1.grid()

plt.show()
fig = plt.figure(figsize=(15,10))

plt.title('COVID cases evolution in Italy', fontsize=20)



ax1 = fig.add_subplot(111)

sns.lineplot(x = 'Date', y='Confirmed_Cases', data = train_agg[train_agg['Country_Region'] == 'Italy'], ax=ax1)

ax1.axvline('2020-03-08', color='red', linestyle='--')



ax2 = ax1.twinx()

sns.lineplot(x = 'Date', y='Fatalities', data = train_agg[train_agg['Country_Region'] == 'Italy'], color='yellow', ax=ax2)



fig.legend(('Cumulated confirmed cases','Lockdown start date','Cumulated fatalities'),loc="upper right")

ax1.grid()

plt.show()
# List of the countries affected by covid during the studied period

countries_with_confirmed_cases = np.ravel(train_agg.loc[:, ['Country_Region','Confirmed_Cases']].groupby(['Country_Region']).sum() == 0)



# Find index of list where value is False

countries_with_confirmed_cases_list = [i for i, value in enumerate(countries_with_confirmed_cases) if value == False] 



# Generate list of countries with at least 1 confirmed case in the study period

confirmed_cases_list = train_agg['Country_Region'].unique()[countries_with_confirmed_cases_list]
@interact

def confirmed_cases_over_time(country = confirmed_cases_list):

    fig = plt.figure(figsize=(15,10))

    plt.title('COVID cases evolution', fontsize=20)

    

    ax1 = fig.add_subplot(111)

    sns.lineplot(x = 'Date', y='Confirmed_Cases', data = train_agg[train_agg['Country_Region'] == country], ax=ax1)



    ax2 = ax1.twinx()

    sns.lineplot(x = 'Date', y='Fatalities', data = train_agg[train_agg['Country_Region'] == country], color='yellow', ax=ax2)



    fig.legend(('Cumulated confirmed cases','Cumulated fatalities'),loc="upper right")

    ax1.grid()

    plt.show()
# Date format supported by plotly

train_agg['Date'] = train_agg['Date'].dt.strftime('%Y-%m-%d')
# Matching function between the ISO code and country names

def fuzzy_search(country):

    try:

        result = pycountry.countries.search_fuzzy(country)

    except Exception:

        return np.nan

    else:

        return result[0].alpha_3



# Manually change name of Korea and Taiwan countries and improve matching

train_agg.loc[train_agg['Country_Region'] == 'Korea, South', 'Country_Region'] = 'Korea, Republic of'

train_agg.loc[train_agg['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'



# ISO mapping for countries in train

iso_map = {country: fuzzy_search(country) for country in train_agg['Country_Region'].unique()}



# Mapping to train

train_agg['iso'] = train_agg['Country_Region'].map(iso_map)



# Many thanks to TeYang Lau for that part
# Map and find countries without any ISO

train_agg['iso'] = train_agg['Country_Region'].map(iso_map)

train_agg[train_agg['iso'].isna()]['Country_Region'].unique()
# Use logscale to have better differenciation of colours in the final plot

train_agg['Fatalities_log10']= np.log10(train_agg['Confirmed_Cases']).replace(-np.inf, 0)
plt.figure(figsize=(15,10))

fig = px.choropleth(train_agg, locations="iso",

                    color="Fatalities_log10",

                    hover_name="Country_Region",

                    animation_frame='Date',

                    color_continuous_scale="Blues")

fig.show()
# Aggregated dataframe sorted by Date

train_agg = train_agg.sort_values(['Date']).reset_index()
dates = pd.to_datetime(train_agg['Date']).apply(lambda x: x.strftime('%Y-%m-%d')).unique()
@interact

def top5(infected = ['Fatalities', 'Confirmed_Cases'] ,date = dates):

    top5_df = train_agg[(train_agg[infected]>0) & (train_agg['Date'] == date)].groupby('Country_Region').max().sort_values([infected], ascending=False)[:5]

    print(top5_df)
# Define covid death rate

train_agg['Rate'] = np.where(train_agg['Fatalities'] != 0, np.round((train_agg['Fatalities']/train_agg['Confirmed_Cases']),4),0)
def top5_covid_death_rate():

    top5_df = train_agg.groupby(['Country_Region']).mean().sort_values(['Rate'], ascending=False)[:5]['Rate']

    print(top5_df)

    

top5_covid_death_rate()
countries_with_fatalities = np.ravel(train_agg.loc[:, ['Country_Region','Fatalities']].groupby(['Country_Region']).sum() < 1)

train_agg['Country_Region'] = train_agg['Country_Region'].sort_values()

countries_with_fatalities_list = [i for i, value in enumerate(countries_with_fatalities) if value == False] 

fatalities_list = np.ravel(sorted(train_agg['Country_Region'].unique()))[countries_with_fatalities_list]
def top5_covid_death_rate_adjusted():

    dicti = {}

    for countries in fatalities_list:

        

        min_day_fatalities = pd.to_datetime(min(train_agg.loc[(train_agg['Country_Region']==countries) & (train_agg['Fatalities']!=0), 'Date']))

        max_day_fatalities = pd.to_datetime(max(train_agg.loc[(train_agg['Country_Region']==countries) & (train_agg['Fatalities']!=0), 'Date']))

        duration = (max_day_fatalities - min_day_fatalities).days

        

        if duration != 0:

            dicti[countries] = sum(train_agg[train_agg['Country_Region']==countries].sort_values(['Rate'], ascending=False)['Rate'])/duration

        else:

            dicti[countries] = 0

    

    # top5 highest rate

    dicti = sorted(dicti.items(), key=lambda elm: elm[1], reverse=True)

    dicti = dict(collections.OrderedDict(dicti))

    

    for i in range(5):

        print(list(dicti.keys())[i], list(dicti.values())[i])



top5_covid_death_rate_adjusted()
train_agg.groupby(['Country_Region']).std().sort_values(['Rate'], ascending=False)[:5]['Rate']