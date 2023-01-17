# importing libraries



import datetime

import os

import sys

import random



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



from plotly import tools, subplots

import plotly.offline as py

from plotly.offline import plot

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio



from datetime import datetime
confirmed_global_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

confirmed_global = pd.read_csv(confirmed_global_url)



deaths_global_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

deaths_global = pd.read_csv(deaths_global_url)



recovered_global_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

recovered_global = pd.read_csv(recovered_global_url)



confirmed_us_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"

confirmed_us = pd.read_csv(confirmed_us_url)



deaths_us_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"

deaths_us = pd.read_csv(deaths_us_url)
confirmed_global
# Renaming the columns

confirmed_global.rename(columns = {

    'Country/Region':'Country_Region',

    'Province/State': 'Province_State',

    'Longitude': 'Long',

    'Latitude': 'Lat'

}, inplace = True)



recovered_global.rename(columns = {

    'Country/Region':'Country_Region',

    'Province/State': 'Province_State',

    'Longitude': 'Long',

    'Latitude': 'Lat'

}, inplace = True)



deaths_global.rename(columns = {

    'Country/Region':'Country_Region',

    'Province/State': 'Province_State',

    'Longitude': 'Long',

    'Latitude': 'Lat'

}, inplace = True)
# calculating the NaN values

print(confirmed_global.isnull().sum())

print(recovered_global.isnull().sum())

print(deaths_global.isnull().sum())
# replacing Nan values to string values 'nan'

confirmed_global["Province_State"].fillna("nan", inplace = True)

recovered_global["Province_State"].fillna("nan", inplace = True)

deaths_global["Province_State"].fillna("nan", inplace = True)
confirmed_global.Country_Region.unique()
confirmed_global = confirmed_global[~confirmed_global["Province_State"].str.match('Diamond Princess')]

confirmed_global = confirmed_global[~confirmed_global["Country_Region"].str.match('Diamond Princess')]



deaths_global = deaths_global[~deaths_global["Province_State"].str.match('Diamond Princess')]

deaths_global = deaths_global[~deaths_global["Country_Region"].str.match('Diamond Princess')]



recovered_global = recovered_global[~recovered_global["Province_State"].str.match('Diamond Princess')]

recovered_global = recovered_global[~recovered_global["Country_Region"].str.match('Diamond Princess')]
# declaring function for converting Date formats

def convert_date(data):

    try:

        data.columns = list(data.columns[:4]) + [datetime.strptime(dt, "%m/%d/%y").date().strftime("%Y-%m-%d") for dt in data.columns[4:]]

    except:

        data.columns = list(data.columns[:4]) + [datetime.strptime(dt, "%m/%d/%Y").date().strftime("%Y-%m-%d") for dt in data.columns[4:]]
# calling function to change date formats

convert_date(confirmed_global)

convert_date(recovered_global)

convert_date(deaths_global)
confirmed_global
confirmed_global_df = confirmed_global.melt(id_vars = ['Country_Region','Province_State','Lat','Long'],

                                            value_vars = confirmed_global.columns[4:],

                                            var_name = 'Date',

                                            value_name = 'Confirmed_Cases')
deaths_global_df = deaths_global.melt(id_vars = ['Country_Region','Province_State','Lat','Long'],

                                            value_vars = confirmed_global.columns[4:],

                                            var_name = 'Date',

                                            value_name = 'Deaths')
recovered_global_df = recovered_global.melt(id_vars = ['Country_Region','Province_State','Lat','Long'],

                                            value_vars = confirmed_global.columns[4:],

                                            var_name = 'Date',

                                            value_name = 'Recovered')
recovered_global_df
train = confirmed_global_df.merge(deaths_global_df, on = ['Country_Region', 'Province_State','Date'])

train = train.merge(recovered_global_df, on = ['Country_Region', 'Province_State','Date'])
global_dataset = train.groupby('Date')['Confirmed_Cases','Recovered','Deaths'].sum().reset_index()
global_dataset['Daily_Rise'] = global_dataset['Confirmed_Cases'] - global_dataset['Confirmed_Cases'].shift(1)

global_dataset['Mortality_Rate'] = global_dataset['Deaths']/ global_dataset['Confirmed_Cases']

global_dataset
global_dataset_df = pd.melt(global_dataset,

                           id_vars = ['Date'],

                           value_vars = ['Confirmed_Cases','Recovered', 'Mortality_Rate', 'Deaths', 'Daily_Rise'])

global_dataset_df
visual_confirmed = global_dataset_df[global_dataset_df["variable"].str.match('Confirmed_Cases')]

visual_deaths = global_dataset_df[global_dataset_df["variable"].str.match('Deaths')]

visual_recovered = global_dataset_df[global_dataset_df["variable"].str.match('Recovered')]

visual_mortality = global_dataset_df[global_dataset_df["variable"].str.match('Mortality_Rate')]

visual_daily_rise = global_dataset_df[global_dataset_df["variable"].str.match('Daily_Rise')]
visual_confirmed
visual_deaths
visual_recovered
visual_mortality
visual_daily_rise
fig = px.line(global_dataset_df,

             x = 'Date',

             y = 'value',

             color = 'variable',

             title = 'Global Confirmed/ Deaths/ REcovered/ cases with Mortality and Daily Rises')

fig.show()
fig = px.line(global_dataset_df,

             x = 'Date',

             y = 'value',

             color = 'variable',

             title = 'Global Confirmed/ Deaths/ REcovered/ cases with Mortality and Daily Rises (Logrithmic)',

             log_y = True)

fig.show()
fig = px.line(visual_confirmed,

             x = 'Date',

             y = 'value',

             color = 'variable',

             title = 'Confirmed cases over time(Globally)')

fig.show()
fig = px.line(visual_deaths,

             x = 'Date',

             y = 'value',

             color = 'variable',

             title = 'Deaths reported over time(Globally)')

fig.show()
fig = px.line(visual_mortality,

             x = 'Date',

             y = 'value',

             color = 'variable',

             title = 'Change in Mortality Rate over time')

fig.show()
fig = px.line(visual_daily_rise,

             x = 'Date',

             y = 'value',

             color = 'variable',

             title = 'Daily rise of the infected people(Globally)')

fig.show()
country_wise = train.groupby(['Country_Region','Province_State','Date'])['Confirmed_Cases','Recovered','Deaths'].sum().reset_index()
country_wise
country_wise = pd.melt(country_wise,

                           id_vars = ['Date','Country_Region','Province_State'],

                           value_vars = ['Confirmed_Cases','Recovered', 'Deaths'])

country_wise
country_wise_visual_confirmed = country_wise[country_wise["variable"].str.match('Confirmed_Cases')]

country_wise_visual_deaths = country_wise[country_wise["variable"].str.match('Deaths')]

country_wise_visual_recovered = country_wise[country_wise["variable"].str.match('Recovered')]
country_wise
fig = px.line(country_wise_visual_confirmed,

             x = 'Date',

             y = 'value',

             color = 'Country_Region',

             title = 'Confirmed cases')

fig.show()
fig = px.line(country_wise_visual_deaths,

             x = 'Date',

             y = 'value',

             color = 'Country_Region',

             title = 'People that are missed')

fig.show()
fig = px.line(country_wise_visual_recovered,

             x = 'Date',

             y = 'value',

             color = 'Country_Region',

             title = 'Recovered cases')

fig.show()
current_date = country_wise_visual_confirmed['Date'][country_wise_visual_confirmed.index[-1]]
# finding out total deaths, confirmed cases and recovered cases 

country_wise_total_confirmed = country_wise_visual_confirmed[country_wise_visual_confirmed["Date"].str.match(current_date)]

country_wise_total_deaths = country_wise_visual_deaths[country_wise_visual_deaths["Date"].str.match(current_date)]

country_wise_total_recovered = country_wise_visual_recovered[country_wise_visual_recovered["Date"].str.match(current_date)]
country_wise_total_deaths
country_wise_total_recovered
country_wise_total_confirmed
worst_hit_countries_30 = country_wise_total_confirmed.sort_values(by = 'value', ascending = False).head(30)



plt.figure(figsize = (12, 10))

sns.barplot(data = worst_hit_countries_30, y = 'Country_Region', x = 'value', hue = 'Country_Region', dodge = False)

plt.legend(loc = 'lower right')

plt.xlabel('Total Confirmed reported')

plt.ylabel('Countries')

plt.title('Worst 30 countries hit by coronavirus(Confirmed)')

plt.show()
worst_hit_countries_30 = country_wise_total_deaths.sort_values(by = 'value', ascending = False).head(30)



plt.figure(figsize = (12, 10))

sns.barplot(data = worst_hit_countries_30, y = 'Country_Region', x = 'value', hue = 'Country_Region', dodge = False)

plt.legend(loc = 'lower right')

plt.xlabel('Most number of people missed')

plt.ylabel('Countries')

plt.title('Worst 30 countries hit by coronavirus(Deaths)')

plt.show()
recovering_countries_30 = country_wise_total_recovered.sort_values(by = 'value', ascending = False).head(30)



fig = px.bar(recovering_countries_30,

             x='value', y='Country_Region', color='Country_Region', barmode='relative',

             title=f'Most Recovered', text='value', height=1500, width = 950, orientation='h')

fig.show()
locations = confirmed_global[confirmed_global["Province_State"].str.match('nan')].reset_index()

locations = locations[['Country_Region','Lat','Long']]

locations
country_wise_total_confirmed = country_wise_total_confirmed.groupby(['Country_Region','Date'])[['value']].sum().reset_index()

country_wise_total_deaths = country_wise_total_deaths.groupby(['Country_Region','Date'])[['value']].sum().reset_index()

country_wise_total_recovered = country_wise_total_recovered.groupby(['Country_Region','Date'])[['value']].sum().reset_index()
total_countries = country_wise_total_confirmed.merge(country_wise_total_deaths, on = ['Country_Region','Date'])

total_countries = total_countries.merge(country_wise_total_recovered, on = ['Country_Region','Date'])
total_countries
country_wise_total_confirmed
total_countries.rename(columns = {

    'value_x': 'Confirmed_Cases',

    'value_y': 'Deaths',

    'value' : 'Recovered'}, inplace = True)



#total_countries.drop(['variable_x','variable_y','variable'], axis = 1, inplace = True)
fig = px.choropleth(total_countries,

                   locations = 'Country_Region',

                   locationmode='country names',

                   color = 'Confirmed_Cases',

                   hover_name = 'Country_Region',

                   color_continuous_scale="Viridis",

                   scope = 'world',

                   hover_data = ['Confirmed_Cases','Deaths','Recovered'],

                   title='Covid19 cases worldwide')



fig.update_geos(fitbounds="locations", visible=True)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



fig.show()
fig = px.choropleth(total_countries,

                   locations = 'Country_Region',

                   locationmode='country names',

                   color = 'Confirmed_Cases',

                   hover_name = 'Country_Region',

                   color_continuous_scale="Viridis",

                   scope = 'north america',

                   hover_data = ['Confirmed_Cases','Deaths','Recovered'],

                   title='Covid19 cases worldwide')

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()

country_names = []

for i in (total_countries["Country_Region"]):

    country_names.append(i)
country_names.remove('Holy See')

country_names.remove('Kosovo')

country_names.remove('MS Zaandam')

country_names.remove('Timor-Leste')

country_names.remove('US')

country_names.remove('West Bank and Gaza')

country_names.remove("Western Sahara")
country_names = [sub.replace('Burma', 'Myanmar') for sub in country_names] 

country_names = [sub.replace('Congo (Brazzaville)', 'Congo') for sub in country_names] 

country_names = [sub.replace('Congo (Kinshasa)', 'Democratic Republic of the Congo') for sub in country_names] 

country_names = [sub.replace('Cote d\'Ivoire', 'Côte d\'Ivoire') for sub in country_names] 

country_names = [sub.replace('Korea, South', 'South Korea') for sub in country_names] 

country_names = [sub.replace('Taiwan*', 'Taiwan') for sub in country_names]
!pip install pycountry

!pip install pycountry-convert
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2



continents = {

    'NA': 'North America',

    'SA': 'South America', 

    'AS': 'Asia',

    'OC': 'Australia',

    'AF': 'Africa',

    'EU': 'Europe'

}

y = [continents[country_alpha2_to_continent_code(country_name_to_country_alpha2(country))] for country in country_names]

#continent and countries

continents_country = pd.DataFrame(list(zip(country_names, y)), 

               columns =['Country_Region', 'Continent'])
continents_country['Country_Region'] = continents_country['Country_Region'].replace({'Congo':'Congo (Brazzaville)',

                                                                                  'Democratic Republic of the Congo':'Congo (Kinshasa)',

                                                                                  'Côte d\'Ivoire': 'Cote d\'Ivoire',

                                                                                  'South Korea': 'Korea, South',

                                                                                  'Myanmar': 'Burma',

                                                                                  'Taiwan': 'Taiwan*'})
total_countries_and_cont = total_countries.merge(continents_country, on = ['Country_Region'])
african_countries  = total_countries_and_cont[total_countries_and_cont["Continent"].str.match('Africa')].reset_index()

asian_countries  = total_countries_and_cont[total_countries_and_cont["Continent"].str.match('Asia')].reset_index()

north_american_countries  = total_countries_and_cont[total_countries_and_cont["Continent"].str.match('North America')].reset_index()

european_countries  = total_countries_and_cont[total_countries_and_cont["Continent"].str.match('Europe')].reset_index()

australian_countries  = total_countries_and_cont[total_countries_and_cont["Continent"].str.match('Australia')].reset_index()

south_american_countries  = total_countries_and_cont[total_countries_and_cont["Continent"].str.match('South America')].reset_index()





fig = px.choropleth(african_countries,

                   locations = 'Country_Region',

                   locationmode='country names',

                   color = 'Confirmed_Cases',

                   hover_name = 'Country_Region',

                   color_continuous_scale="Viridis",

                   scope = 'africa',

                   hover_data = ['Confirmed_Cases','Deaths','Recovered'],

                   title='Covid19 cases worldwide')

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth(asian_countries,

                   locations = 'Country_Region',

                   locationmode='country names',

                   color = 'Confirmed_Cases',

                   hover_name = 'Country_Region',

                   color_continuous_scale="Viridis",

                   scope = 'asia',

                   hover_data = ['Confirmed_Cases','Deaths','Recovered'],

                   title='Covid19 cases worldwide')

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth(south_american_countries,

                   locations = 'Country_Region',

                   locationmode='country names',

                   color = 'Confirmed_Cases',

                   hover_name = 'Country_Region',

                   color_continuous_scale="Viridis",

                   scope = 'south america',

                   hover_data = ['Confirmed_Cases','Deaths','Recovered'],

                   title='Covid19 cases worldwide')

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth(australian_countries,

                   locations = 'Country_Region',

                   locationmode='country names',

                   color = 'Confirmed_Cases',

                   hover_name = 'Country_Region',

                   color_continuous_scale="Viridis",

                   hover_data = ['Confirmed_Cases','Deaths','Recovered'],

                   title='Covid19 cases worldwide')

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth(european_countries,

                   locations = 'Country_Region',

                   locationmode='country names',

                   color = 'Confirmed_Cases',

                   hover_name = 'Country_Region',

                   color_continuous_scale="Viridis",

                   scope = 'europe',

                   hover_data = ['Confirmed_Cases','Deaths','Recovered'],

                   title='Covid19 cases worldwide')

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
confirmed_us.drop(['UID','iso2','iso3','Admin2','Country_Region','code3','Combined_Key'],axis = 1, inplace = True)
deaths_us.drop(['UID','iso2','iso3','Admin2','Country_Region','code3','Combined_Key','Population'], axis = 1, inplace = True)
# declaring function for converting Date formats

def convert_date_us(data):

    try:

        data.columns = list(data.columns[:4]) + [datetime.strptime(dt, "%m/%d/%y").date().strftime("%Y-%m-%d") for dt in data.columns[4:]]

    except:

        data.columns = list(data.columns[:4]) + [datetime.strptime(dt, "%m/%d/%Y").date().strftime("%Y-%m-%d") for dt in data.columns[4:]]
convert_date_us(deaths_us)

convert_date_us(confirmed_us)
confirmed_us
from urllib.request import urlopen

import json

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)
confirmed_us_df = confirmed_us.melt(id_vars = ['Province_State','FIPS','Lat','Long_'],

                                            value_vars = confirmed_us.columns[4:],

                                            var_name = 'Date',

                                            value_name = 'Confirmed_Cases')
deaths_us_df = deaths_us.melt(id_vars = ['Province_State','FIPS','Lat','Long_'],

                                            value_vars = deaths_us.columns[4:],

                                            var_name = 'Date',

                                            value_name = 'Confirmed_Cases')
confirmed_us_df = confirmed_us_df.groupby(['Province_State','FIPS','Date'])['Confirmed_Cases'].sum().reset_index()
confirmed_us_df.info()
confirmed_us_df['FIPS'] = confirmed_us_df['FIPS'].astype(int)
x = confirmed_us_df.groupby(['Province_State','FIPS'])['Confirmed_Cases'].sum().reset_index()

x
# due to missing data (FIPS) from few counties the map won't highlight all the regions info 

fig = px.choropleth(x,

                    geojson=counties, 

                    locations='FIPS', color='Confirmed_Cases',

                    color_continuous_scale="Viridis",

                    range_color=(confirmed_us_df['Confirmed_Cases'].min(), confirmed_us_df['Confirmed_Cases'].max()),

                    hover_name = 'Province_State',

                    scope = 'usa')



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
india = confirmed_global[confirmed_global['Country_Region'].str.match('India')]

india
india_1 = india.melt(id_vars = ['Country_Region'],

                                            value_vars = confirmed_global.columns[4:],

                                            var_name = 'Date',

                                            value_name = 'Confirmed_Cases')

india_1
india_2 = india_1.iloc[:, 2:3].values

india_2
last = len(india_2)

train_selection_value = int(last/100 * 70)

test_selection_value = last - train_selection_value
train_selection_value
test_selection_value
india_train_df = india_1[:train_selection_value]

india_test_df = india_1[train_selection_value-1:]
india_train_df
india_train = india_1.iloc[:, 2:3].values

india_train = india_train[:train_selection_value]



india_test = india_1.iloc[:, 2:3].values

india_test = india_test[train_selection_value-1:]
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

india_train_scaled = sc.fit_transform(india_train)

india_test_scaled = sc.fit_transform(india_test)



X_train = []

y_train = []

for i in range(25, train_selection_value-1):

    X_train.append(india_train_scaled[i-25: i, 0])

    y_train.append(india_train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)



X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train
# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
# Initialising the RNN

regressor = Sequential()



# Adding the first LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))



# Adding a second LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a third LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a fourth LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))



# Adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

regressor.fit(X_train, y_train, epochs = 100, batch_size = 5)

real = india_test_df.iloc[:, 2:3].values
dataset_total = pd.concat((india_train_df['Confirmed_Cases'], india_test_df['Confirmed_Cases']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(india_test_df) - 25:].values

inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)
X_test = []



for i in range(25, len(inputs)):

    X_test.append(inputs[i-25:i, 0])

    

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted = regressor.predict(X_test)

predicted = sc.inverse_transform(predicted)
# Visualising the results

plt.figure(figsize = (25,8))

plt.plot(real, color = 'red', label = 'Real')

plt.plot(predicted, color = 'blue', label = 'Predicted')

plt.title('Prediction')

plt.xlabel('Time')

plt.ylabel('confirmed')

plt.legend()

plt.show()