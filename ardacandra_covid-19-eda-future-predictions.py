import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go



import datetime as dt

import pycountry



from fbprophet import Prophet
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df.info()

df.head()
df = df.set_index('SNo')
columns_with_na = [col for col in df.columns if df[col].isnull().any()]

columns_with_na
df.loc[df['Province/State'].isna()]
df['Province/State'] = df['Province/State'].fillna('None')

df['Province/State'].value_counts()
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])

df['Last Update'] = pd.to_datetime(df['Last Update'])
sns.set(rc={'axes.facecolor':'black','figure.facecolor':'black','axes.grid':'True','grid.color':'dimgrey','axes.labelcolor':'white','text.color':'white', 'xtick.color':'white', 'ytick.color':'white'})



confirmed_per_day = df.groupby('ObservationDate')['Confirmed'].sum()

plot = sns.lineplot(x=confirmed_per_day.index, y=confirmed_per_day.values, color='orange')

plot.set_title('Number of People Infected')

for item in plot.get_xticklabels():

    item.set_rotation(30)
deaths_per_day = df.groupby('ObservationDate')['Deaths'].sum()

plot = sns.lineplot(x=deaths_per_day.index, y=deaths_per_day.values, color='red')

plot.set_title('Number of Deaths')

for item in plot.get_xticklabels():

    item.set_rotation(30)
df['ObservationDate'].max()
latest_distribution_by_country = df.loc[df['ObservationDate'] == df['ObservationDate'].max()].groupby('Country/Region')['Confirmed'].sum()



fig = px.pie(values=latest_distribution_by_country.values, names=latest_distribution_by_country.index, title='Latest Distribution of Covid-19', hole=.3)

fig.update_traces(textposition='inside')

fig.show()
latest_distribution_in_china = df.loc[(df['ObservationDate'] == df['ObservationDate'].max()) & (df['Country/Region'] == 'Mainland China')].groupby('Province/State')['Confirmed'].sum()

fig = px.pie(values=latest_distribution_in_china.values, names=latest_distribution_in_china.index, title='Latest Distribution in China', hole=.3)

fig.update_traces(textposition='inside')

fig.show()
latest_distribution_in_US = df.loc[(df['ObservationDate'] == df['ObservationDate'].max()) & (df['Country/Region'] == 'US')].groupby('Province/State')['Confirmed'].sum()

fig = px.pie(values=latest_distribution_in_US.values, names=latest_distribution_in_US.index, title='Latest Distribution in US',hole=.3)

fig.update_traces(textposition='inside')

fig.show()
latest_deaths_by_country = df.loc[df['ObservationDate'] == df['ObservationDate'].max()].groupby('Country/Region')['Deaths'].sum()



fig = px.pie(values=latest_deaths_by_country.values, names=latest_deaths_by_country.index, title='Latest Distribution of Deaths', hole=.3)

fig.update_traces(textposition='inside')

fig.show()
confirmed_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

deaths_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

recovered_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
confirmed_df.head()
deaths_df.head()
recovered_df.head()
latest_date = confirmed_df.columns[-1]
countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3

    

def get_iso(country):

    if country == 'Brunei':

        return countries['Brunei Darussalam']

    elif country == 'US':

        return countries['United States']

    elif country == 'Holy See':

        return countries['Holy See (Vatican City State)']

    elif country == 'Iran':

        return countries['Iran, Islamic Republic of']

    elif country == 'Korea, South':

        return countries["Korea, Democratic People's Republic of"]

    elif country == 'Cruise Ship':

        return None

    elif country == 'Taiwan*':

        return countries['Taiwan, Province of China']

    elif country == 'Vietnam':

        return countries['Viet Nam']

    elif country == 'Russia':

        return countries['Russian Federation']

    elif country == 'Moldova':

        return countries['Moldova, Republic of']

    elif country == 'Bolivia':

        return countries['Bolivia, Plurinational State of']

    elif country == 'Congo (Kinshasa)' or country == 'Congo (Brazzaville)':

        return countries['Congo']

    elif country == "Cote d'Ivoire":

        return countries["Côte d'Ivoire"]

    elif country == 'Reunion':

        return countries['Réunion']

    elif country == 'Venezuela':

        return countries['Venezuela, Bolivarian Republic of']

    elif country == 'Curacao':

        return countries['Curaçao']

    elif country == 'occupied Palestinian territory':

        return countries['Palestine, State of']

    elif country == 'Kosovo':

        return 'UNK'

    elif country == 'Tanzania':

        return countries['Tanzania, United Republic of']

    elif country == 'Gambia, The':

        return countries['Gambia']

    elif country == 'Bahamas, The':

        return countries['Bahamas']

    return countries[country]

    

confirmed_df['Country_iso'] = confirmed_df['Country/Region'].apply(get_iso)

confirmed_df.head()
plot_df = pd.DataFrame(confirmed_df[['Country_iso', 'Country/Region', latest_date]].groupby(['Country_iso', 'Country/Region'])[latest_date].sum())

plot_df = plot_df.reset_index()

plot_df.head()
fig = px.choropleth(plot_df, locations="Country_iso",

                    color=latest_date,

                    hover_name="Country/Region", 

                    color_continuous_scale=px.colors.sequential.YlOrRd,

                    range_color=[0, 5000])

fig.update_layout(

    title_text = 'Distribution of Latest Total Confirmed Cases',

)



fig.show()
deaths_df['Country_iso'] = deaths_df['Country/Region'].apply(get_iso)

deaths_df.head()



plot_df = pd.DataFrame(deaths_df[['Country_iso', 'Country/Region', latest_date]].groupby(['Country_iso', 'Country/Region'])[latest_date].sum())

plot_df = plot_df.reset_index()



fig = px.choropleth(plot_df, locations="Country_iso",

                    color=latest_date,

                    hover_name="Country/Region", 

                    color_continuous_scale=px.colors.sequential.OrRd,

                    range_color=[0, 300])

fig.update_layout(

    title_text = 'Distribution of Latest Total Deaths',

)



fig.show()
date_columns = confirmed_df.drop(['Province/State', 'Country/Region', 'Lat', 'Long', 'Country_iso'], axis=1).columns

animation_df = pd.DataFrame(confirmed_df.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).groupby(['Country_iso'])[date_columns].sum())

animation_df = animation_df.reset_index()



animation_df = animation_df.melt(id_vars=['Country_iso'], var_name='Date', value_name='Value')

animation_df.tail()
fig = px.choropleth(animation_df, locations='Country_iso',

                    color='Value', 

                    color_continuous_scale=px.colors.sequential.YlOrRd,

                    range_color=[0, 5000],

                    animation_frame='Date')

fig.update_layout(

    title_text = 'Animation of Confirmed Infections',

)



fig.show()
confirmed_df.head()
ts_df = confirmed_df.drop(['Province/State', 'Country/Region', 'Lat', 'Long', 'Country_iso'], axis=1).transpose()

ts_df['sum_infected'] = ts_df.sum(axis=1)

ts_df = ts_df.reset_index()[['index', 'sum_infected']]

ts_df['index'] = pd.to_datetime(ts_df['index'])

ts_df = ts_df.rename(columns={'index':'ds', 'sum_infected':'y'})

ts_df.head()
model = Prophet()

model.fit(ts_df)
future = model.make_future_dataframe(periods=90)

future.tail()
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
plt.style.use('default')

fig, ax = plt.subplots()

fig = model.plot(forecast, xlabel='Date', ylabel='Total Confirmed', ax=ax)

ax.set_title('Predictions for Number of Confirmed Cases in 3 Months Time (with Prophet)')

fig.show()
ts_df = deaths_df.drop(['Province/State', 'Country/Region', 'Lat', 'Long', 'Country_iso'], axis=1).transpose()

ts_df['sum_deaths'] = ts_df.sum(axis=1)

ts_df = ts_df.reset_index()[['index', 'sum_deaths']]

ts_df['index'] = pd.to_datetime(ts_df['index'])

ts_df = ts_df.rename(columns={'index':'ds', 'sum_deaths':'y'})

ts_df.head()
model = Prophet()

model.fit(ts_df)
future = model.make_future_dataframe(periods=90)

future.tail()
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
plt.style.use('default')

fig, ax = plt.subplots()

fig = model.plot(forecast, xlabel='Date', ylabel='Total Deaths', ax=ax)

ax.set_title('Predictions for Number of  Deaths in 3 Months Time (with Prophet)')

fig.show()