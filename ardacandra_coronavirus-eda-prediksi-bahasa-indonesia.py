#untuk olah data

import pandas as pd

import numpy as np

import datetime as dt

import pycountry

import re



#untuk plotting

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import folium

from folium import Marker

from folium.plugins import MarkerCluster



#untuk membuat model time-series

from fbprophet import Prophet

from sklearn.metrics import mean_squared_error
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
#memilih style supaya plot terlihat lebih menarik

sns.set(rc={'axes.facecolor':'black','figure.facecolor':'black','axes.grid':'True','grid.color':'dimgrey','axes.labelcolor':'white','text.color':'white', 'xtick.color':'white', 'ytick.color':'white'})
fig, ax = plt.subplots(figsize=(8, 6))



confirmed_per_day = df.groupby('ObservationDate')['Confirmed'].sum()

sns.lineplot(x=confirmed_per_day.index, y=confirmed_per_day.values, color='orange', label='Infeksi', ax=ax)



deaths_per_day = df.groupby('ObservationDate')['Deaths'].sum()

sns.lineplot(x=deaths_per_day.index, y=deaths_per_day.values, color='red', label='Kematian', ax=ax)



recovered_per_day = df.groupby('ObservationDate')['Recovered'].sum()

sns.lineplot(x=recovered_per_day.index, y=recovered_per_day.values, color='blue', label='Penyembuhan', ax=ax)



for item in ax.get_xticklabels():

    item.set_rotation(30)

ax.set_title('Jumlah Total Infeksi, Kematian, dan Penyembuhan Secara Global')

fig.show()
latest_distribution_by_country = df.loc[df['ObservationDate'] == df['ObservationDate'].max()].groupby('Country/Region')['Confirmed'].sum()



fig = px.pie(values=latest_distribution_by_country.values, names=latest_distribution_by_country.index, title='Distribusi Terbaru Jumlah Kasus Infeksi Covid-19', hole=.3)

fig.update_traces(textposition='inside')

fig.show()
latest_distribution_in_china = df.loc[(df['ObservationDate'] == df['ObservationDate'].max()) & (df['Country/Region'] == 'Mainland China')].groupby('Province/State')['Confirmed'].sum()

fig = px.pie(values=latest_distribution_in_china.values, names=latest_distribution_in_china.index, title='Distribusi Terbaru Total Infeksi di Cina', hole=.3)

fig.update_traces(textposition='inside')

fig.show()
latest_distribution_in_US = df.loc[(df['ObservationDate'] == df['ObservationDate'].max()) & (df['Country/Region'] == 'US')].groupby('Province/State')['Confirmed'].sum()

fig = px.pie(values=latest_distribution_in_US.values, names=latest_distribution_in_US.index, title='Distribusi Terbaru Total Infeksi di Amerika',hole=.3)

fig.update_traces(textposition='inside')

fig.show()
latest_deaths_by_country = df.loc[df['ObservationDate'] == df['ObservationDate'].max()].groupby('Country/Region')['Deaths'].sum()



fig = px.pie(values=latest_deaths_by_country.values, names=latest_deaths_by_country.index, title='Distribusi Terbaru Kematian Akibat 2019 Novel Coronavirus', hole=.3)

fig.update_traces(textposition='inside')

fig.show()
fig, ax = plt.subplots(figsize=(8, 6))



confirmed_china = df.loc[df['Country/Region']=='Mainland China'].groupby('ObservationDate')['Confirmed'].sum()

sns.lineplot(x=confirmed_china.index, y=confirmed_china.values, color='red', label='Cina', ax=ax)



confirmed_italy = df.loc[df['Country/Region']=='Italy'].groupby('ObservationDate')['Confirmed'].sum()

sns.lineplot(x=confirmed_italy.index, y=confirmed_italy.values, color='blue', label='Itali', ax=ax)



confirmed_US = df.loc[df['Country/Region']=='US'].groupby('ObservationDate')['Confirmed'].sum()

sns.lineplot(x=confirmed_US.index, y=confirmed_US.values, color='white', label='Amerika', ax=ax)



confirmed_spain = df.loc[df['Country/Region']=='Spain'].groupby('ObservationDate')['Confirmed'].sum()

sns.lineplot(x=confirmed_spain.index, y=confirmed_spain.values, color='orange', label='Spanyol', ax=ax)



confirmed_germany = df.loc[df['Country/Region']=='Germany'].groupby('ObservationDate')['Confirmed'].sum()

sns.lineplot(x=confirmed_germany.index, y=confirmed_germany.values, color='yellow', label='Jerman', ax=ax)



for item in ax.get_xticklabels():

    item.set_rotation(30)

ax.set_title('Perkembangan Jumlah Kasus di 5 Negara dengan Kasus Terbanyak')

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

    elif country == 'Congo (Kinshasa)':

        return countries['Congo']

    elif country == 'Congo (Brazzaville)':

        return countries['Congo']

    elif country == 'Republic of the Congo':

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

    elif country == 'The Gambia':

        return countries['Gambia']

    elif country == 'The Bahamas':

        return countries['Bahamas']

    elif country == 'Bahamas, The':

        return countries['Bahamas']

    elif country == 'Cape Verde':

        return 'CPV'

    elif country == 'East Timor':

        return 'Timor Leste'

    elif country == 'Syria':

        return 'Syrian Arab Republic'

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

                    range_color=[0, 20000])

fig.update_layout(

    title_text = 'Distribusi Terbaru Kasus-Kasus yang Terkonfirmasi',

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

    title_text = 'Distribusi Terbaru Kematian karena 2019 Novel Coronavirus',

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

                    range_color=[0, 10000],

                    animation_frame='Date')

fig.update_layout(

    title_text = 'Animasi Perkembangan Jumlah Kasus',

)



fig.show()
cases_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

cases_df = cases_df[cases_df['age']!='Belgium']

cases_df.head()
cases_df.columns
cases_df = cases_df[['ID', 'age', 'sex', 'city', 'province', 'country', 'wuhan(0)_not_wuhan(1)', 'latitude', 'longitude', 'date_confirmation', 'symptoms', 'outcome']]

cases_df.head()
age_and_sex_df = cases_df[['age', 'sex']].dropna()

age_and_sex_df['sex'] = age_and_sex_df['sex'].str.lower()



for idx, age in enumerate(age_and_sex_df['age']):

    if '-' in age:

        x = re.split('-', age)

        age_and_sex_df['age'].iloc[idx] = (int(x[0]) + int(x[1]))/2 

age_and_sex_df['age'] = age_and_sex_df['age'].astype(float)

        

age_and_sex_df.tail()
plt.style.use('default')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 6))

fig.tight_layout()



index = age_and_sex_df['sex'].value_counts().index

values = age_and_sex_df['sex'].value_counts().values

sns.barplot(x=index, y=values, ax=ax1)

ax1.set_title('Distribusi Jenis Kelamin')



age_df = age_and_sex_df[['sex', 'age']].dropna()

sns.distplot(age_df.loc[age_df['sex']=='male']['age'], hist=True, ax=ax2)

ax2.set_title('Distribusi Umur pada Pria')



sns.distplot(age_df.loc[age_df['sex']=='female']['age'], hist=True, ax=ax3)

ax3.set_title('Distribusi Umur pada Wanita')

    

plt.show()
m_1 = folium.Map(location=[0, 0], tiles='openstreetmap', zoom_start=2)



for idx, row in cases_df[~pd.isna(cases_df['outcome'])].iterrows():

    if pd.notnull(row['latitude']):

        Marker([row['latitude'], row['longitude']], popup=folium.Popup((

                                                            'Province : {province}<br>'

                                                            'City : {city}<br>'

                                                            'Date Confirmation : {date_confirmation}<br>'

                                                            'Symptoms :{symptoms}<br>'

                                                            'Outcome : {outcome}').format(

                                                            province=row['province'],

                                                            city=row['city'],

                                                            date_confirmation=row['date_confirmation'],

                                                            symptoms=row['symptoms'],

                                                            outcome=row['outcome']), max_width=450)

              ).add_to(m_1)

    

m_1
symptoms_df = pd.DataFrame(cases_df['symptoms'].dropna().reset_index(drop=True))

symptoms_df['fever'] = 0

symptoms_df['cough'] = 0

symptoms_df['headache'] = 0



for idx, s in enumerate(symptoms_df['symptoms']):

    if 'fever' in s:

        symptoms_df['fever'].iloc[idx] = 1

    if 'cough' in s:

        symptoms_df['cough'].iloc[idx] = 1

    if 'headache' in s:

        symptoms_df['headache'].iloc[idx] = 1

        

symptoms_df.head()
plt.style.use('default')

fig, ax = plt.subplots(figsize=(8, 6))

fig.tight_layout()



index = ['Demam', 'Batuk', 'Pusing']

values = [symptoms_df['fever'].sum(), symptoms_df['cough'].sum(), symptoms_df['headache'].sum()]

sns.barplot(x=index, y=values, ax=ax)

ax.set_title('Jumlah Symptom yang Terlihat pada Kasus-Kasus di Dataset')
confirmed_df.head()
ts_df = confirmed_df.drop(['Province/State', 'Country/Region', 'Lat', 'Long', 'Country_iso'], axis=1).transpose()

ts_df['sum_infected'] = ts_df.sum(axis=1)

ts_df = ts_df.reset_index()[['index', 'sum_infected']]

ts_df['index'] = pd.to_datetime(ts_df['index'])

ts_df = ts_df.rename(columns={'index':'ds', 'sum_infected':'y'})

ts_df.head()
ts_df['y'] = ts_df['y'].diff()

ts_df.head()
X_train = ts_df['ds'][:int(ts_df.shape[0]*0.9)]

X_test = ts_df['ds'][int(ts_df.shape[0]*0.9):]

y_train = ts_df['y'][:int(ts_df.shape[0]*0.9)]

y_test = ts_df['y'][int(ts_df.shape[0]*0.9):]
model = Prophet()

model.fit(pd.concat([X_train, y_train], axis=1, sort=False))
future = model.make_future_dataframe(periods=90)

future.tail()
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
plt.style.use('default')

fig, ax = plt.subplots()



fig = model.plot(forecast,ax=ax)



sns.scatterplot(x=X_test, y=y_test, ax=ax, color='red')



ax.set_title('Prediksi Jumlah Kasus yang Dikonfirmasi per Hari')

plt.xlabel('Tanggal')

plt.ylabel('Jumlah yang Dikonfirmasi')



for item in ax.get_xticklabels():

    item.set_rotation(30)

fig.show()
expected = y_test

predictions = model.predict(pd.DataFrame(X_test).reset_index())



mse = mean_squared_error(expected, predictions['trend'])

print('MSE : ' + str(mse))
fig, ax = plt.subplots()



sns.scatterplot(data=ts_df, x='ds', y=y_train.cumsum(), ax=ax, color='black')

sns.lineplot(data=forecast, x='ds', y=forecast['trend'].cumsum(), ax=ax, color='blue')



ax.set_title('Prediksi Jumlah Kumulatif Kasus Terkonfirmasi')

plt.xlabel('Tanggal')

plt.ylabel('Total Kasus Positif')

plt.xlim(dt.datetime(2020, 1, 20), dt.datetime(2020, 6, 6))



for item in ax.get_xticklabels():

    item.set_rotation(30)

fig.show()
ts_df = deaths_df.drop(['Province/State', 'Country/Region', 'Lat', 'Long', 'Country_iso'], axis=1).transpose()

ts_df['sum_deaths'] = ts_df.sum(axis=1)

ts_df = ts_df.reset_index()[['index', 'sum_deaths']]

ts_df['index'] = pd.to_datetime(ts_df['index'])

ts_df = ts_df.rename(columns={'index':'ds', 'sum_deaths':'y'})

ts_df['y'] = ts_df['y'].diff()

ts_df.head()
X_train = ts_df['ds'][:int(ts_df.shape[0]*0.9)]

X_test = ts_df['ds'][int(ts_df.shape[0]*0.9):]

y_train = ts_df['y'][:int(ts_df.shape[0]*0.9)]

y_test = ts_df['y'][int(ts_df.shape[0]*0.9):]
model = Prophet()

model.fit(pd.concat([X_train, y_train], axis=1, sort=False))
future = model.make_future_dataframe(periods=90)

future.tail()
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
plt.style.use('default')

fig, ax = plt.subplots()



fig = model.plot(forecast,ax=ax)



sns.scatterplot(x=X_test, y=y_test, ax=ax, color='red')



ax.set_title('Jumlah Kematian per Hari')

plt.xlabel('Tanggal')

plt.ylabel('Jumlah Kematian')



for item in ax.get_xticklabels():

    item.set_rotation(30)

fig.show()
expected = y_test

predictions = model.predict(pd.DataFrame(X_test).reset_index())



mse = mean_squared_error(expected, predictions['trend'])

print('MSE : ' + str(mse))
fig, ax = plt.subplots()



sns.scatterplot(data=ts_df, x='ds', y=y_train.cumsum(), ax=ax, color='black')

sns.lineplot(data=forecast, x='ds', y=forecast['trend'].cumsum(), ax=ax, color='blue')



ax.set_title('Prediksi Jumlah Kumulatif Kematian')

plt.xlabel('Tanggal')

plt.ylabel('Total Kematian')

plt.xlim(dt.datetime(2020, 1, 20), dt.datetime(2020, 6, 6))



for item in ax.get_xticklabels():

    item.set_rotation(30)

fig.show()