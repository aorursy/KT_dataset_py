%matplotlib inline

import pandas as pd

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 99)

pd.set_option('display.max_rows', 99)

import os

import numpy as np

from matplotlib import pyplot as plt

from tqdm import tqdm

import datetime as dt
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

import seaborn as sns

sns.set_palette(sns.color_palette('tab20', 20))



import plotly.express as px

import plotly.graph_objects as go
COMP = '../input/covid19-global-forecasting-week-4'

DATEFORMAT = '%Y-%m-%d'





def get_comp_data(COMP):

    train = pd.read_csv(f'{COMP}/train.csv')

    test = pd.read_csv(f'{COMP}/test.csv')

    submission = pd.read_csv(f'{COMP}/submission.csv')

    print(train.shape, test.shape, submission.shape)

    train['Country_Region'] = train['Country_Region'].str.replace(',', '')

    test['Country_Region'] = test['Country_Region'].str.replace(',', '')



    train['Location'] = train['Country_Region'] + '-' + train['Province_State'].fillna('')



    test['Location'] = test['Country_Region'] + '-' + test['Province_State'].fillna('')



    train['LogConfirmed'] = to_log(train.ConfirmedCases)

    train['LogFatalities'] = to_log(train.Fatalities)

    train = train.drop(columns=['Province_State'])

    test = test.drop(columns=['Province_State'])



    country_codes = pd.read_csv('../input/covid19-metadata/country_codes.csv', keep_default_na=False)

    train = train.merge(country_codes, on='Country_Region', how='left')

    test = test.merge(country_codes, on='Country_Region', how='left')



    train['DateTime'] = pd.to_datetime(train['Date'])

    test['DateTime'] = pd.to_datetime(test['Date'])

    

    return train, test, submission





def process_each_location(df):

    dfs = []

    for loc, df in tqdm(df.groupby('Location')):

        df = df.sort_values(by='Date')

        df['Fatalities'] = df['Fatalities'].cummax()

        df['ConfirmedCases'] = df['ConfirmedCases'].cummax()

        df['LogFatalities'] = df['LogFatalities'].cummax()

        df['LogConfirmed'] = df['LogConfirmed'].cummax()

        df['LogConfirmedNextDay'] = df['LogConfirmed'].shift(-1)

        df['ConfirmedNextDay'] = df['ConfirmedCases'].shift(-1)

        df['DateNextDay'] = df['Date'].shift(-1)

        df['LogFatalitiesNextDay'] = df['LogFatalities'].shift(-1)

        df['FatalitiesNextDay'] = df['Fatalities'].shift(-1)

        df['LogConfirmedDelta'] = df['LogConfirmedNextDay'] - df['LogConfirmed']

        df['ConfirmedDelta'] = df['ConfirmedNextDay'] - df['ConfirmedCases']

        df['LogFatalitiesDelta'] = df['LogFatalitiesNextDay'] - df['LogFatalities']

        df['FatalitiesDelta'] = df['FatalitiesNextDay'] - df['Fatalities']

        dfs.append(df)

    return pd.concat(dfs)





def add_days(d, k):

    return dt.datetime.strptime(d, DATEFORMAT) + dt.timedelta(days=k)





def to_log(x):

    return np.log(x + 1)





def to_exp(x):

    return np.exp(x) - 1

start = dt.datetime.now()

train, test, submission = get_comp_data(COMP)

train.shape, test.shape, submission.shape

train.head(2)

test.head(2)
train[train.geo_region.isna()].Country_Region.unique()

train = train.fillna('#N/A')

test = test.fillna('#N/A')



train[train.duplicated(['Date', 'Location'])]

train.count()
train.describe()

train.nunique()

train.dtypes

train.count()



TRAIN_START = train.Date.min()

TEST_START = test.Date.min()

TRAIN_END = train.Date.max()

TEST_END = test.Date.max()

TRAIN_START, TRAIN_END, TEST_START, TEST_END
train = train.sort_values(by='Date')

countries_latest_state = train[train['Date'] == TRAIN_END].groupby([

    'Country_Region', 'continent', 'geo_region', 'country_iso_code_3']).sum()[[

    'ConfirmedCases', 'Fatalities']].reset_index()

countries_latest_state['Log10Confirmed'] = np.log10(countries_latest_state.ConfirmedCases + 1)

countries_latest_state['Log10Fatalities'] = np.log10(countries_latest_state.Fatalities + 1)

countries_latest_state = countries_latest_state.sort_values(by='Fatalities', ascending=False)

countries_latest_state.to_csv('countries_latest_state.csv', index=False)



countries_latest_state.shape

countries_latest_state.head()
fig = go.Figure(data=go.Choropleth(

    locations = countries_latest_state['country_iso_code_3'],

    z = countries_latest_state['Log10Confirmed'],

    text = countries_latest_state['Country_Region'],

    colorscale = 'viridis_r',

    autocolorscale=False,

    reversescale=False,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_tickprefix = '10^',

    colorbar_title = 'Confirmed cases <br>(log10 scale)',

))



_ = fig.update_layout(

    title_text=f'COVID-19 Global Cases [Updated: {TRAIN_END}]',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    )

)



fig.show()
fig = go.Figure(data=go.Choropleth(

    locations = countries_latest_state['country_iso_code_3'],

    z = countries_latest_state['Log10Fatalities'],

    text = countries_latest_state['Country_Region'],

    colorscale = 'viridis_r',

    autocolorscale=False,

    reversescale=False,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_tickprefix = '10^',

    colorbar_title = 'Deaths <br>(log10 scale)',

))



_ = fig.update_layout(

    title_text=f'COVID-19 Global Deaths [Updated: {TRAIN_END}]',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    )

)



fig.show()
countries_latest_state['DeathConfirmedRatio'] = (countries_latest_state.Fatalities + 1) / (countries_latest_state.ConfirmedCases + 1)

countries_latest_state['DeathConfirmedRatio'] = countries_latest_state['DeathConfirmedRatio'].clip(0, 0.15) 

fig = px.scatter(countries_latest_state,

                 x='ConfirmedCases',

                 y='Fatalities',

                 color='DeathConfirmedRatio',

                 size='Log10Fatalities',

                 size_max=20,

                 hover_name='Country_Region',

                 color_continuous_scale='viridis_r'

)

_ = fig.update_layout(

    title_text=f'COVID-19 Deaths vs Confirmed Cases by Country [Updated: {TRAIN_END}]',

    xaxis_type="log",

    yaxis_type="log",

    width = 1600,

    height = 900,

)

fig.show()
# The source dataset is not necessary cumulative we will force it

latest_loc = train[train['Date'] == TRAIN_END][['Location', 'ConfirmedCases', 'Fatalities']]

max_loc = train.groupby(['Location'])[['ConfirmedCases', 'Fatalities']].max().reset_index()

check = pd.merge(latest_loc, max_loc, on='Location')

np.mean(check.ConfirmedCases_x == check.ConfirmedCases_y)

np.mean(check.Fatalities_x == check.Fatalities_y)

check[check.Fatalities_x != check.Fatalities_y]

check[check.ConfirmedCases_x != check.ConfirmedCases_y]
train_clean = process_each_location(train)



train_clean.shape

train_clean.tail()
regional_progress = train_clean.groupby(['DateTime', 'continent']).sum()[['ConfirmedCases', 'Fatalities']].reset_index()

regional_progress['Log10Confirmed'] = np.log10(regional_progress.ConfirmedCases + 1)

regional_progress['Log10Fatalities'] = np.log10(regional_progress.Fatalities + 1)

regional_progress = regional_progress[regional_progress.continent != '#N/A']

regional_progress = regional_progress.sort_values(by=['continent', 'DateTime'])



regional_progress['ConfirmedCasesDiff'] = regional_progress.groupby('continent').ConfirmedCases.diff().rolling(3).mean()

regional_progress['FatalitiesDiff'] = regional_progress.groupby('continent').Fatalities.diff().rolling(3).mean()
fig = px.area(regional_progress, x="DateTime", y="ConfirmedCases", color="continent")

_ = fig.update_layout(

    title_text=f'COVID-19 Cumulative Confirmed Cases by Continent [Updated: {TRAIN_END}]',

    width=1600,

    height=900

)

fig.show()

fig2 = px.line(regional_progress, x='DateTime', y='ConfirmedCases', color='continent')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases by Continent [Updated: {TRAIN_END}]'

)

fig2.show()



fig3 = px.line(regional_progress, x='DateTime', y='ConfirmedCasesDiff', color='continent')

_ = fig3.update_layout(

    title_text=f'COVID-19 Daily New Confirmed Cases by Continent [Updated: {TRAIN_END}]'

)

fig3.show()



fig = px.area(regional_progress, x="DateTime", y="Fatalities", color="continent")

_ = fig.update_layout(

    title_text=f'COVID-19 Cumulative Confirmed Deaths by Continent [Updated: {TRAIN_END}]'

)

fig.show()

fig2 = px.line(regional_progress, x='DateTime', y='Fatalities', color='continent')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Deaths by Continent [Updated: {TRAIN_END}]'

)

fig2.show()

fig3 = px.line(regional_progress, x='DateTime', y='FatalitiesDiff', color='continent')

_ = fig3.update_layout(

    title_text=f'COVID-19 Daily New Fatalities by Continent [Updated: {TRAIN_END}]'

)

fig3.show()
china = train_clean[train_clean.Location.str.startswith('China')]

top10_locations = china.groupby('Location')[['ConfirmedCases']].max().sort_values(

    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]

fig2 = px.line(china[china.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases in China [Updated: {TRAIN_END}]'

)

fig2.show()
europe = train_clean[train_clean.continent == 'Europe']

top10_locations = europe.groupby('Location')[['ConfirmedCases']].max().sort_values(

    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]

fig2 = px.line(europe[europe.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases in Europe [Updated: {TRAIN_END}]'

)

fig2.show()
us = train_clean[train_clean.Country_Region == 'US']

top10_locations = us.groupby('Location')[['ConfirmedCases']].max().sort_values(

    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]

fig2 = px.line(us[us.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases in the USA [Updated: {TRAIN_END}]'

)

fig2.show()
africa = train_clean[train_clean.continent == 'Africa']

top10_locations = africa.groupby('Location')[['ConfirmedCases']].max().sort_values(

    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]

fig2 = px.line(africa[africa.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases in Africa [Updated: {TRAIN_END}]'

)

fig2.show()
country_progress = train_clean.groupby(['Date', 'DateTime', 'Country_Region']).sum()[[

    'ConfirmedCases', 'Fatalities', 'ConfirmedDelta', 'FatalitiesDelta']].reset_index()

top10_countries = country_progress.groupby('Country_Region')[['Fatalities']].max().sort_values(

    by='Fatalities', ascending=False).reset_index().Country_Region.values[:10]



fig2 = px.line(country_progress[country_progress.Country_Region.isin(top10_countries)],

               x='DateTime', y='ConfirmedCases', color='Country_Region')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases by Country [Updated: {TRAIN_END}]'

)

fig2.show()

fig3 = px.line(country_progress[country_progress.Country_Region.isin(top10_countries)],

               x='DateTime', y='Fatalities', color='Country_Region')

_ = fig3.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Deaths by Country [Updated: {TRAIN_END}]'

)

fig3.show()
countries_0301 = country_progress[country_progress.Date == '2020-03-01'][[

    'Country_Region', 'ConfirmedCases', 'Fatalities']]

countries_0331 = country_progress[country_progress.Date == '2020-03-31'][[

    'Country_Region', 'ConfirmedCases', 'Fatalities']]

countries_in_march = pd.merge(countries_0301, countries_0331, on='Country_Region', suffixes=['_0301', '_0331'])

countries_in_march['IncreaseInMarch'] = countries_in_march.ConfirmedCases_0331 / (countries_in_march.ConfirmedCases_0301 + 1)

countries_in_march = countries_in_march[countries_in_march.ConfirmedCases_0331 > 200].sort_values(

    by='IncreaseInMarch', ascending=False)

countries_in_march.tail(15)
selected_countries = [

    'Italy', 'Vietnam', 'Bahrain', 'Singapore', 'Taiwan*', 'Japan', 'Kuwait', 'Korea, South', 'China']

fig2 = px.line(country_progress[country_progress.Country_Region.isin(selected_countries)],

               x='DateTime', y='ConfirmedCases', color='Country_Region')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases by Country [Updated: {TRAIN_END}]'

)

fig2.show()

fig3 = px.line(country_progress[country_progress.Country_Region.isin(selected_countries)],

               x='DateTime', y='Fatalities', color='Country_Region')

_ = fig3.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Deaths by Country [Updated: {TRAIN_END}]'

)

fig3.show()
train_clean['Geo#Country#Contintent'] = train_clean.Location + '#' + train_clean.Country_Region + '#' + train_clean.continent

latest = train_clean[train_clean.Date == TRAIN_END][[

    'Geo#Country#Contintent', 'ConfirmedCases', 'Fatalities', 'LogConfirmed', 'LogFatalities']]

daily_confirmed_deltas = train_clean[train_clean.Date >= '2020-03-17'].pivot(

    'Geo#Country#Contintent', 'Date', 'LogConfirmedDelta').round(3).reset_index()

daily_confirmed_deltas = latest.merge(daily_confirmed_deltas, on='Geo#Country#Contintent')

daily_confirmed_deltas.shape

daily_confirmed_deltas.head()

daily_confirmed_deltas.to_csv('daily_confirmed_deltas.csv', index=False)
deltas = train_clean[np.logical_and(

        train_clean.LogConfirmed > 2,

        ~train_clean.Location.str.startswith('China')

)].dropna().sort_values(by='LogConfirmedDelta', ascending=False)



deltas['start'] = deltas['LogConfirmed'].round(0)

confirmed_deltas = pd.concat([

    deltas.groupby('start')[['LogConfirmedDelta']].mean(),

    deltas.groupby('start')[['LogConfirmedDelta']].std(),

    deltas.groupby('start')[['LogConfirmedDelta']].count()

], axis=1)



deltas.mean()



confirmed_deltas.columns = ['avg', 'std', 'cnt']

confirmed_deltas

confirmed_deltas.to_csv('confirmed_deltas.csv')
fig = px.box(deltas,  x="start", y="LogConfirmedDelta", range_y=[0, 0.35])

fig.show()
fig = px.box(deltas[deltas.Date >= '2020-03-01'],  x="DateTime", y="LogConfirmedDelta", range_y=[0, 0.6])

fig.update_layout(

    width = 1600,

    height = 800,

)

fig.show()
deltas = train_clean[np.logical_and(

        train_clean.LogConfirmed > 0,

        ~train_clean.Location.str.startswith('China')

)].dropna().sort_values(by='LogConfirmedDelta', ascending=False)

deltas = deltas[deltas['Date'] >= '2020-03-12']



confirmed_deltas = pd.concat([

    deltas.groupby('Location')[['LogConfirmedDelta']].mean(),

    deltas.groupby('Location')[['LogConfirmedDelta']].std(),

    deltas.groupby('Location')[['LogConfirmedDelta']].count(),

    deltas.groupby('Location')[['LogConfirmed']].max()

], axis=1)

confirmed_deltas.columns = ['avg', 'std', 'cnt', 'max']



confirmed_deltas.sort_values(by='avg').head(10)

confirmed_deltas.sort_values(by='avg').tail(10)

confirmed_deltas.to_csv('confirmed_deltas.csv')
end = dt.datetime.now()

print('Finished', end, (end - start).seconds, 's')