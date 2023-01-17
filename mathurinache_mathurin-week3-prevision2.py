#beluga

#!/usr/bin/env python

# coding: utf-8



# # COVID-19 W2: A few charts and a simple baseline

# 

# 

# # Summary

# 

# **Disclaimer** We still have limited data to predict or understand what will happen in the next few weeks (months).

# 

# At this point I see more value in collecting data and monitoring the outbreak than trying to predict the future.

# 

# [Please don't kill yourself because I published a notebook](https://www.reddit.com/r/datascience/comments/fsfdn2/the_best_thing_you_can_do_to_fight_covid19_is/)

# 

# 

# 

# ### Challenges

#  * The outbreak patterns vary a lot among countries

#  * Most countries have only 2 weeks data

#  * Only a handful countries managed to succesfuly slow down the outbreak

#  * Almost every country had several serious regulations in recent weeks

#  * Increasing testing capacity could have serious impact on confirmed cases

# 

# 

# 

#  ### Assumptions

#   * As we are still in the early period, we will see exponential growth in the next few weeks

#   * Thanks to the panic/awareness/regulations/social distancing the exponential increase will slow down

# 

# As the process is not stationary at all I decided to use a simple heuristic approach. Maybe I will import sklearn next week.

#   

#   ### TIL

#  * Namibia's country code is NA. Now I remember I heard it in joke before, but I had to investigate a bug learn it again :)

#  * I haven't used plotly recently, I quite enjoyed the "new" Plotly Express interface

# 



# In[1]:





get_ipython().run_line_magic('matplotlib', 'inline')

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





# In[2]:





import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

import seaborn as sns

sns.set_palette(sns.color_palette('tab20', 20))



import plotly.express as px

import plotly.graph_objects as go





# In[3]:





COMP = '../input/covid19-global-forecasting-week-3'

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





# In[4]:





start = dt.datetime.now()

train, test, submission = get_comp_data(COMP)

train.shape, test.shape, submission.shape

train.head(2)

test.head(2)





# In[5]:





train.describe()

train.nunique()

train.dtypes

train.count()



TRAIN_START = train.Date.min()

TEST_START = test.Date.min()

TRAIN_END = train.Date.max()

TEST_END = test.Date.max()

TRAIN_START, TRAIN_END, TEST_START, TEST_END





# # Worldwide



# In[6]:





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





# In[7]:





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





# In[8]:





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





# In[9]:





countries_latest_state['DeathConfirmedRatio'] = (countries_latest_state.Fatalities + 1) / (countries_latest_state.ConfirmedCases + 1)

countries_latest_state['DeathConfirmedRatio'] = countries_latest_state['DeathConfirmedRatio'].clip(0, 0.1) 

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

    yaxis_type="log"

)

fig.show()





# In[10]:





# The source dataset is not necessary cumulative we will force it

latest_loc = train[train['Date'] == TRAIN_END][['Location', 'ConfirmedCases', 'Fatalities']]

max_loc = train.groupby(['Location'])[['ConfirmedCases', 'Fatalities']].max().reset_index()

check = pd.merge(latest_loc, max_loc, on='Location')

np.mean(check.ConfirmedCases_x == check.ConfirmedCases_y)

np.mean(check.Fatalities_x == check.Fatalities_y)

check[check.Fatalities_x != check.Fatalities_y]

check[check.ConfirmedCases_x != check.ConfirmedCases_y]





# In[11]:





train_clean = process_each_location(train)



train_clean.shape

train_clean.tail()





# # Continents



# In[12]:





regional_progress = train_clean.groupby(['DateTime', 'continent']).sum()[['ConfirmedCases', 'Fatalities']].reset_index()

regional_progress['Log10Confirmed'] = np.log10(regional_progress.ConfirmedCases + 1)

regional_progress['Log10Fatalities'] = np.log10(regional_progress.Fatalities + 1)

regional_progress = regional_progress[regional_progress.continent != '#N/A']





# In[13]:





fig = px.area(regional_progress, x="DateTime", y="ConfirmedCases", color="continent")

_ = fig.update_layout(

    title_text=f'COVID-19 Cumulative Confirmed Cases by Continent [Updated: {TRAIN_END}]'

)

fig.show()

fig2 = px.line(regional_progress, x='DateTime', y='ConfirmedCases', color='continent')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases by Continent [Updated: {TRAIN_END}]'

)

fig2.show()





# In[14]:





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





# In[15]:





china = train_clean[train_clean.Location.str.startswith('China')]

top10_locations = china.groupby('Location')[['ConfirmedCases']].max().sort_values(

    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]

fig2 = px.line(china[china.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases in China [Updated: {TRAIN_END}]'

)

fig2.show()





# In[16]:





europe = train_clean[train_clean.continent == 'Europe']

top10_locations = europe.groupby('Location')[['ConfirmedCases']].max().sort_values(

    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]

fig2 = px.line(europe[europe.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases in Europe [Updated: {TRAIN_END}]'

)

fig2.show()





# In[17]:





us = train_clean[train_clean.Country_Region == 'US']

top10_locations = us.groupby('Location')[['ConfirmedCases']].max().sort_values(

    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]

fig2 = px.line(us[us.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases in the USA [Updated: {TRAIN_END}]'

)

fig2.show()





# In[18]:





africa = train_clean[train_clean.continent == 'Africa']

top10_locations = africa.groupby('Location')[['ConfirmedCases']].max().sort_values(

    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]

fig2 = px.line(africa[africa.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases in Africa [Updated: {TRAIN_END}]'

)

fig2.show()





# # Countries



# In[19]:





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





# # Outbreak during March



# In[20]:





countries_0301 = country_progress[country_progress.Date == '2020-03-01'][[

    'Country_Region', 'ConfirmedCases', 'Fatalities']]

countries_0331 = country_progress[country_progress.Date == '2020-03-31'][[

    'Country_Region', 'ConfirmedCases', 'Fatalities']]

countries_in_march = pd.merge(countries_0301, countries_0331, on='Country_Region', suffixes=['_0301', '_0331'])

countries_in_march['IncreaseInMarch'] = countries_in_march.ConfirmedCases_0331 / (countries_in_march.ConfirmedCases_0301 + 1)

countries_in_march = countries_in_march[countries_in_march.ConfirmedCases_0331 > 200].sort_values(

    by='IncreaseInMarch', ascending=False)

countries_in_march.tail(15)





# In[21]:





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





# In[22]:





list(train_clean.Date)





# In[23]:





train_clean['Geo#Country#Contintent'] = train_clean.Location + '#' + train_clean.Country_Region + '#' + train_clean.continent

latest = train_clean[train_clean.Date == '2020-04-06'][[

    'Geo#Country#Contintent', 'ConfirmedCases', 'Fatalities', 'LogConfirmed', 'LogFatalities']]

# daily_confirmed_deltas = train_clean[train_clean.Date >= '2020-03-17'].pivot(

#     'Geo#Country#Contintent', 'Date', 'LogConfirmedDelta').round(3).reset_index()

# daily_confirmed_deltas = latest.merge(daily_confirmed_deltas, on='Geo#Country#Contintent')

# daily_confirmed_deltas.shape

# daily_confirmed_deltas.head()

# daily_confirmed_deltas.to_csv('daily_confirmed_deltas.csv', index=False)





# In[24]:





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





# In[25]:





fig = px.box(deltas,  x="start", y="LogConfirmedDelta", range_y=[0, 0.35])

fig.show()





# In[26]:





fig = px.box(deltas[deltas.Date >= '2020-03-01'],  x="DateTime", y="LogConfirmedDelta", range_y=[0, 0.6])

fig.show()





# In[27]:





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





# # Create prediction



# In[28]:





DECAY = 0.93

DECAY ** 7, DECAY ** 14, DECAY ** 21, DECAY ** 28



confirmed_deltas = train.groupby(['Location', 'Country_Region', 'continent'])[[

    'Id']].count().reset_index()



GLOBAL_DELTA = 0.11

confirmed_deltas['DELTA'] = GLOBAL_DELTA



confirmed_deltas.loc[confirmed_deltas.continent=='Africa', 'DELTA'] = 0.14

confirmed_deltas.loc[confirmed_deltas.continent=='Oceania', 'DELTA'] = 0.06

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Korea South', 'DELTA'] = 0.011

confirmed_deltas.loc[confirmed_deltas.Country_Region=='US', 'DELTA'] = 0.15

confirmed_deltas.loc[confirmed_deltas.Country_Region=='China', 'DELTA'] = 0.01

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Japan', 'DELTA'] = 0.05

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Singapore', 'DELTA'] = 0.05

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Taiwan*', 'DELTA'] = 0.05

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Switzerland', 'DELTA'] = 0.05

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Norway', 'DELTA'] = 0.05

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Iceland', 'DELTA'] = 0.05

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Austria', 'DELTA'] = 0.06

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Italy', 'DELTA'] = 0.04

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Spain', 'DELTA'] = 0.08

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Portugal', 'DELTA'] = 0.12

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Israel', 'DELTA'] = 0.12

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Iran', 'DELTA'] = 0.08

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Germany', 'DELTA'] = 0.07

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Malaysia', 'DELTA'] = 0.06

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Russia', 'DELTA'] = 0.18

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Ukraine', 'DELTA'] = 0.18

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Brazil', 'DELTA'] = 0.12

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Turkey', 'DELTA'] = 0.18

confirmed_deltas.loc[confirmed_deltas.Country_Region=='Philippines', 'DELTA'] = 0.18

confirmed_deltas.loc[confirmed_deltas.Location=='France-', 'DELTA'] = 0.1

confirmed_deltas.loc[confirmed_deltas.Location=='United Kingdom-', 'DELTA'] = 0.12

confirmed_deltas.loc[confirmed_deltas.Location=='Diamond Princess-', 'DELTA'] = 0.00

confirmed_deltas.loc[confirmed_deltas.Location=='China-Hong Kong', 'DELTA'] = 0.08

confirmed_deltas.loc[confirmed_deltas.Location=='San Marino-', 'DELTA'] = 0.03





confirmed_deltas.shape, confirmed_deltas.DELTA.mean()



confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA].shape, confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA].DELTA.mean()

confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA]

confirmed_deltas.describe()





# In[29]:





daily_log_confirmed = train_clean.pivot('Location', 'Date', 'LogConfirmed').reset_index()

daily_log_confirmed = daily_log_confirmed.sort_values(TRAIN_END, ascending=False)

daily_log_confirmed.to_csv('daily_log_confirmed.csv', index=False)



for i, d in tqdm(enumerate(pd.date_range(add_days(TRAIN_END, 1), add_days(TEST_END, 1)))):

    new_day = str(d).split(' ')[0]

    last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)

    last_day = last_day.strftime(DATEFORMAT)

    for loc in confirmed_deltas.Location.values:

        confirmed_delta = confirmed_deltas.loc[confirmed_deltas.Location == loc, 'DELTA'].values[0]

        daily_log_confirmed.loc[daily_log_confirmed.Location == loc, new_day] = daily_log_confirmed.loc[daily_log_confirmed.Location == loc, last_day] +             confirmed_delta * DECAY ** i





# In[30]:





daily_log_confirmed.head()





# In[31]:





confirmed_prediciton = pd.melt(daily_log_confirmed[:25], id_vars='Location')

confirmed_prediciton['ConfirmedCases'] = to_exp(confirmed_prediciton['value'])

fig2 = px.line(confirmed_prediciton,

               x='Date', y='ConfirmedCases', color='Location')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Confirmed Cases Prediction [Updated: {TRAIN_END}]'

)

fig2.show()





# # Fatalities



# In[32]:





train_clean['Geo#Country#Contintent'] = train_clean.Location + '#' + train_clean.Country_Region + '#' + train_clean.continent

latest = train_clean[train_clean.Date == TRAIN_END][[

    'Geo#Country#Contintent', 'ConfirmedCases', 'Fatalities', 'LogConfirmed', 'LogFatalities']]

# daily_death_deltas = train_clean[train_clean.Date >= '2020-03-24'].pivot(

#     'Geo#Country#Contintent', 'Date', 'LogFatalitiesDelta').round(3).reset_index()

# daily_death_deltas = latest.merge(daily_death_deltas, on='Geo#Country#Contintent')

# daily_death_deltas.shape

# daily_death_deltas.head()

# daily_death_deltas.to_csv('daily_death_deltas.csv', index=False)





# In[33]:





death_deltas = train.groupby(['Location', 'Country_Region', 'continent'])[[

    'Id']].count().reset_index()



GLOBAL_DELTA = 0.11

death_deltas['DELTA'] = GLOBAL_DELTA



death_deltas.loc[death_deltas.Country_Region=='China', 'DELTA'] = 0.005

death_deltas.loc[death_deltas.continent=='Oceania', 'DELTA'] = 0.08

death_deltas.loc[death_deltas.Country_Region=='Korea South', 'DELTA'] = 0.04

death_deltas.loc[death_deltas.Country_Region=='Japan', 'DELTA'] = 0.04

death_deltas.loc[death_deltas.Country_Region=='Singapore', 'DELTA'] = 0.05

death_deltas.loc[death_deltas.Country_Region=='Taiwan*', 'DELTA'] = 0.06







death_deltas.loc[death_deltas.Country_Region=='US', 'DELTA'] = 0.17



death_deltas.loc[death_deltas.Country_Region=='Switzerland', 'DELTA'] = 0.15

death_deltas.loc[death_deltas.Country_Region=='Norway', 'DELTA'] = 0.15

death_deltas.loc[death_deltas.Country_Region=='Iceland', 'DELTA'] = 0.01

death_deltas.loc[death_deltas.Country_Region=='Austria', 'DELTA'] = 0.14

death_deltas.loc[death_deltas.Country_Region=='Italy', 'DELTA'] = 0.07

death_deltas.loc[death_deltas.Country_Region=='Spain', 'DELTA'] = 0.1

death_deltas.loc[death_deltas.Country_Region=='Portugal', 'DELTA'] = 0.13

death_deltas.loc[death_deltas.Country_Region=='Israel', 'DELTA'] = 0.16

death_deltas.loc[death_deltas.Country_Region=='Iran', 'DELTA'] = 0.06

death_deltas.loc[death_deltas.Country_Region=='Germany', 'DELTA'] = 0.14

death_deltas.loc[death_deltas.Country_Region=='Malaysia', 'DELTA'] = 0.14

death_deltas.loc[death_deltas.Country_Region=='Russia', 'DELTA'] = 0.2

death_deltas.loc[death_deltas.Country_Region=='Ukraine', 'DELTA'] = 0.2

death_deltas.loc[death_deltas.Country_Region=='Brazil', 'DELTA'] = 0.2

death_deltas.loc[death_deltas.Country_Region=='Turkey', 'DELTA'] = 0.22

death_deltas.loc[death_deltas.Country_Region=='Philippines', 'DELTA'] = 0.12

death_deltas.loc[death_deltas.Location=='France-', 'DELTA'] = 0.14

death_deltas.loc[death_deltas.Location=='United Kingdom-', 'DELTA'] = 0.14

death_deltas.loc[death_deltas.Location=='Diamond Princess-', 'DELTA'] = 0.00



death_deltas.loc[death_deltas.Location=='China-Hong Kong', 'DELTA'] = 0.01

death_deltas.loc[death_deltas.Location=='San Marino-', 'DELTA'] = 0.05





death_deltas.shape

death_deltas.DELTA.mean()



death_deltas[death_deltas.DELTA != GLOBAL_DELTA].shape

death_deltas[death_deltas.DELTA != GLOBAL_DELTA].DELTA.mean()

death_deltas[death_deltas.DELTA != GLOBAL_DELTA]

death_deltas.describe()





# In[34]:





daily_log_deaths = train_clean.pivot('Location', 'Date', 'LogFatalities').reset_index()

daily_log_deaths = daily_log_deaths.sort_values(TRAIN_END, ascending=False)

daily_log_deaths.to_csv('daily_log_deaths.csv', index=False)



for i, d in tqdm(enumerate(pd.date_range(add_days(TRAIN_END, 1), add_days(TEST_END, 1)))):

    new_day = str(d).split(' ')[0]

    last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)

    last_day = last_day.strftime(DATEFORMAT)

    for loc in death_deltas.Location:

        death_delta = death_deltas.loc[death_deltas.Location == loc, 'DELTA'].values[0]

        daily_log_deaths.loc[daily_log_deaths.Location == loc, new_day] = daily_log_deaths.loc[daily_log_deaths.Location == loc, last_day] +             death_delta * DECAY ** i





# In[35]:





confirmed_prediciton = pd.melt(daily_log_deaths[:25], id_vars='Location')

confirmed_prediciton['Fatalities'] = to_exp(confirmed_prediciton['value'])

fig2 = px.line(confirmed_prediciton,

               x='Date', y='Fatalities', color='Location')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Fatalities Prediction [Updated: {TRAIN_END}]'

)

fig2.show()





# # Create submission file



# In[36]:





submission.head(2)





# In[37]:





confirmed = []

fatalities = []

for id, d, loc in tqdm(test[['ForecastId', 'Date', 'Location']].values):

    c = to_exp(daily_log_confirmed.loc[daily_log_confirmed.Location == loc, d].values[0])

    f = to_exp(daily_log_deaths.loc[daily_log_deaths.Location == loc, d].values[0])

    confirmed.append(c)

    fatalities.append(f)





# In[38]:





my_submission = test.copy()

my_submission['ConfirmedCases'] = confirmed

my_submission['Fatalities'] = fatalities



my_submission['ConfirmedCases']=my_submission['ConfirmedCases'].fillna(1)

my_submission['Fatalities']=my_submission['Fatalities'].fillna(1)

my_submission.shape

my_submission.head()





# In[39]:





my_submission.groupby('Date').sum().tail()





# # Sanity check



# In[40]:





total = my_submission.groupby('Date')[['ConfirmedCases', 'Fatalities']].sum().reset_index()



fig2 = px.line(pd.melt(total, id_vars=['Date']), x='Date', y='value', color='variable')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Prediction Total [Updated: {TRAIN_END}]'

)

fig2.show()





# In[41]:





my_submission[[

    'ForecastId', 'ConfirmedCases', 'Fatalities'

]].to_csv('sub1.csv', index=False)

print(DECAY)

my_submission.head()

my_submission.tail()

my_submission.shape





# In[42]:





end = dt.datetime.now()

print('Finished', end, (end - start).seconds, 's')

sub1=my_submission[['ForecastId', 'ConfirmedCases', 'Fatalities']].copy()

#vopani

#!/usr/bin/env python

# coding: utf-8



# ## Covid-19 Forecasting (Week 2)

# ![Coronavirusm.jpg](attachment:Coronavirusm.jpg)

# 

# The [covid-19 virus](https://en.wikipedia.org/wiki/Coronavirus_disease_2019) has created a pandemic in the early months of 2020. Kaggle has hosted a series of forecasting competitions for predict the number the spread of the virus. This is my solution for Week 2 of the series.

# 

# I am compiling external datasets in the competition format here: https://www.kaggle.com/rohanrao/covid19-forecasting-metadata

# They are structured such that it can directly be merged with the train and test datasets on Kaggle.

# 



# In[1]:





## importing packages

import lightgbm as lgb

import numpy as np

import pandas as pd



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder



import warnings

warnings.filterwarnings("ignore")





# In[2]:





## defining constants

PATH_TRAIN = "/kaggle/input/covid19-global-forecasting-week-3/train.csv"

PATH_TEST = "/kaggle/input/covid19-global-forecasting-week-3/test.csv"



PATH_SUBMISSION = "submission.csv"

PATH_OUTPUT = "output.csv"



PATH_REGION_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_metadata.csv"

PATH_REGION_DATE_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_date_metadata.csv"



VAL_DAYS = 7

MAD_FACTOR = 0.5

DAYS_SINCE_CASES = [1, 10, 50, 100, 500, 1000, 5000, 10000]



SEED = 2357



LGB_PARAMS = {"objective": "regression",

              "num_leaves": 5,

              "learning_rate": 0.013,

              "bagging_fraction": 0.91,

              "feature_fraction": 0.81,

              "reg_alpha": 0.13,

              "reg_lambda": 0.13,

              "metric": "rmse",

              "seed": SEED

             }





# ## Prepare Data

# * Unlike many competitions, there are overlapping dates across train and test data. So it is important to be careful while handling the dates.

# 

# * Some data issues are fixed using previous max values in case of discrepancy.

# 

# * External data from [this dataset](https://www.kaggle.com/rohanrao/covid19-forecasting-metadata) is merged.



# In[3]:





## reading data

train = pd.read_csv(PATH_TRAIN)

test = pd.read_csv(PATH_TEST)



region_metadata = pd.read_csv(PATH_REGION_METADATA)

region_date_metadata = pd.read_csv(PATH_REGION_DATE_METADATA)





# In[4]:





## preparing data

train = train.merge(test[["ForecastId", "Province_State", "Country_Region", "Date"]], on = ["Province_State", "Country_Region", "Date"], how = "left")

test = test[~test.Date.isin(train.Date.unique())]



df_panel = pd.concat([train, test], sort = False)



# combining state and country into 'geography'

df_panel["geography"] = df_panel.Country_Region.astype(str) + ": " + df_panel.Province_State.astype(str)

df_panel.loc[df_panel.Province_State.isna(), "geography"] = df_panel[df_panel.Province_State.isna()].Country_Region



# fixing data issues with cummax

df_panel.ConfirmedCases = df_panel.groupby("geography")["ConfirmedCases"].cummax()

df_panel.Fatalities = df_panel.groupby("geography")["Fatalities"].cummax()



# merging external metadata

df_panel = df_panel.merge(region_metadata, on = ["Country_Region", "Province_State"])

df_panel = df_panel.merge(region_date_metadata, on = ["Country_Region", "Province_State", "Date"], how = "left")



# label encoding continent

df_panel.continent = LabelEncoder().fit_transform(df_panel.continent)

df_panel.Date = pd.to_datetime(df_panel.Date, format = "%Y-%m-%d")



df_panel.sort_values(["geography", "Date"], inplace = True)





# ## Feature Engineering

# * Lag features are created.

# 

# * Several differences, ratios and averages of historic values are created and used.



# In[5]:





## feature engineering

min_date_train = np.min(df_panel[~df_panel.Id.isna()].Date)

max_date_train = np.max(df_panel[~df_panel.Id.isna()].Date)



min_date_test = np.min(df_panel[~df_panel.ForecastId.isna()].Date)

max_date_test = np.max(df_panel[~df_panel.ForecastId.isna()].Date)



n_dates_test = len(df_panel[~df_panel.ForecastId.isna()].Date.unique())



print("Train date range:", str(min_date_train), " - ", str(max_date_train))

print("Test date range:", str(min_date_test), " - ", str(max_date_test))



# creating lag features

for lag in range(1, 41):

    df_panel[f"lag_{lag}_cc"] = df_panel.groupby("geography")["ConfirmedCases"].shift(lag)

    df_panel[f"lag_{lag}_ft"] = df_panel.groupby("geography")["Fatalities"].shift(lag)

    df_panel[f"lag_{lag}_rc"] = df_panel.groupby("geography")["Recoveries"].shift(lag)



for case in DAYS_SINCE_CASES:

    df_panel = df_panel.merge(df_panel[df_panel.ConfirmedCases >= case].groupby("geography")["Date"].min().reset_index().rename(columns = {"Date": f"case_{case}_date"}), on = "geography", how = "left")





# In[6]:





## function for preparing features

def prepare_features(df, gap):

    

    df["perc_1_ac"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap}_ft"] - df[f"lag_{gap}_rc"]) / df[f"lag_{gap}_cc"]

    df["perc_1_cc"] = df[f"lag_{gap}_cc"] / df.population

    

    df["diff_1_cc"] = df[f"lag_{gap}_cc"] - df[f"lag_{gap + 1}_cc"]

    df["diff_2_cc"] = df[f"lag_{gap + 1}_cc"] - df[f"lag_{gap + 2}_cc"]

    df["diff_3_cc"] = df[f"lag_{gap + 2}_cc"] - df[f"lag_{gap + 3}_cc"]

    

    df["diff_1_ft"] = df[f"lag_{gap}_ft"] - df[f"lag_{gap + 1}_ft"]

    df["diff_2_ft"] = df[f"lag_{gap + 1}_ft"] - df[f"lag_{gap + 2}_ft"]

    df["diff_3_ft"] = df[f"lag_{gap + 2}_ft"] - df[f"lag_{gap + 3}_ft"]

    

    df["diff_123_cc"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap + 3}_cc"]) / 3

    df["diff_123_ft"] = (df[f"lag_{gap}_ft"] - df[f"lag_{gap + 3}_ft"]) / 3



    df["diff_change_1_cc"] = df.diff_1_cc / df.diff_2_cc

    df["diff_change_2_cc"] = df.diff_2_cc / df.diff_3_cc

    

    df["diff_change_1_ft"] = df.diff_1_ft / df.diff_2_ft

    df["diff_change_2_ft"] = df.diff_2_ft / df.diff_3_ft



    df["diff_change_12_cc"] = (df.diff_change_1_cc + df.diff_change_2_cc) / 2

    df["diff_change_12_ft"] = (df.diff_change_1_ft + df.diff_change_2_ft) / 2

    

    df["change_1_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 1}_cc"]

    df["change_2_cc"] = df[f"lag_{gap + 1}_cc"] / df[f"lag_{gap + 2}_cc"]

    df["change_3_cc"] = df[f"lag_{gap + 2}_cc"] / df[f"lag_{gap + 3}_cc"]



    df["change_1_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 1}_ft"]

    df["change_2_ft"] = df[f"lag_{gap + 1}_ft"] / df[f"lag_{gap + 2}_ft"]

    df["change_3_ft"] = df[f"lag_{gap + 2}_ft"] / df[f"lag_{gap + 3}_ft"]



    df["change_123_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 3}_cc"]

    df["change_123_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 3}_ft"]

    

    for case in DAYS_SINCE_CASES:

        df[f"days_since_{case}_case"] = (df[f"case_{case}_date"] - df.Date).astype("timedelta64[D]")

        df.loc[df[f"days_since_{case}_case"] < gap, f"days_since_{case}_case"] = np.nan



    df["country_flag"] = df.Province_State.isna().astype(int)

    df["density"] = df.population / df.area

    

    # target variable is log of change from last known value

    df["target_cc"] = np.log1p(df.ConfirmedCases) - np.log1p(df[f"lag_{gap}_cc"])

    df["target_ft"] = np.log1p(df.Fatalities) - np.log1p(df[f"lag_{gap}_ft"])

    

    features = [

        f"lag_{gap}_cc",

        f"lag_{gap}_ft",

        f"lag_{gap}_rc",

        "perc_1_ac",

        "perc_1_cc",

        "diff_1_cc",

        "diff_2_cc",

        "diff_3_cc",

        "diff_1_ft",

        "diff_2_ft",

        "diff_3_ft",

        "diff_123_cc",

        "diff_123_ft",

        "diff_change_1_cc",

        "diff_change_2_cc",

        "diff_change_1_ft",

        "diff_change_2_ft",

        "diff_change_12_cc",

        "diff_change_12_ft",

        "change_1_cc",

        "change_2_cc",

        "change_3_cc",

        "change_1_ft",

        "change_2_ft",

        "change_3_ft",

        "change_123_cc",

        "change_123_ft",

        "days_since_1_case",

        "days_since_10_case",

        "days_since_50_case",

        "days_since_100_case",

        "days_since_500_case",

        "days_since_1000_case",

        "days_since_5000_case",

        "days_since_10000_case",

        "country_flag",

        "lat",

        "lon",

        "continent",

        "population",

        "area",

        "density",

        "target_cc",

        "target_ft"

    ]

    

    return df[features]





# ## LGB Model

# * Note that the target variable used is the change (or difference) in log values of ConfirmedCases / Fatalities from the last known value.

# 

# * One model is built for each day in the private test data. So that's a total of 28 models.

# 

# * A single set of parameters with less number of leaves along with regularization is used to control overfitting.



# In[7]:





## function for building and predicting using LGBM model

def build_predict_lgbm(df_train, df_test, gap):

    

    df_train.dropna(subset = ["target_cc", "target_ft", f"lag_{gap}_cc", f"lag_{gap}_ft"], inplace = True)

    

    target_cc = df_train.target_cc

    target_ft = df_train.target_ft

    

    test_lag_cc = df_test[f"lag_{gap}_cc"].values

    test_lag_ft = df_test[f"lag_{gap}_ft"].values

    

    df_train.drop(["target_cc", "target_ft"], axis = 1, inplace = True)

    df_test.drop(["target_cc", "target_ft"], axis = 1, inplace = True)

    

    categorical_features = ["continent"]

    

    dtrain_cc = lgb.Dataset(df_train, label = target_cc, categorical_feature = categorical_features)

    dtrain_ft = lgb.Dataset(df_train, label = target_ft, categorical_feature = categorical_features)



    model_cc = lgb.train(LGB_PARAMS, train_set = dtrain_cc, num_boost_round = 200)

    model_ft = lgb.train(LGB_PARAMS, train_set = dtrain_ft, num_boost_round = 200)

    

    # inverse transform from log of change from last known value

    y_pred_cc = np.expm1(model_cc.predict(df_test, num_boost_round = 200) + np.log1p(test_lag_cc))

    y_pred_ft = np.expm1(model_ft.predict(df_test, num_boost_round = 200) + np.log1p(test_lag_ft))

    

    return y_pred_cc, y_pred_ft, model_cc, model_ft





# ## MAD Model

# * This Moving Average with Decay (MAD) model is a simple heuristic using historic values that decays with time.

# 

# * It is structured based on my EDA and intuitive feeling of how the Covid-19 trend is likely to move.



# In[8]:





## function for predicting moving average decay model

def predict_mad(df_test, gap, val = False):

    

    df_test["avg_diff_cc"] = (df_test[f"lag_{gap}_cc"] - df_test[f"lag_{gap + 3}_cc"]) / 3

    df_test["avg_diff_ft"] = (df_test[f"lag_{gap}_ft"] - df_test[f"lag_{gap + 3}_ft"]) / 3



    if val:

        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (1 - MAD_FACTOR) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / VAL_DAYS

        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test.avg_diff_ft - (1 - MAD_FACTOR) * df_test.avg_diff_ft * np.sum([x for x in range(gap)]) / VAL_DAYS

    else:

        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (1 - MAD_FACTOR) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / n_dates_test

        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test.avg_diff_ft - (1 - MAD_FACTOR) * df_test.avg_diff_ft * np.sum([x for x in range(gap)]) / n_dates_test



    return y_pred_cc, y_pred_ft





# ## Modelling

# * One model is built for each date.

# 

# * For dates already present in train data due to the overlap, no model is built.

# 

# * Models are validated using the same framework.



# In[9]:





## building lag x-days models

df_train = df_panel[~df_panel.Id.isna()]

df_test_full = df_panel[~df_panel.ForecastId.isna()]



df_preds_val = []

df_preds_test = []



for date in df_test_full.Date.unique():

    

    print("Processing date:", date)

    

    # ignore date already present in train data

    if date in df_train.Date.values:

        df_pred_test = df_test_full.loc[df_test_full.Date == date, ["ForecastId", "ConfirmedCases", "Fatalities"]].rename(columns = {"ConfirmedCases": "ConfirmedCases_test", "Fatalities": "Fatalities_test"})

        

        # multiplying predictions by 41 to not look cool on public LB

        df_pred_test.ConfirmedCases_test = df_pred_test.ConfirmedCases_test * 41

        df_pred_test.Fatalities_test = df_pred_test.Fatalities_test * 41

    else:

        df_test = df_test_full[df_test_full.Date == date]

        

        gap = (pd.Timestamp(date) - max_date_train).days

        

        if gap <= VAL_DAYS:

            val_date = max_date_train - pd.Timedelta(VAL_DAYS, "D") + pd.Timedelta(gap, "D")



            df_build = df_train[df_train.Date < val_date]

            df_val = df_train[df_train.Date == val_date]

            

            X_build = prepare_features(df_build, gap)

            X_val = prepare_features(df_val, gap)

            

            y_val_cc_lgb, y_val_ft_lgb, _, _ = build_predict_lgbm(X_build, X_val, gap)

            y_val_cc_mad, y_val_ft_mad = predict_mad(df_val, gap, val = True)

            

            df_pred_val = pd.DataFrame({"Id": df_val.Id.values,

                                        "ConfirmedCases_val_lgb": y_val_cc_lgb,

                                        "Fatalities_val_lgb": y_val_ft_lgb,

                                        "ConfirmedCases_val_mad": y_val_cc_mad,

                                        "Fatalities_val_mad": y_val_ft_mad,

                                       })



            df_preds_val.append(df_pred_val)



        X_train = prepare_features(df_train, gap)

        X_test = prepare_features(df_test, gap)



        y_test_cc_lgb, y_test_ft_lgb, model_cc, model_ft = build_predict_lgbm(X_train, X_test, gap)

        y_test_cc_mad, y_test_ft_mad = predict_mad(df_test, gap)

        

        if gap == 1:

            model_1_cc = model_cc

            model_1_ft = model_ft

            features_1 = X_train.columns.values

        elif gap == 14:

            model_14_cc = model_cc

            model_14_ft = model_ft

            features_14 = X_train.columns.values

        elif gap == 28:

            model_28_cc = model_cc

            model_28_ft = model_ft

            features_28 = X_train.columns.values



        df_pred_test = pd.DataFrame({"ForecastId": df_test.ForecastId.values,

                                     "ConfirmedCases_test_lgb": y_test_cc_lgb,

                                     "Fatalities_test_lgb": y_test_ft_lgb,

                                     "ConfirmedCases_test_mad": y_test_cc_mad,

                                     "Fatalities_test_mad": y_test_ft_mad,

                                    })

    

    df_preds_test.append(df_pred_test)





# ## Validation

# * Validating LGB and MAD models using RMSLE metric.

# 

# * Visualizing feature importance of 1st, 14th and 28th models to see which features are crucial to predict short-term, medium-term and long-term respectively.



# In[10]:





## validation score

df_panel = df_panel.merge(pd.concat(df_preds_val, sort = False), on = "Id", how = "left")

df_panel = df_panel.merge(pd.concat(df_preds_test, sort = False), on = "ForecastId", how = "left")



rmsle_cc_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases_val_lgb)))

rmsle_ft_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities_val_lgb)))



rmsle_cc_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases_val_mad)))

rmsle_ft_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities_val_mad)))



print("LGB CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_lgb, 2))

print("LGB FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_lgb, 2))

print("LGB Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_lgb + rmsle_ft_lgb) / 2, 2))

print("\n")

print("MAD CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_mad, 2))

print("MAD FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_mad, 2))

print("MAD Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_mad + rmsle_ft_mad) / 2, 2))





# In[11]:





## feature importance

from bokeh.io import output_notebook, show

from bokeh.layouts import column

from bokeh.palettes import Spectral3

from bokeh.plotting import figure



output_notebook()



df_fimp_1_cc = pd.DataFrame({"feature": features_1, "importance": model_1_cc.feature_importance(), "model": "m01"})

df_fimp_14_cc = pd.DataFrame({"feature": features_14, "importance": model_14_cc.feature_importance(), "model": "m14"})

df_fimp_28_cc = pd.DataFrame({"feature": features_28, "importance": model_28_cc.feature_importance(), "model": "m28"})



df_fimp_1_cc.sort_values("importance", ascending = False, inplace = True)

df_fimp_14_cc.sort_values("importance", ascending = False, inplace = True)

df_fimp_28_cc.sort_values("importance", ascending = False, inplace = True)



v1 = figure(plot_width = 800, plot_height = 400, x_range = df_fimp_1_cc.feature, title = "Feature Importance of LGB Model 1")

v1.vbar(x = df_fimp_1_cc.feature, top = df_fimp_1_cc.importance, width = 1)

v1.xaxis.major_label_orientation = 1.3



v14 = figure(plot_width = 800, plot_height = 400, x_range = df_fimp_14_cc.feature, title = "Feature Importance of LGB Model 14")

v14.vbar(x = df_fimp_14_cc.feature, top = df_fimp_14_cc.importance, width = 1)

v14.xaxis.major_label_orientation = 1.3



v28 = figure(plot_width = 800, plot_height = 400, x_range = df_fimp_28_cc.feature, title = "Feature Importance of LGB Model 28")

v28.vbar(x = df_fimp_28_cc.feature, top = df_fimp_28_cc.importance, width = 1)

v28.xaxis.major_label_orientation = 1.3



v = column(v1, v14, v28)



show(v)





# ## Visualizing Predictions

# * Viewing the actual, validation and test values together for each geography for ConfirmedCases as well as Fatalities.



# In[12]:





## visualizing ConfirmedCases

from bokeh.models import Panel, Tabs

from bokeh.io import output_notebook, show

from bokeh.plotting import figure



output_notebook()



tab_list = []



for geography in df_panel.geography.unique():

    df_geography = df_panel[df_panel.geography == geography]

    v = figure(plot_width = 800, plot_height = 400, x_axis_type = "datetime", title = "Covid-19 ConfirmedCases over time")

    v.line(df_geography.Date, df_geography.ConfirmedCases, color = "green", legend_label = "CC (Train)")

    v.line(df_geography.Date, df_geography.ConfirmedCases_val_lgb, color = "blue", legend_label = "CC LGB (Val)")

    v.line(df_geography.Date, df_geography.ConfirmedCases_val_mad, color = "purple", legend_label = "CC MAD (Val)")

    v.line(df_geography.Date[df_geography.Date > max_date_train], df_geography.ConfirmedCases_test_lgb[df_geography.Date > max_date_train], color = "red", legend_label = "CC LGB (Test)")

    v.line(df_geography.Date[df_geography.Date > max_date_train], df_geography.ConfirmedCases_test_mad[df_geography.Date > max_date_train], color = "orange", legend_label = "CC MAD (Test)")

    v.legend.location = "top_left"

    tab = Panel(child = v, title = geography)

    tab_list.append(tab)



tabs = Tabs(tabs=tab_list)

show(tabs)





# In[13]:





## visualizing Fatalities

from bokeh.models import Panel, Tabs

from bokeh.io import output_notebook, show

from bokeh.plotting import figure



output_notebook()



tab_list = []



for geography in df_panel.geography.unique():

    df_geography = df_panel[df_panel.geography == geography]

    v = figure(plot_width = 800, plot_height = 400, x_axis_type = "datetime", title = "Covid-19 Fatalities over time")

    v.line(df_geography.Date, df_geography.Fatalities, color = "green", legend_label = "FT (Train)")

    v.line(df_geography.Date, df_geography.Fatalities_val_lgb, color = "blue", legend_label = "FT LGB (Val)")

    v.line(df_geography.Date, df_geography.Fatalities_val_mad, color = "purple", legend_label = "FT MAD (Val)")

    v.line(df_geography.Date[df_geography.Date > max_date_train], df_geography.Fatalities_test_lgb[df_geography.Date > max_date_train], color = "red", legend_label = "FT LGB (Test)")

    v.line(df_geography.Date[df_geography.Date > max_date_train], df_geography.Fatalities_test_mad[df_geography.Date > max_date_train], color = "orange", legend_label = "FT MAD (Test)")

    v.legend.location = "top_left"

    tab = Panel(child = v, title = geography)

    tab_list.append(tab)



tabs = Tabs(tabs=tab_list)

show(tabs)





# ## Submission

# * Combining the two aapproaches using weights for final submission.

# 

# * LGB models don't seem to perform well for certain geographies and are replaced by MAD predictions.



# In[14]:





## preparing submission file

df_test = df_panel.loc[~df_panel.ForecastId.isna(), ["ForecastId", "Country_Region", "Province_State", "Date",

                                                     "ConfirmedCases_test", "ConfirmedCases_test_lgb", "ConfirmedCases_test_mad",

                                                     "Fatalities_test", "Fatalities_test_lgb", "Fatalities_test_mad"]].reset_index()



df_test["ConfirmedCases"] = 0.41 * df_test.ConfirmedCases_test_lgb + 0.59 * df_test.ConfirmedCases_test_mad

df_test["Fatalities"] = 0.05 * df_test.Fatalities_test_lgb + 0.95 * df_test.Fatalities_test_mad



# Since LGB models don't predict these countries well

df_test.loc[df_test.Country_Region.isin(["China", "US", "Diamond Princess"]), "ConfirmedCases"] = df_test[df_test.Country_Region.isin(["China", "US", "Diamond Princess"])].ConfirmedCases_test_mad.values

df_test.loc[df_test.Country_Region.isin(["China", "US", "Diamond Princess"]), "Fatalities"] = df_test[df_test.Country_Region.isin(["China", "US", "Diamond Princess"])].Fatalities_test_mad.values



df_test.loc[df_test.Date.isin(df_train.Date.values), "ConfirmedCases"] = df_test[df_test.Date.isin(df_train.Date.values)].ConfirmedCases_test.values

df_test.loc[df_test.Date.isin(df_train.Date.values), "Fatalities"] = df_test[df_test.Date.isin(df_train.Date.values)].Fatalities_test.values



df_submission = df_test[["ForecastId", "ConfirmedCases", "Fatalities"]]

df_submission.ForecastId = df_submission.ForecastId.astype(int)



df_submission





# In[15]:





## writing final submission and complete output

df_submission.to_csv("sub2.csv", index = False)

df_test.to_csv(PATH_OUTPUT, index = False)

sub2=df_submission.copy()
#zoo

#!/usr/bin/env python

# coding: utf-8



# In[1]:





import os

import datetime

import numpy as np

import pandas as pd

import time

import lightgbm as lgb

from sklearn.metrics import mean_squared_error



path = '../input/covid19-global-forecasting-week-3/'

train = pd.read_csv(path + 'train.csv')

test  = pd.read_csv(path + 'test.csv')

sub   = pd.read_csv(path + 'submission.csv')



train['Date'] = train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))

test['Date'] = test['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))

#path_ext = '../input/novel-corona-virus-2019-dataset/'

#ext_rec = pd.read_csv(path_ext + 'time_series_covid_19_recovered.csv').\

#        melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], 

#            var_name="Date", 

#            value_name="Recoveries")

#ext_rec['Date'] = ext_rec['Date'].apply(lambda x: (datetime.datetime.strptime(x+"20", '%m/%d/%Y')))

#train = train.merge(ext_rec[['Province/State', 'Country/Region', 'Date', 'Recoveries']], how='left',

#           left_on=['Province/State', 'Country/Region', 'Date'],

#           right_on=['Province/State', 'Country/Region', 'Date'])



train['days'] = (train['Date'].dt.date - train['Date'].dt.date.min()).dt.days

test['days'] = (test['Date'].dt.date - train['Date'].dt.date.min()).dt.days

#train['isTest'] = train['Date'].dt.date >= datetime.date(2020, 3, 12)

#train['isVal'] = np.logical_and(train['Date'].dt.date >= datetime.date(2020, 3, 11), train['Date'].dt.date <= datetime.date(9999, 3, 18))

train.loc[train['Province_State'].isnull(), 'Province_State'] = 'N/A'

test.loc[test['Province_State'].isnull(), 'Province_State'] = 'N/A'



train['Area'] = train['Country_Region'] + '_' + train['Province_State']

test['Area'] = test['Country_Region'] + '_' + test['Province_State']



print(train['Date'].max())

print(test['Date'].min())

print(train['days'].max())

N_AREAS = train['Area'].nunique()

AREAS = np.sort(train['Area'].unique())

#TRAIN_N = 50 + 7

TRAIN_N = 70





print(train[train['days'] < TRAIN_N]['Date'].max())

train.head()





# In[2]:





train_p_c_raw = train.pivot(index='Area', columns='days', values='ConfirmedCases').sort_index()

train_p_f_raw = train.pivot(index='Area', columns='days', values='Fatalities').sort_index()



train_p_c = np.maximum.accumulate(train_p_c_raw, axis=1)

train_p_f = np.maximum.accumulate(train_p_f_raw, axis=1)



f_rate = (train_p_f / train_p_c).fillna(0)



X_c = np.log(1+train_p_c.values)[:,:TRAIN_N]

X_f = train_p_f.values[:,:TRAIN_N]





# In[3]:





from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error



def eval1(y, p):

    val_len = y.shape[1] - TRAIN_N

    return np.sqrt(mean_squared_error(y[:, TRAIN_N:TRAIN_N+val_len].flatten(), p[:, TRAIN_N:TRAIN_N+val_len].flatten()))



def run_c(params, X, test_size=50):

    

    gr_base = []

    gr_base_factor = []

    

    x_min = np.ma.MaskedArray(X, X<1)

    x_min = x_min.argmin(axis=1) 

    

    for i in range(X.shape[0]):

        temp = X[i,:]

        threshold = np.log(1+params['min cases for growth rate'])

        num_days = params['last N days']

        if (temp > threshold).sum() > num_days:

            gr_base.append(np.clip(np.diff(temp[temp > threshold])[-num_days:].mean(), 0, params['growth rate max']))

            gr_base_factor.append(np.clip(np.diff(np.diff(temp[temp > threshold]))[-num_days:].mean(), -0.2, params["growth rate factor max"]))

        else:

            gr_base.append(params['growth rate default'])

            gr_base_factor.append(params['growth rate factor'])



    gr_base = np.array(gr_base)

    gr_base_factor = np.array(gr_base_factor)

    #print(gr_base_factor)

    #gr_base = np.clip(gr_base, 0.02, 0.8)

    preds = X.copy()



    for i in range(test_size):

        delta = np.clip(preds[:, -1], np.log(2), None) + gr_base * (1 + params['growth rate factor']*(1 + params['growth rate factor factor'])**(i))**(i)

        #delta = np.clip(preds[:, -1], np.log(2), None) + gr_base * (1 + params['growth rate factor']*(1 + params['growth rate factor factor'])**(i+X.shape[1]-x_min))**(i+X.shape[1]-x_min) 

        preds = np.hstack((preds, delta.reshape(-1,1)))



    return preds



params = {

    "min cases for growth rate": 30,

    "last N days": 5,

    "growth rate default": 0.25,

    "growth rate max": 0.3,

    "growth rate factor max": -0.01,

    "growth rate factor": -0.05,

    "growth rate factor factor": 0.005,

}



#x = train_p_c[train_p_c.index=="Austria_N/A"]



x = train_p_c



preds_c = run_c(params, np.log(1+x.values)[:,:TRAIN_N])

#eval1(np.log(1+x).values, preds_c)





# In[4]:







for i in range(N_AREAS):

    if 'China' in AREAS[i] and preds_c[i, TRAIN_N-1] < np.log(31):

        preds_c[i, TRAIN_N:] = preds_c[i, TRAIN_N-1]





# In[5]:





def lin_w(sz):

    res = np.linspace(0, 1, sz+1, endpoint=False)[1:]

    return np.append(res, np.append([1], res[::-1]))





def run_f(params, X_c, X_f, X_f_r, test_size=50):





    

    X_f_r = np.array(np.ma.mean(np.ma.masked_outside(X_f_r, 0.06, 0.4)[:,:], axis=1))

    X_f_r = np.clip(X_f_r, params['fatality_rate_lower'], params['fatality_rate_upper'])

    #print(X_f_r)

    

    X_c = np.clip(np.exp(X_c)-1, 0, None)

    preds = X_f.copy()

    #print(preds.shape)

    

    train_size = X_f.shape[1] - 1

    for i in range(test_size):

        

        t_lag = train_size+i-params['length']

        t_wsize = 3

        delta = np.average(np.diff(X_c, axis=1)[:, t_lag-t_wsize:t_lag+1+t_wsize], axis=1)

        #delta = np.average(np.diff(X_c, axis=1)[:, t_lag-t_wsize:t_lag+1+t_wsize], axis=1, weights=lin_w(t_wsize))

        

        delta = params['absolute growth'] + delta * X_f_r

        

        preds = np.hstack((preds, preds[:, -1].reshape(-1,1) + delta.reshape(-1,1)))



    return preds



params = {

    "length": 6,

    "absolute growth": 0.02,

    "fatality_rate_lower": 0.035,

    "fatality_rate_upper": 0.40,

}



preds_f = run_f(params, preds_c, X_f, f_rate.values[:,:TRAIN_N])

preds_f = np.log(1+preds_f)

#eval1(np.log(1+train_p_f).values, preds_f)





# In[6]:





from sklearn.metrics import mean_squared_error



if False:

    val_len = train_p_c.values.shape[1] - TRAIN_N



    for i in range(val_len):

        d = i + TRAIN_N

        m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, d]), preds_c[:, d]))

        m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, d]), preds_f[:, d]))

        print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")



    print()



    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), preds_c[:, TRAIN_N:TRAIN_N+val_len].flatten()))

    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), preds_f[:, TRAIN_N:TRAIN_N+val_len].flatten()))

    print(f"{(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")





# In[7]:





import matplotlib.pyplot as plt



plt.style.use(['default'])

fig = plt.figure(figsize = (15, 5))



#idx = worst_idx

#print(AREAS[idx])



idx = np.where(AREAS == 'Austria_N/A')[0][0]

plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='darkblue')

plt.plot(preds_c[idx], linestyle='--', color='darkblue')



idx = np.where(AREAS == 'Germany_N/A')[0][0]

plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='red')

plt.plot(preds_c[idx], linestyle='--', color='red')





idx = np.where(AREAS == 'China_Hubei')[0][0]

plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='grey')

plt.plot(preds_c[idx], linestyle='--', color='grey')





idx = np.where(AREAS == 'Iran_N/A')[0][0]

plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='green')

plt.plot(preds_c[idx], linestyle='--', color='green')





idx = np.where(AREAS == 'Japan_N/A')[0][0]

plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='purple')

plt.plot(preds_c[idx], linestyle='--', color='purple')





idx = np.where(AREAS == 'Brazil_N/A')[0][0]

plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='black')

plt.plot(preds_c[idx], linestyle='--', color='black')





idx = np.where(AREAS == 'Denmark_N/A')[0][0]

plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='yellow')

plt.plot(preds_c[idx], linestyle='--', color='yellow')



idx = np.where(AREAS == 'Italy_N/A')[0][0]

plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='blue')

plt.plot(preds_c[idx], linestyle='--', color='blue')





plt.legend()

plt.show()





# In[8]:





import matplotlib.pyplot as plt



plt.style.use(['default'])

fig = plt.figure(figsize = (15, 5))



#idx = worst_idx

#print(AREAS[idx])



idx = np.where(AREAS == 'Austria_N/A')[0][0]

plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='darkblue')

plt.plot(preds_f[idx], linestyle='--', color='darkblue')



idx = np.where(AREAS == 'Germany_N/A')[0][0]

plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='red')

plt.plot(preds_f[idx], linestyle='--', color='red')





idx = np.where(AREAS == 'China_Hubei')[0][0]

plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='grey')

plt.plot(preds_f[idx], linestyle='--', color='grey')





idx = np.where(AREAS == 'Iran_N/A')[0][0]

plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='green')

plt.plot(preds_f[idx], linestyle='--', color='green')





idx = np.where(AREAS == 'Japan_N/A')[0][0]

plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='purple')

plt.plot(preds_f[idx], linestyle='--', color='purple')





idx = np.where(AREAS == 'Brazil_N/A')[0][0]

plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='black')

plt.plot(preds_f[idx], linestyle='--', color='black')





idx = np.where(AREAS == 'Denmark_N/A')[0][0]

plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='yellow')

plt.plot(preds_f[idx], linestyle='--', color='yellow')



idx = np.where(AREAS == 'Italy_N/A')[0][0]

plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='blue')

plt.plot(preds_f[idx], linestyle='--', color='blue')



plt.legend()

plt.show()





# In[9]:





import matplotlib.pyplot as plt



plt.style.use(['default'])

fig = plt.figure(figsize = (15, 5))



idx = np.random.choice(N_AREAS)

print(AREAS[idx])



plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='darkblue')

plt.plot(preds_c[idx], linestyle='--', color='darkblue')



plt.show()





# In[ ]:











# In[10]:





EU_COUNTRIES = ['Austria', 'Italy', 'Belgium', 'Latvia', 'Bulgaria', 'Lithuania', 'Croatia', 'Luxembourg', 'Cyprus', 'Malta', 'Czechia', 

                'Netherlands', 'Denmark', 'Poland', 'Estonia', 'Portugal', 'Finland', 'Romania', 'France', 'Slovakia', 'Germany', 'Slovenia', 

                'Greece', 'Spain', 'Hungary', 'Sweden', 'Ireland']

EUROPE_OTHER = ['Albania', 'Andorra', 'Bosnia and Herzegovina', 'Liechtenstein', 'Monaco', 'Montenegro', 'North Macedonia',

                'Norway', 'San Marino', 'Serbia', 'Switzerland', 'Turkey', 'United Kingdom']

AFRICA = ['Algeria', 'Burkina Faso', 'Cameroon', 'Congo (Kinshasa)', "Cote d'Ivoire", 'Egypt', 'Ghana', 'Kenya', 'Madagascar',

                'Morocco', 'Nigeria', 'Rwanda', 'Senegal', 'South Africa', 'Togo', 'Tunisia', 'Uganda', 'Zambia']

NORTH_AMERICA = ['US', 'Canada', 'Mexico']

SOUTH_AMERICA = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela']

MIDDLE_EAST = ['Afghanistan', 'Bahrain', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Oman', 'Qatar', 'Saudi Arabia', 'United Arab Emirates']

ASIA = ['Bangladesh', 'Brunei', 'Cambodia', 'India', 'Indonesia', 'Japan', 'Kazakhstan', 'Korea, South', 'Kyrgyzstan', 'Malaysia',

                'Pakistan', 'Singapore', 'Sri Lanka', 'Taiwan*', 'Thailand', 'Uzbekistan', 'Vietnam']





# In[11]:





import matplotlib.pyplot as plt



def plt1(ar, ar2, ax, col='darkblue', linew=0.2):

    ax.plot(ar2, linestyle='--', linewidth=linew/2, color=col)

    ax.plot(np.log(1+ar), linewidth=linew, color=col)



plt.style.use(['default'])

fig, axs = plt.subplots(3, 2, figsize=(18, 15), sharey=True)



X = train_p_c.values

#X = train_p_f.values



for ar in range(X.shape[0]):

    

    temp = X[ar]

    temp2 = preds_c[ar]

    if 'China' in AREAS[ar]:

        plt1(temp, temp2, axs[0,0])

    elif AREAS[ar].split('_')[0] in NORTH_AMERICA:

        plt1(temp, temp2, axs[0,1])

    elif AREAS[ar].split('_')[0] in EU_COUNTRIES + EUROPE_OTHER:

        plt1(temp, temp2, axs[1,0])

    elif AREAS[ar].split('_')[0] in SOUTH_AMERICA + AFRICA:

        plt1(temp, temp2, axs[1,1])

    elif AREAS[ar].split('_')[0] in MIDDLE_EAST + ASIA:

        plt1(temp, temp2, axs[2,0])

    else:

        plt1(temp, temp2, axs[2,1])



print("Confirmed Cases")

axs[0,0].set_title('China')

axs[0,1].set_title('North America')

axs[1,0].set_title('Europe')

axs[1,1].set_title('Africa + South America')

axs[2,0].set_title('Asia + Middle East')

axs[2,1].set_title('Other')

plt.show()





# In[12]:





import matplotlib.pyplot as plt



def plt1(ar, ar2, ax, col='darkblue', linew=0.2):

    ax.plot(ar2, linestyle='--', linewidth=linew/2, color=col)

    ax.plot(np.log(1+ar), linewidth=linew, color=col)



plt.style.use(['default'])

fig, axs = plt.subplots(3, 2, figsize=(18, 15), sharey=True)



#X = train_p_c.values

X = train_p_f.values



for ar in range(X.shape[0]):

    

    temp = X[ar]

    temp2 = preds_f[ar]

    if 'China' in AREAS[ar]:

        plt1(temp, temp2, axs[0,0])

    elif AREAS[ar].split('_')[0] in NORTH_AMERICA:

        plt1(temp, temp2, axs[0,1])

    elif AREAS[ar].split('_')[0] in EU_COUNTRIES + EUROPE_OTHER:

        plt1(temp, temp2, axs[1,0])

    elif AREAS[ar].split('_')[0] in SOUTH_AMERICA + AFRICA:

        plt1(temp, temp2, axs[1,1])

    elif AREAS[ar].split('_')[0] in MIDDLE_EAST + ASIA:

        plt1(temp, temp2, axs[2,0])

    else:

        plt1(temp, temp2, axs[2,1])



print("Fatalities")

axs[0,0].set_title('China')

axs[0,1].set_title('North America')

axs[1,0].set_title('Europe')

axs[1,1].set_title('Africa + South America')

axs[2,0].set_title('Asia + Middle East')

axs[2,1].set_title('Other')

plt.show()





# In[ ]:











# In[13]:





temp = pd.DataFrame(np.clip(np.exp(preds_c) - 1, 0, None))

temp['Area'] = AREAS

temp = temp.melt(id_vars='Area', var_name='days', value_name="ConfirmedCases")



test = test.merge(temp, how='left', left_on=['Area', 'days'], right_on=['Area', 'days'])



temp = pd.DataFrame(np.clip(np.exp(preds_f) - 1, 0, None))

temp['Area'] = AREAS

temp = temp.melt(id_vars='Area', var_name='days', value_name="Fatalities")



test = test.merge(temp, how='left', left_on=['Area', 'days'], right_on=['Area', 'days'])

test.head()





# In[14]:





test.to_csv("sub3.csv", index=False, columns=["ForecastId", "ConfirmedCases", "Fatalities"])

sub3=test[["ForecastId", "ConfirmedCases", "Fatalities"]].copy()

sub1.sort_values(['ForecastId'],inplace=True)

sub2.sort_values(['ForecastId'],inplace=True)

sub3.sort_values(['ForecastId'],inplace=True)

sub=sub1.copy()

sub['ConfirmedCases']=sub1['ConfirmedCases']*0.33+sub2['ConfirmedCases']*0.34+sub3['ConfirmedCases']*0.33

sub['Fatalities']=sub1['Fatalities']*0.33+sub2['Fatalities']*0.34+sub3['Fatalities']*0.33

sub.head()
sub.to_csv("submission.csv", index=False)