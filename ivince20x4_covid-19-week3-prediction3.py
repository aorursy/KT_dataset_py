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

import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))

import plotly.express as px
import plotly.graph_objects as go
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
countries_latest_state = train[train['Date'] == TRAIN_END].groupby(['Country_Region']).sum()[['ConfirmedCases', 
                                                                                              'Fatalities']].reset_index()
countries_latest_state['Log10Confirmed'] = np.log10(countries_latest_state.ConfirmedCases + 1)
countries_latest_state['Log10Fatalities'] = np.log10(countries_latest_state.Fatalities + 1)
countries_latest_state = countries_latest_state.sort_values(by='Fatalities', ascending=False)

countries_latest_state['DeathConfirmedRatio'] = (countries_latest_state.Fatalities + 1) / (countries_latest_state.ConfirmedCases + 1)
countries_latest_state['DeathConfirmedRatio'] = countries_latest_state['DeathConfirmedRatio'].clip(0, 0.1)

countries_latest_state.shape
countries_latest_state.head()
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
train_clean['Location'].nunique()
country_progress = train_clean.groupby(['Date', 'DateTime', 'Country_Region']).sum()[[
    'ConfirmedCases', 'Fatalities', 'ConfirmedDelta', 'FatalitiesDelta']].reset_index()
countries_0301 = country_progress[country_progress.Date == '2020-03-01'][[
    'Country_Region', 'ConfirmedCases', 'Fatalities']]
countries_0331 = country_progress[country_progress.Date == '2020-03-31'][[
    'Country_Region', 'ConfirmedCases', 'Fatalities']]
countries_in_march = pd.merge(countries_0301, countries_0331, on='Country_Region', suffixes=['_0301', '_0331'])
countries_in_march['IncreaseInMarch'] = countries_in_march.ConfirmedCases_0331 / (countries_in_march.ConfirmedCases_0301 + 1)
countries_in_march = countries_in_march[countries_in_march.ConfirmedCases_0331 > 200].sort_values(
    by='IncreaseInMarch', ascending=False)
countries_in_march.tail(15)
train_clean['Geo#Country#Contintent'] = train_clean.Location + '#' + train_clean.Country_Region
latest = train_clean[train_clean.Date == TRAIN_END][[
    'Geo#Country#Contintent', 'ConfirmedCases', 'Fatalities', 'LogConfirmed', 'LogFatalities']]
daily_confirmed_deltas = train_clean[train_clean.Date >= '2020-03-17'].pivot(
    'Geo#Country#Contintent', 'Date', 'LogConfirmedDelta').round(3).reset_index()
daily_confirmed_deltas = latest.merge(daily_confirmed_deltas, on='Geo#Country#Contintent')
daily_confirmed_deltas.shape
daily_confirmed_deltas.head()
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
DECAY = 0.93
DECAY ** 7, DECAY ** 14, DECAY ** 21, DECAY ** 28

confirmed_deltas = train.groupby(['Location', 'Country_Region'])[[
    'Id']].count().reset_index()

GLOBAL_DELTA = 0.11
confirmed_deltas['DELTA'] = GLOBAL_DELTA

#confirmed_deltas.loc[confirmed_deltas.continent=='Africa', 'DELTA'] = 0.14
#confirmed_deltas.loc[confirmed_deltas.continent=='Oceania', 'DELTA'] = 0.06
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Korea South', 'DELTA'] = 0.011
confirmed_deltas.loc[confirmed_deltas.Country_Region=='US', 'DELTA'] = 0.15
confirmed_deltas.loc[confirmed_deltas.Country_Region=='China', 'DELTA'] = 0.00
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Japan', 'DELTA'] = 0.05
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Singapore', 'DELTA'] = 0.04
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Taiwan*', 'DELTA'] = 0.04
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
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Malaysia', 'DELTA'] = 0.04
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Russia', 'DELTA'] = 0.18
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Ukraine', 'DELTA'] = 0.18
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Brazil', 'DELTA'] = 0.12
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Turkey', 'DELTA'] = 0.16
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Philippines', 'DELTA'] = 0.16
confirmed_deltas.loc[confirmed_deltas.Location=='France-', 'DELTA'] = 0.1
confirmed_deltas.loc[confirmed_deltas.Location=='United Kingdom-', 'DELTA'] = 0.12
confirmed_deltas.loc[confirmed_deltas.Location=='Diamond Princess-', 'DELTA'] = 0.00
confirmed_deltas.loc[confirmed_deltas.Location=='China-Hong Kong', 'DELTA'] = 0.08
confirmed_deltas.loc[confirmed_deltas.Location=='San Marino-', 'DELTA'] = 0.03


confirmed_deltas.shape, confirmed_deltas.DELTA.mean()

confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA].shape, confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA].DELTA.mean()
confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA]
confirmed_deltas.describe()
daily_log_confirmed = train_clean.pivot('Location', 'Date', 'LogConfirmed').reset_index()
daily_log_confirmed = daily_log_confirmed.sort_values(TRAIN_END, ascending=False)

for i, d in tqdm(enumerate(pd.date_range(add_days(TRAIN_END, 1), add_days(TEST_END, 1)))):
    new_day = str(d).split(' ')[0]
    last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)
    last_day = last_day.strftime(DATEFORMAT)
    for loc in confirmed_deltas.Location.values:
        confirmed_delta = confirmed_deltas.loc[confirmed_deltas.Location == loc, 'DELTA'].values[0]
        daily_log_confirmed.loc[daily_log_confirmed.Location == loc, new_day] = daily_log_confirmed.loc[daily_log_confirmed.Location == loc, last_day] + \
            confirmed_delta * DECAY ** i
daily_log_confirmed.head()
confirmed_prediciton = pd.melt(daily_log_confirmed[:25], id_vars='Location')
confirmed_prediciton['ConfirmedCases'] = to_exp(confirmed_prediciton['value'])
confirmed_prediciton.shape
train_clean['Geo#Country#Contintent'] = train_clean.Location + '#' + train_clean.Country_Region
latest = train_clean[train_clean.Date == TRAIN_END][[
    'Geo#Country#Contintent', 'ConfirmedCases', 'Fatalities', 'LogConfirmed', 'LogFatalities']]
daily_death_deltas = train_clean[train_clean.Date >= '2020-03-17'].pivot(
    'Geo#Country#Contintent', 'Date', 'LogFatalitiesDelta').round(3).reset_index()
daily_death_deltas = latest.merge(daily_death_deltas, on='Geo#Country#Contintent')
daily_death_deltas.shape
daily_death_deltas.head()
death_deltas = train.groupby(['Location', 'Country_Region'])[[
    'Id']].count().reset_index()

GLOBAL_DELTA = 0.11
death_deltas['DELTA'] = GLOBAL_DELTA

death_deltas.loc[death_deltas.Country_Region=='China', 'DELTA'] = 0.000
#death_deltas.loc[death_deltas.continent=='Oceania', 'DELTA'] = 0.08
death_deltas.loc[death_deltas.Country_Region=='Korea South', 'DELTA'] = 0.04
death_deltas.loc[death_deltas.Country_Region=='Japan', 'DELTA'] = 0.04
death_deltas.loc[death_deltas.Country_Region=='Singapore', 'DELTA'] = 0.04
death_deltas.loc[death_deltas.Country_Region=='Taiwan*', 'DELTA'] = 0.04

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
death_deltas.loc[death_deltas.Country_Region=='Malaysia', 'DELTA'] = 0.10
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
daily_log_deaths = train_clean.pivot('Location', 'Date', 'LogFatalities').reset_index()
daily_log_deaths = daily_log_deaths.sort_values(TRAIN_END, ascending=False)

for i, d in tqdm(enumerate(pd.date_range(add_days(TRAIN_END, 1), add_days(TEST_END, 1)))):
    new_day = str(d).split(' ')[0]
    last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)
    last_day = last_day.strftime(DATEFORMAT)
    for loc in death_deltas.Location:
        death_delta = death_deltas.loc[death_deltas.Location == loc, 'DELTA'].values[0]
        daily_log_deaths.loc[daily_log_deaths.Location == loc, new_day] = daily_log_deaths.loc[daily_log_deaths.Location == loc, last_day] + \
            death_delta * DECAY ** i
confirmed_prediciton = pd.melt(daily_log_deaths[:25], id_vars='Location')
confirmed_prediciton['Fatalities'] = to_exp(confirmed_prediciton['value'])
confirmed_prediciton.shape
submission.head(2)
confirmed = []
fatalities = []
for id, d, loc in tqdm(test[['ForecastId', 'Date', 'Location']].values):
    c = to_exp(daily_log_confirmed.loc[daily_log_confirmed.Location == loc, d].values[0])
    f = to_exp(daily_log_deaths.loc[daily_log_deaths.Location == loc, d].values[0])
    confirmed.append(c)
    fatalities.append(f)
my_submission = test.copy()
my_submission['ConfirmedCases'] = confirmed
my_submission['Fatalities'] = fatalities
my_submission.shape
my_submission.head()



my_submission.groupby('Date').sum().tail()
my_submission.Location.nunique()
import warnings
warnings.filterwarnings("ignore")
for place in my_submission.Location.unique():
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(my_submission[my_submission['Location']==place].groupby('Date')['ConfirmedCases'].sum(), 
               marker='o',color='b', linestyle='--')
    ax[1].plot(my_submission[my_submission['Location']==place].groupby('Date')['Fatalities'].sum(), 
               marker='v',color='r',linestyle='--')
    ax[0].set_ylabel('Predicted cases')
    ax[1].set_ylabel('Predicted deaths')
    ax[1].set_xlabel('Date')
    plt.xticks(rotation=45)

    ax[0].set_title('Total predicted cases and fatalities in {}'.format(place))
    plt.show()
my_submission[['ConfirmedCases','Fatalities']]= my_submission[['ConfirmedCases','Fatalities']].round(1)
my_submission[[
    'ForecastId', 'ConfirmedCases', 'Fatalities'
]].to_csv('submission.csv', index=False)
print(DECAY)
my_submission.head()
my_submission.tail()
my_submission.shape