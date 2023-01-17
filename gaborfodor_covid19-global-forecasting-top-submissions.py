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
DATEFORMAT = '%Y-%m-%d'





def get_comp_data(COMP):

    train = pd.read_csv(f'{COMP}/train.csv')

    test = pd.read_csv(f'{COMP}/test.csv')

    print(train.shape, test.shape)

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

    

    return train, test





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
COMP = '../input/covid19-global-forecasting-week-4/'

start = dt.datetime.now()

train, test = get_comp_data(COMP)

train.shape, test.shape

train.head(2)

test.head(2)
train[train.geo_region.isna()].Country_Region.unique()

train = train.fillna('#N/A')

test = test.fillna('#N/A')



train[train.duplicated(['Date', 'Location'])]

train.count()

TRAIN_START = train.Date.min()

TEST_START = test.Date.min()

TRAIN_END = train.Date.max()

TEST_END = test.Date.max()

TRAIN_START, TRAIN_END, TEST_START, TEST_END



train_w2, test_w2 = get_comp_data('../input/covid19-global-forecasting-week-2')

train_w2.shape, test_w2.shape

train_w3, test_w3 = get_comp_data('../input/covid19-global-forecasting-week-3')

train_w3.shape, test_w3.shape
top_submissions = dict(

    beluga = '../input/covid-19-w3-a-few-charts-and-submission/submission.csv',

    Kaz = '../input/gbr-169v3-v2/submission.csv',

    Lockdown = '../input/gbt3n/submission.csv',

    PDD = '../input/cv19w3-v3fix-cpmp-oscii-belug-full-8118/submission.csv',

    KGMON = '../input/covid19-w3-submission-blend-4-models/submission.csv',

    Vopani = '../input/covid-19-w3-lgb-mad/submission.csv',

    OsciiArt = '../input/covid-19-lightgbm-with-weather-2/submission.csv',

    Northquay = '../input/c19-wk3-uploader/submission.csv',

)

top_submissions



predictions = train.copy()

predictions.shape



SUBM_DIR = './data/subms/'

for team, f in top_submissions.items():

    s = pd.read_csv(f)

    s = s.merge(test_w3, on='ForecastId')

    s = s[['Location', 'Date', 'ConfirmedCases', 'Fatalities']]

    

    predictions = predictions.merge(s, on=['Location', 'Date'], suffixes = ['', f' ({team})'], how='outer')

    s.shape

    s.tail(2)



daily_total = predictions.groupby('Date').sum().reset_index()

daily_total = daily_total[daily_total.Date > train_w3.Date.max()]

cols = ['Date'] + [c for c in predictions.columns if c.startswith('Confirmed')]



melted = pd.melt(daily_total[cols], id_vars='Date')



fig2 = px.line(melted, x='Date', y='value', color='variable')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'W3 Submissions - Confirmed Cases [Updated: {TRAIN_END}]'

)

fig2.show()
daily_total = predictions.groupby('Date').sum().reset_index()

daily_total = daily_total[daily_total.Date > train_w3.Date.max()]

cols = ['Date'] + [c for c in predictions.columns if c.startswith('Fatalities')]



melted = pd.melt(daily_total[cols], id_vars='Date')



fig2 = px.line(melted, x='Date', y='value', color='variable')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'W3 Submissions - Fatalities [Updated: {TRAIN_END}]'

)

fig2.show()
top_submissions = dict(

    beluga = '../input/covid-19-a-few-charts-and-a-simple-baseline/submission.csv',

    Kaz = '../input/gr1621-v2/submission.csv',

    KGMON = '../input/covid19-w2-final-v2/submission.csv',

    Vopani = '../input/covid19-metadata/w2_submission_Vopani.csv',

    OsciiArt = '../input/covid19-metadata/w2_submission_OsciiArt.csv',

    DAV = '../input/kernel303a0b031a/submission.csv'

)

top_submissions



predictions = train.copy()

predictions.shape



SUBM_DIR = './data/subms/'

for team, f in top_submissions.items():

    s = pd.read_csv(f)

    s = s.merge(test_w2, on='ForecastId')

    s = s[['Location', 'Date', 'ConfirmedCases', 'Fatalities']]

    

    predictions = predictions.merge(s, on=['Location', 'Date'], suffixes = ['', f' ({team})'], how='outer')

    s.shape

    s.tail(2)
melted
daily_total = predictions.groupby('Date').sum().reset_index()

daily_total = daily_total[daily_total.Date > train_w2.Date.max()]

cols = ['Date'] + [c for c in predictions.columns if c.startswith('Confirmed')]



melted = pd.melt(daily_total[cols], id_vars='Date')



fig2 = px.line(melted[melted.Date <= '2020-04-30'], x='Date', y='value', color='variable')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'W2 Submissions - Confirmed Cases [Updated: {TRAIN_END}]',

    width = 1600,

    height = 800,

)

fig2.show()
daily_total = predictions.groupby('Date').sum().reset_index()

daily_total = daily_total[daily_total.Date > train_w2.Date.max()]

cols = ['Date'] + [c for c in predictions.columns if c.startswith('Fatalities')]



melted = pd.melt(daily_total[cols], id_vars='Date')



fig2 = px.line(melted[melted.Date <= '2020-04-30'], x='Date', y='value', color='variable')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'W2 Submissions - Fatalities [Updated: {TRAIN_END}]',

    width = 1600,

    height = 800,

)

fig2.show()
train_w1, test_w1 = get_comp_data('../input/covid19-metadata')

train_w1.shape, test_w1.shape



top_submissions = dict(

    RalphNeumann='../input/covid19-metadata/w1_RalphNeumann_submission.csv',

    StephenKeller='../input/covid19-metadata/w1_StephenKeller_submission.csv', 

    beluga='../input/covid19-metadata/w1_beluga_submission.csv',

    OsciiArt='../input/covid19-metadata/w1_OsciiArt_submission.csv'   

)

top_submissions



predictions = train.copy()

predictions.shape



SUBM_DIR = './data/subms/'

for team, f in top_submissions.items():

    s = pd.read_csv(f)

    s = s.merge(test_w1, on='ForecastId')

    s = s[['Location', 'Date', 'ConfirmedCases', 'Fatalities']]

    

    predictions = predictions.merge(s, on=['Location', 'Date'], suffixes = ['', f' ({team})'], how='outer')

    s.shape

    s.tail(2)
daily_total = predictions.groupby('Date').sum().reset_index()

daily_total = daily_total[daily_total.Date > train_w1.Date.max()]

cols = ['Date'] + [c for c in predictions.columns if c.startswith('Confirmed')]



melted = pd.melt(daily_total[cols], id_vars='Date')



fig2 = px.line(melted[melted.Date <= '2020-04-23'], x='Date', y='value', color='variable')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'W1 Submissions - Confirmed Cases [Updated: {TRAIN_END}]'

)

fig2.show()
daily_total = predictions.groupby('Date').sum().reset_index()

daily_total = daily_total[daily_total.Date > train_w2.Date.max()]

cols = ['Date'] + [c for c in predictions.columns if c.startswith('Fatalities')]



melted = pd.melt(daily_total[cols], id_vars='Date')



fig2 = px.line(melted[melted.Date <= '2020-04-23'], x='Date', y='value', color='variable')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'W1 Submissions - Fatalities [Updated: {TRAIN_END}]'

)

fig2.show()