# import standard libraries

import pandas as pd

import numpy as np

import os

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns



# hide warnings for cleaner output (there were some regarding indexing not using pandas .loc reference)

import warnings

warnings.filterwarnings('ignore')
# read in data (provided via course website), week 1

data_dir = '/kaggle/input/covid19-global-forecasting-week-1'

wk1_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

wk1_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))



# extract geocode, use average location

gc = wk1_train[['Country/Region', 'Lat', 'Long']]

gc = gc.groupby('Country/Region').mean().reset_index()
# read in data (provided via course website), week 2

data_dir = '/kaggle/input/covid19-global-forecasting-week-2'

wk2_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

wk2_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))



# make heading consistent with week1

wk2_train.columns = list(wk1_train.columns[:3]) + list(wk2_train.columns[3:])

wk2_test.columns = list(wk1_test.columns[:3]) + list(wk2_test.columns[3:])



# add geocode to week 2

wk2_train = wk2_train.merge(gc, on='Country/Region')

wk2_test = wk2_test.merge(gc, on='Country/Region')

# daily visualization

# Note: resource intensive



cases = wk2_train[['Country/Region', 'Lat', 'Long', 'Date', 'ConfirmedCases', 'Fatalities']]

cases_melt = pd.melt(cases, id_vars = ['Country/Region', 'Lat', 'Long', 'Date'])



fig = px.scatter_geo(cases_melt, lat = 'Lat', lon = 'Long', 

                     size = 'value', color = 'variable', 

                     animation_frame='Date', projection = 'natural earth')

fig.show()
# visualizing the cases by date for hot spots: China, Italy, Spain, US

# these should represent the major topics that are in the news in recent days

cases_by_date = wk2_train[['Country/Region', 'Date', 'ConfirmedCases', 'Fatalities']]

countries = ['China', 'Italy', 'Spain', 'US']

cases_by_date = cases_by_date[cases_by_date['Country/Region'].isin(countries) ]

cases_by_date = cases_by_date.groupby(['Country/Region', 'Date']).sum().reset_index()



cases_by_date_melt = pd.melt(cases_by_date, id_vars=['Country/Region', 'Date'])



fig = px.line(cases_by_date_melt, x = 'Date', y = 'value', color='Country/Region', facet_col='variable')

fig.show()
# does the fatalitiy rate increase? decrease? remain steady?

fatalities_ratio = cases_by_date['Fatalities'] / cases_by_date['ConfirmedCases']

fatalities_ratio.name = 'Fatalities Ratio'

fatalities_ratio = cases_by_date[['Date', 'Country/Region']].join(fatalities_ratio)

fig = px.line(fatalities_ratio, x = 'Date', y = 'Fatalities Ratio', color = 'Country/Region')

fig.show()
# evaluating the mean number of cases globally

mean_cases_by_date = cases_by_date_melt.groupby(['Date', 'variable']).mean().reset_index()

std_cases_by_date = cases_by_date_melt.groupby(['Date', 'variable']).std().reset_index()



mean_cases_by_date['std'] = std_cases_by_date['value']



fig = px.scatter(mean_cases_by_date, x='Date', y='value', facet_col='variable')

fig.show()
# looking at the standard deviation of the average

fig = px.scatter(mean_cases_by_date, x='Date', y='std', facet_col='variable')

fig.show()
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression



# Note on RMSLE (root mean squrared log error) https://www.kaggle.com/questions-and-answers/60012

from sklearn.metrics import mean_squared_log_error



# method for evaluating both the linear (with log transformation) and logistic fit

def regressions(train_X, train_y, test_X, test_y):

    # linear regression (with log transformation)

    lin_reg = LinearRegression().fit(train_X, np.log1p(train_y))

    lin_train_RMSLE = np.sqrt(mean_squared_log_error(train_y, np.exp(lin_reg.predict(train_X))))

    lin_test_RMSLE = np.sqrt(mean_squared_log_error(test_y, np.exp(lin_reg.predict(test_X))))

    print('Linear regresion (with log transformation) train RMSLE: ' + str(lin_train_RMSLE))

    print('Linear regresion (with log transformation) test RMSLE: ' + str(lin_test_RMSLE))

    print(' ')

    

    # logistic regression

    logistic_reg = LogisticRegression(max_iter=500).fit(train_X, train_y)

    logistic_train_RMSLE = np.sqrt(mean_squared_log_error(train_y, logistic_reg.predict(train_X)))

    logistic_test_RMSLE = np.sqrt(mean_squared_log_error(test_y, logistic_reg.predict(test_X)))

    print('Logistic regression train RMSLE: ' + str(logistic_train_RMSLE))

    print('Logistic regression test RMSLE: ' + str(logistic_test_RMSLE))

    print(' ')

    

    # tying the data sets together and return an output

    day = train_X.append(test_X)['days']

    actual = train_y.append(test_y)

    exp_predictions = pd.Series(np.exp(lin_reg.predict(train_X.append(test_X))))

    exp_predictions.index = day.index

    logistic_predictions = pd.Series(logistic_reg.predict(train_X.append(test_X)))

    logistic_predictions.index = day.index

    

    df = pd.DataFrame({'day':day,

                      'actual':actual,

                      'exp_predictions':exp_predictions,

                      'logistic_predictions': logistic_predictions})

    return df

    

US_train = wk1_train[wk1_train['Country/Region'] == 'US']



# set start data day before first case

start_date_index = (US_train['ConfirmedCases'] != 0).idxmax() - 27



# alternative method of setting start date

# start_date_index = (US_train['ConfirmedCases'] >= 100).idxmax() 

start_date = pd.Timestamp(US_train.loc[start_date_index, 'Date'])



US_start_date = start_date



US_train['days'] = (US_train['Date'].apply(pd.Timestamp) - start_date).dt.days

US_train = US_train.groupby('days').sum()[['ConfirmedCases']].reset_index()

US_train = US_train[US_train['days'] >= 0]



US_test = wk2_train[wk2_train['Country/Region'] == 'US']

US_test['days'] = (US_test['Date'].apply(pd.Timestamp) - start_date).dt.days

US_test = US_test.groupby('days').sum()[['ConfirmedCases']].reset_index()

US_test = US_test[US_test['days'] >= 0]



# subset for Mar 24, 25, 26

day_delta = (pd.Timestamp('2020-03-24') - start_date).days

day_delta_end = (pd.Timestamp('2020-03-26') - start_date).days

US_test = US_test[(US_test['days'] >= day_delta) & (US_test['days'] <= day_delta_end)] 



df = regressions(US_train[['days']], US_train['ConfirmedCases'], US_test[['days']], US_test['ConfirmedCases'])

df_melt = pd.melt(df, id_vars='day')

fig = px.line(df_melt, x='day', y='value', color='variable')

fig.show()
China_train = wk1_train[wk1_train['Country/Region'] == 'China']



# define start date (no offset needed since China data starts with a postive ConfirmedCase)

start_date = pd.Timestamp(wk1_train['Date'].min())



China_start_date = start_date



China_train['days'] = (China_train['Date'].apply(pd.Timestamp) - start_date).dt.days

China_train = China_train.groupby('days').sum()[['ConfirmedCases']].reset_index()

China_train = China_train[China_train['days'] >= 0]



China_test = wk2_train[wk2_train['Country/Region'] == 'China']

China_test['days'] = (China_test['Date'].apply(pd.Timestamp) - start_date).dt.days

China_test = China_test.groupby('days').sum()[['ConfirmedCases']].reset_index()

China_test = China_test[China_test['days'] >= 0]



# subset for Mar 24, 25, 26

day_delta = (pd.Timestamp('2020-03-24') - start_date).days

day_delta_end = (pd.Timestamp('2020-03-31') - start_date).days

China_test = China_test[(China_test['days'] >= day_delta) & (China_test['days'] <= day_delta_end)] 





df = regressions(China_train[['days']], China_train['ConfirmedCases'], China_test[['days']], China_test['ConfirmedCases'])

df_melt = pd.melt(df, id_vars='day')

fig = px.line(df_melt, x='day', y='value', color='variable')

fig.show()
# using previous algorithm to predict future (wk3)

data_dir = '/kaggle/input/covid19-global-forecasting-week-4'

wk4_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
# using the same start date as previous part (week 2)

start_date = US_start_date



US_test = wk4_train[wk4_train['Country_Region'] == 'US']

US_test['days'] = (US_test['Date'].apply(pd.Timestamp) - US_start_date).dt.days

US_test = US_test.groupby('days').sum()[['ConfirmedCases']].reset_index()

US_test = US_test[US_test['days'] >= 0]



# subset for Mar 24 - 31

day_delta = (pd.Timestamp('2020-03-24') - start_date).days

day_delta_end = (pd.Timestamp('2020-03-31') - start_date).days

US_test = US_test[(US_test['days'] >= day_delta) & (US_test['days'] <= day_delta_end)] 



# get results and graph

df = regressions(US_train[['days']], US_train['ConfirmedCases'], US_test[['days']], US_test['ConfirmedCases'])

df_melt = pd.melt(df, id_vars='day')

fig = px.line(df_melt, x='day', y='value', color='variable')

fig.show()
# using the same start date as previous part (week 2)

start_date = China_start_date



China_test = wk4_train[wk4_train['Country_Region'] == 'China']

China_test['days'] = (China_test['Date'].apply(pd.Timestamp) - start_date).dt.days

China_test = China_test.groupby('days').sum()[['ConfirmedCases']].reset_index()

China_test = China_test[China_test['days'] >= 0]



# subset for Mar 24 - 31

day_delta = (pd.Timestamp('2020-03-24') - start_date).days

day_delta_end = (pd.Timestamp('2020-03-31') - start_date).days

China_test = China_test[(China_test['days'] >= day_delta) & (China_test['days'] <= day_delta_end)] 



df = regressions(China_train[['days']], China_train['ConfirmedCases'], China_test[['days']], China_test['ConfirmedCases'])

df_melt = pd.melt(df, id_vars='day')

fig = px.line(df_melt, x='day', y='value', color='variable')

fig.show()