#Libraries to be imported

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import warnings

from sklearn.utils import check_array 

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



%matplotlib inline

warnings.filterwarnings("ignore")

init_notebook_mode(connected=True)
df_sample_sub = pd.read_csv("../input/ltfs-2020/sample_submission_IIzFVsf.csv")

df_test = pd.read_csv("../input/ltfs-2020/test_1eLl9Yf.csv")

df_train = pd.read_csv("../input/ltfs-2020/train_fwYjLYX.csv")
display(df_train.info())

display(df_train.head())

display(df_train.describe())
print('Minimum date from training set: {}'.format(pd.to_datetime(df_train.application_date.min()).date()))

print('Maximum date from training set: {}'.format(pd.to_datetime(df_train.application_date.max()).date()))
max_date_train = pd.to_datetime(df_train.application_date.max()).date()

max_date_test = pd.to_datetime(df_test.application_date.max()).date()

lag_size = (max_date_test - max_date_train).days

print('Maximum date from training set: {}'.format(max_date_train))

print('Maximum date from test set: {}'.format(max_date_test))

print('Forecast Lag: {}'.format(lag_size))
daily_cases_1 = df_train[df_train['segment'] == 1].groupby(['branch_id','state','zone','application_date'], as_index = False)['case_count'].sum()

daily_cases_2 = df_train[df_train['segment'] == 2].groupby(['state','application_date'], as_index = False)['case_count'].sum()
daily_cases_1_sc = []

for state in daily_cases_1['state'].unique():

    current_daily_cases_1 = daily_cases_1[daily_cases_1['state'] == state]

    daily_cases_1_sc.append(go.Scatter(x=current_daily_cases_1['application_date'], y=current_daily_cases_1['case_count'], name=('%s' % state)))



layout = go.Layout(title='Daily Case Count - Segment 1', xaxis=dict(title='Date'), yaxis=dict(title='Case Count'))

fig = go.Figure(data=daily_cases_1_sc, layout=layout)

iplot(fig)
daily_cases_2_sc = []

for state in daily_cases_2['state'].unique():

    current_daily_cases_2 = daily_cases_2[daily_cases_2['state'] == state]

    daily_cases_2_sc.append(go.Scatter(x=current_daily_cases_2['application_date'], y=current_daily_cases_2['case_count'], name=('%s' % state)))



layout = go.Layout(title='Daily Case Count - Segment 2', xaxis=dict(title='Date'), yaxis=dict(title='Case Count'))

fig = go.Figure(data=daily_cases_2_sc, layout=layout)

iplot(fig)
df_train['application_date'] = pd.to_datetime(df_train['application_date'])

df_train = df_train.sort_values('application_date').groupby(['application_date','segment'], as_index=False)

df_train = df_train.agg({'case_count':['sum']})

df_train.columns = ['application_date','segment', 'case_count']

df_train.head()
def series_to_supervised(data, window=1, lag=1, dropnan = True):

    cols, names = list(), list()

    #Input Sequence (t-n, ... t-1)

    for i in range(window-1,0,-1):

        cols.append(data.shift(i))

        names+=[('%s(t-%d)' % (col,i)) for col in data.columns]

    #Current Timestamp (t=0)

    cols.append(data)

    names+=[('%s(t)' % (col)) for col in data.columns]

    #Target Timestamp (t=lag)

    cols.append(data.shift(-lag))

    names+=[('%s(t+%d)' %  (col,lag)) for col in data.columns]

    #Put it all together

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    # Drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg
df_train_1 = df_train[df_train['segment'] == 1] #Segment 1 records

df_train_2 = df_train[df_train['segment'] == 2] #Segment 2 records
window = 30 #use the last 30 days

lag = 1 #predict the next day

series = series_to_supervised(df_train_1.drop(['application_date','segment'], axis=1), window=window, lag=lag)

series.head()
df_train_1.case_count[:31]
def train_test_split(data, n_test):

    return data[:-n_test], data[-n_test:]
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def model_fit(train, config):

    # unpack config

    n_input, n_nodes, n_epochs, n_batch = config

    df = series_to_supervised(train, window=n_input)

    data = df.to_numpy()

    train_x, train_y = data[:, :-1], data[:, -1]

    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))

    # define model

    model = Sequential()

    model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, 1)))

    model.add(Dense(n_nodes, activation='relu'))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    # fit

    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)

    return model
def model_predict(model, history, config):

    # unpack config

    window, _, _, _ = config

    x_input = np.array(history[-window:]).reshape((1, window, 1))

    # forecast

    yhat = model.predict(x_input, verbose=0)

    return yhat[0]
#walk-forward validation for univariate data

def walk_forward_validation(data, n_test, cfg):

    predictions = list()

    # split dataset

    train, test = train_test_split(data, n_test)

    # fit model

    model = model_fit(train, cfg)

    # seed history with training dataset

    history = [x for x in train.to_numpy()]

    test = test.to_numpy()

    # step over each time-step in the test set

    for i in range(len(test)):

        # fit model and make forecast for history

        yhat = model_predict(model, history, cfg)

        # store forecast in list of predictions

        predictions.append(yhat)

        # add actual observation to history for the next loop

        history.append(test[i])

    # estimate prediction error

    error = mean_absolute_percentage_error(test, predictions)

    print(' > %.3f' % error)

    return error
# repeat evaluation of a config

def repeat_evaluate(data, config, n_test, n_repeats=30):

    # fit and evaluate the model n times

    scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]

    return scores
# summarize model performance

def summarize_scores(name, scores):

    # print a summary

    scores_m, score_std = np.mean(scores), np.std(scores)

    print('%s: %.3f MAPE (+/- %.3f)' % (name, scores_m, score_std))

    # box and whisker plot

    plt.boxplot(scores)

    plt.show()
print('Date Range for Segment 1: {} days'.format((df_train_1.application_date.max() - df_train_1.application_date.min()).days))

print('Date Range for Segment 2: {} days'.format((df_train_2.application_date.max() - df_train_2.application_date.min()).days))
config = [30, 50, 100, 100]
#Training segment 1 data

n_test = 225

df_1 = df_train_1.drop(['segment','application_date'], axis=1)

scores = repeat_evaluate(df_1, config, n_test)

summarize_scores('LSTM', scores)
#Training segment 2 data

n_test = 243

df_2 = df_train_2.drop(['segment','application_date'], axis=1)

scores = repeat_evaluate(df_2, config, n_test)

summarize_scores('LSTM', scores)
def train_model(data, config):

    n_input, n_nodes, n_epochs, n_batch = config

    df = series_to_supervised(data, window=n_input)

    data = df.to_numpy()

    train_x, train_y = data[:, :-1], data[:, -1]

    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))

    # define model

    model = Sequential()

    model.add(LSTM(n_nodes, activation='relu', input_shape=(window, 1)))

    model.add(Dense(n_nodes, activation='relu'))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    # fit

    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)

    return model
def time_series_forecasting(train, test, segment, config):

    #Drop the unwanted columns

    df = train.drop(['segment','application_date'], axis=1)

    #Get the window

    window = config[0]

    #Train the model

    model = train_model(df, config)

    #Define the history to be taken into consideration for prediction

    history = [x for x in df.to_numpy()]

    #Get the records of the specified segment

    test_seg = test[test['segment'] == segment]

    #Define an empty case_count column to be inserted later on

    cases = pd.Series([])

    #One by one do prediction and append it to history and the series.

    for i in range(test.shape[0]):

        x_input = np.array(history[-window:]).reshape((1, window, 1))

        y = model.predict(x_input, verbose=0)

        history.append(y[0])

        cases[i] = round((y[0][0]),0) #Since number of cases are supposed to be integer

    #Add the calculated column to the dataset

    test_seg.insert(loc=3, column='case_count', value=cases)

    return test_seg
test1 = time_series_forecasting(df_train_1, df_test, 1, config)

test2 = time_series_forecasting(df_train_2, df_test, 2, config)
submit = pd.concat([test1, test2], ignore_index=True)
submit.to_csv('csv_to_submit.csv', index = False)