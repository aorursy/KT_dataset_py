import numpy as np

import pandas as pd
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'])

test  = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv',  parse_dates=['Date'])



train.rename(columns={

        'Id': 'id',

        'Date': 'date',

        'Province/State':'state',

        'Country/Region':'country',

        'Lat':'lat',

        'Long': 'long',

        'ConfirmedCases': 'confirmed',

        'Fatalities':'deaths',

        }, inplace=True)



test.rename(columns={

        'ForecastId': 'id',

        'Date': 'date',

        'Province/State':'state',

        'Country/Region':'country',

        'Lat':'lat',

        'Long': 'long',

        }, inplace=True)



valid = train[train['date'] >= test['date'].min()]

train = train[(train['date'] < test['date'].min()) & (train['date'] > pd.Timestamp('2020-03-01'))]



train['date'] = (train['date'] - pd.Timestamp('2020-03-01')).dt.days

valid['date'] = (valid['date'] - pd.Timestamp('2020-03-01')).dt.days

test['date']  = (test['date'] - pd.Timestamp('2020-03-01')).dt.days



train['lat-long'] = train['lat'].astype(str) + '-' + train['long'].astype(str)

valid['lat-long'] = valid['lat'].astype(str) + '-' + valid['long'].astype(str)

test['lat-long'] = test['lat'].astype(str) + '-' + test['long'].astype(str)
all_coords = train['lat-long'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0



for coords in all_coords:

    

    X_train_ = train[train['lat-long']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['lat-long']==coords]['confirmed']#.values.reshape(-1,1)

    

    X_valid_ = valid[valid['lat-long']==coords]['date']#.values.reshape(-1,1)

    y_valid_ = valid[valid['lat-long']==coords]['confirmed']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_valid_)+1,1)

    

    z = np.polyfit(X_train_.values, y_train_.values, 3)

    pf = np.poly1d(z)

        

    y_preds_ = np.round(X_valid_.apply(pf)).clip(lower=y_linear)

    

    predictions[coords] = y_preds_

    RMSE[coords]=np.sqrt(np.sum(np.square(y_preds_-y_valid_)))

    total_RMSE += np.sqrt(np.sum(np.square(y_preds_-y_valid_)))



    

print(total_RMSE)
all_coords = train['lat-long'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0



for coords in all_coords:

    

    X_train_ = train[train['lat-long']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['lat-long']==coords]['deaths']#.values.reshape(-1,1)

    

    X_valid_ = valid[valid['lat-long']==coords]['date']#.values.reshape(-1,1)

    y_valid_ = valid[valid['lat-long']==coords]['deaths']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_valid_)+1,1)

    

    z = np.polyfit(X_train_.values, y_train_.values, 3)

    pf = np.poly1d(z)

    

    y_preds_ = np.round(X_valid_.apply(pf)).clip(lower=y_linear)

    predictions[coords] = y_preds_

    RMSE[coords]=np.sqrt(np.sum(np.square(y_preds_-y_valid_)))

    total_RMSE += np.sqrt(np.sum(np.square(y_preds_-y_valid_)))



    

print(total_RMSE)
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'])

test  = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv',  parse_dates=['Date'])



train.rename(columns={

        'Id': 'id',

        'Date': 'date',

        'Province/State':'state',

        'Country/Region':'country',

        'Lat':'lat',

        'Long': 'long',

        'ConfirmedCases': 'confirmed',

        'Fatalities':'deaths',

        }, inplace=True)



test.rename(columns={

        'ForecastId': 'id',

        'Date': 'date',

        'Province/State':'state',

        'Country/Region':'country',

        'Lat':'lat',

        'Long': 'long',

        }, inplace=True)



train = train[train['date'] > pd.Timestamp('2020-03-01')]



train['date'] = (train['date'] - pd.Timestamp('2020-03-01')).dt.days

test['date']  = (test['date'] - pd.Timestamp('2020-03-01')).dt.days



train['lat-long'] = train['lat'].astype(str) + '-' + train['long'].astype(str)

test['lat-long'] = test['lat'].astype(str) + '-' + test['long'].astype(str)
submission = pd.DataFrame()

submission['lat-long'] = test['lat-long']

submission.reset_index(inplace=True)



submission['ConfirmedCases'] = 0

submission['Fatalities'] = 0
all_coords = train['lat-long'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0



for coords in all_coords:

    

    X_train_ = train[train['lat-long']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['lat-long']==coords]['confirmed']#.values.reshape(-1,1)

    

    X_test_ = test[test['lat-long']==coords]['date']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_test_)+1,1)

    

    z = np.polyfit(X_train_.values, y_train_.values, 3)

    pf = np.poly1d(z)

    

    y_preds_ = np.round(X_test_.apply(pf)).clip(lower=y_linear)

        

    submission.loc[submission['lat-long']==coords,'ConfirmedCases'] = y_preds_
all_coords = train['lat-long'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0



for coords in all_coords:

    

    X_train_ = train[train['lat-long']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['lat-long']==coords]['deaths']#.values.reshape(-1,1)

    

    X_test_ = test[test['lat-long']==coords]['date']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_test_)+1,1)

    

    z = np.polyfit(X_train_.values, y_train_.values, 3)

    pf = np.poly1d(z)

    

    y_preds_ = np.round(X_test_.apply(pf)).clip(lower=y_linear)

    

    submission.loc[submission['lat-long']==coords,'Fatalities'] = y_preds_
submission.drop('lat-long', axis=1, inplace=True)

submission['index'] = submission['index'] + 1

submission.rename(columns={

    'index' : 'ForecastId'}, inplace=True)
submission
submission.to_csv('submission.csv', index=False)