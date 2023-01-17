import numpy as np

import pandas as pd
train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv', parse_dates=['Date'])

test  = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv',  parse_dates=['Date'])



train.rename(columns={

        'Id': 'id',

        'Date': 'date',

        'Province_State':'state',

        'Country_Region':'country',

        'ConfirmedCases': 'confirmed',

        'Fatalities':'deaths',

        }, inplace=True)



test.rename(columns={

        'ForecastId': 'id',

        'Date': 'date',

        'Province_State':'state',

        'Country_Region':'country',

        }, inplace=True)



valid = train[train['date'] >= test['date'].min()]

train = train[(train['date'] < test['date'].min()) & (train['date'] > pd.Timestamp('2020-03-01'))]



train['date'] = (train['date'] - pd.Timestamp('2020-03-01')).dt.days

valid['date'] = (valid['date'] - pd.Timestamp('2020-03-01')).dt.days

test['date']  = (test['date'] - pd.Timestamp('2020-03-01')).dt.days



train['loc'] = train['country'].astype(str) + '-' + train['state'].astype(str)

valid['loc'] = valid['country'].astype(str) + '-' + valid['state'].astype(str)

test['loc'] = test['country'].astype(str) + '-' + test['state'].astype(str)
all_coords = train['loc'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0



for coords in all_coords:

    

    X_train_ = train[train['loc']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['loc']==coords]['confirmed']#.values.reshape(-1,1)

    

    X_valid_ = valid[valid['loc']==coords]['date']#.values.reshape(-1,1)

    y_valid_ = valid[valid['loc']==coords]['confirmed']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_valid_)+1,1)

    

    z = np.polyfit(X_train_.values, y_train_.values, 3)

    pf = np.poly1d(z)

        

    y_preds_ = np.round(X_valid_.apply(pf)).clip(lower=y_linear)

    

    predictions[coords] = y_preds_

    RMSE[coords]=np.sqrt(np.sum(np.square(y_preds_-y_valid_)))

    total_RMSE += np.sqrt(np.sum(np.square(y_preds_-y_valid_)))



    

print(total_RMSE)
all_coords = train['loc'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0



for coords in all_coords:

    

    X_train_ = train[train['loc']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['loc']==coords]['deaths']#.values.reshape(-1,1)

    

    X_valid_ = valid[valid['loc']==coords]['date']#.values.reshape(-1,1)

    y_valid_ = valid[valid['loc']==coords]['deaths']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_valid_)+1,1)

    

    z = np.polyfit(X_train_.values, y_train_.values, 3)

    pf = np.poly1d(z)

    

    y_preds_ = np.round(X_valid_.apply(pf)).clip(lower=y_linear)

    predictions[coords] = y_preds_

    RMSE[coords]=np.sqrt(np.sum(np.square(y_preds_-y_valid_)))

    total_RMSE += np.sqrt(np.sum(np.square(y_preds_-y_valid_)))



    

print(total_RMSE)
train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv', parse_dates=['Date'])

test  = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv',  parse_dates=['Date'])



train.rename(columns={

        'Id': 'id',

        'Date': 'date',

        'Province_State':'state',

        'Country_Region':'country',

        'ConfirmedCases': 'confirmed',

        'Fatalities':'deaths',

        }, inplace=True)



test.rename(columns={

        'ForecastId': 'id',

        'Date': 'date',

        'Province_State':'state',

        'Country_Region':'country',

        }, inplace=True)



train = train[train['date'] > pd.Timestamp('2020-03-01')]



train['date'] = (train['date'] - pd.Timestamp('2020-03-01')).dt.days

test['date']  = (test['date'] - pd.Timestamp('2020-03-01')).dt.days



train['loc'] = train['country'].astype(str) + '-' + train['state'].astype(str)

test['loc'] = test['country'].astype(str) + '-' + test['state'].astype(str)
submission = pd.DataFrame()

submission['loc'] = test['loc']

submission.reset_index(inplace=True)



submission['ConfirmedCases'] = 0

submission['Fatalities'] = 0
all_coords = train['loc'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0



for coords in all_coords:

    

    X_train_ = train[train['loc']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['loc']==coords]['confirmed']#.values.reshape(-1,1)

    

    X_test_ = test[test['loc']==coords]['date']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_test_)+1,1)

    

    z = np.polyfit(X_train_.values, y_train_.values, 3)

    pf = np.poly1d(z)

    

    y_preds_ = np.round(X_test_.apply(pf)).clip(lower=y_linear)

        

    submission.loc[submission['loc']==coords,'ConfirmedCases'] = y_preds_
all_coords = train['loc'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0



for coords in all_coords:

    

    X_train_ = train[train['loc']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['loc']==coords]['deaths']#.values.reshape(-1,1)

    

    X_test_ = test[test['loc']==coords]['date']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_test_)+1,1)

    

    z = np.polyfit(X_train_.values, y_train_.values, 3)

    pf = np.poly1d(z)

    

    y_preds_ = np.round(X_test_.apply(pf)).clip(lower=y_linear)

    

    submission.loc[submission['loc']==coords,'Fatalities'] = y_preds_
submission.drop('loc', axis=1, inplace=True)

submission['index'] = submission['index'] + 1

submission.rename(columns={

    'index' : 'ForecastId'}, inplace=True)
submission
submission.to_csv('submission.csv', index=False)