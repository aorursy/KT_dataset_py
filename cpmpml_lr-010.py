# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge



import datetime

import gc

from tqdm import tqdm
def get_cpmp_sub(save_oof=False, save_public_test=False):

    train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

    train['Province_State'].fillna('', inplace=True)

    train['Date'] = pd.to_datetime(train['Date'])

    train['day'] = train.Date.dt.dayofyear

    #train = train[train.day <= 85]

    train['geo'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]

    train



    test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

    test['Province_State'].fillna('', inplace=True)

    test['Date'] = pd.to_datetime(test['Date'])

    test['day'] = test.Date.dt.dayofyear

    test['geo'] = ['_'.join(x) for x in zip(test['Country_Region'], test['Province_State'])]

    test



    day_min = train['day'].min()

    train['day'] -= day_min

    test['day'] -= day_min



    min_test_val_day = test.day.min()

    max_test_val_day = train.day.max()

    max_test_day = test.day.max()

    num_days = max_test_day + 1



    min_test_val_day, max_test_val_day, num_days



    train['ForecastId'] = -1

    test['Id'] = -1

    test['ConfirmedCases'] = 0

    test['Fatalities'] = 0



    debug = False



    data = pd.concat([train,

                      test[test.day > max_test_val_day][train.columns]

                     ]).reset_index(drop=True)

    if debug:

        data = data[data['geo'] >= 'France_'].reset_index(drop=True)

    #del train, test

    gc.collect()



    dates = data[data['geo'] == 'France_'].Date.values



    if 0:

        gr = data.groupby('geo')

        data['ConfirmedCases'] = gr.ConfirmedCases.transform('cummax')

        data['Fatalities'] = gr.Fatalities.transform('cummax')



    geo_data = data.pivot(index='geo', columns='day', values='ForecastId')

    num_geo = geo_data.shape[0]

    geo_data



    geo_id = {}

    for i,g in enumerate(geo_data.index):

        geo_id[g] = i





    ConfirmedCases = data.pivot(index='geo', columns='day', values='ConfirmedCases')

    Fatalities = data.pivot(index='geo', columns='day', values='Fatalities')



    if debug:

        cases = ConfirmedCases.values

        deaths = Fatalities.values

    else:

        cases = np.log1p(ConfirmedCases.values)

        deaths = np.log1p(Fatalities.values)





    def get_dataset(start_pred, num_train, lag_period):

        days = np.arange( start_pred - num_train + 1, start_pred + 1)

        lag_cases = np.vstack([cases[:, d - lag_period : d] for d in days])

        lag_deaths = np.vstack([deaths[:, d - lag_period : d] for d in days])

        target_cases = np.vstack([cases[:, d : d + 1] for d in days])

        target_deaths = np.vstack([deaths[:, d : d + 1] for d in days])

        geo_ids = np.vstack([geo_ids_base for d in days])

        country_ids = np.vstack([country_ids_base for d in days])

        return lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days



    def update_valid_dataset(data, pred_death, pred_case):

        lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days = data

        day = days[-1] + 1

        new_lag_cases = np.hstack([lag_cases[:, 1:], pred_case])

        new_lag_deaths = np.hstack([lag_deaths[:, 1:], pred_death]) 

        new_target_cases = cases[:, day:day+1]

        new_target_deaths = deaths[:, day:day+1] 

        new_geo_ids = geo_ids  

        new_country_ids = country_ids  

        new_days = 1 + days

        return new_lag_cases, new_lag_deaths, new_target_cases, new_target_deaths, new_geo_ids, new_country_ids, new_days



    def fit_eval(lr_death, lr_case, data, start_lag_death, end_lag_death, num_lag_case, fit, score):

        lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days = data



        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], country_ids])

        X_death = np.hstack([lag_deaths[:, -num_lag_case:], country_ids])

        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], lag_deaths[:, -num_lag_case:], country_ids])

        y_death = target_deaths

        y_death_prev = lag_deaths[:, -1:]

        if fit:

            if 0:

                keep = (y_death > 0).ravel()

                X_death = X_death[keep]

                y_death = y_death[keep]

                y_death_prev = y_death_prev[keep]

            lr_death.fit(X_death, y_death)

        y_pred_death = lr_death.predict(X_death)

        y_pred_death = np.maximum(y_pred_death, y_death_prev)



        X_case = np.hstack([lag_cases[:, -num_lag_case:], geo_ids])

        X_case = lag_cases[:, -num_lag_case:]

        y_case = target_cases

        y_case_prev = lag_cases[:, -1:]

        if fit:

            lr_case.fit(X_case, y_case)

        y_pred_case = lr_case.predict(X_case)

        y_pred_case = np.maximum(y_pred_case, y_case_prev)



        if score:

            death_score = val_score(y_death, y_pred_death)

            case_score = val_score(y_case, y_pred_case)

        else:

            death_score = 0

            case_score = 0



        return death_score, case_score, y_pred_death, y_pred_case



    def train_model(train, valid, start_lag_death, end_lag_death, num_lag_case, num_val, score=True):

        alpha = 3

        lr_death = Ridge(alpha=alpha, fit_intercept=False)

        lr_case = Ridge(alpha=alpha, fit_intercept=True)



        (train_death_score, train_case_score, train_pred_death, train_pred_case,

        ) = fit_eval(lr_death, lr_case, train, start_lag_death, end_lag_death, num_lag_case, fit=True, score=score)



        death_scores = []

        case_scores = []



        death_pred = []

        case_pred = []



        for i in range(num_val):



            (valid_death_score, valid_case_score, valid_pred_death, valid_pred_case,

            ) = fit_eval(lr_death, lr_case, valid, start_lag_death, end_lag_death, num_lag_case, fit=False, score=score)



            death_scores.append(valid_death_score)

            case_scores.append(valid_case_score)

            death_pred.append(valid_pred_death)

            case_pred.append(valid_pred_case)



            if 0:

                print('val death: %0.3f' %  valid_death_score,

                      'val case: %0.3f' %  valid_case_score,

                      'val : %0.3f' %  np.mean([valid_death_score, valid_case_score]),

                      flush=True)

            valid = update_valid_dataset(valid, valid_pred_death, valid_pred_case)



        if score:

            death_scores = np.sqrt(np.mean([s**2 for s in death_scores]))

            case_scores = np.sqrt(np.mean([s**2 for s in case_scores]))

            if 0:

                print('train death: %0.3f' %  train_death_score,

                      'train case: %0.3f' %  train_case_score,

                      'val death: %0.3f' %  death_scores,

                      'val case: %0.3f' %  case_scores,

                      'val : %0.3f' % ( (death_scores + case_scores) / 2),

                      flush=True)

            else:

                print('%0.4f' %  case_scores,

                      ', %0.4f' %  death_scores,

                      '= %0.4f' % ( (death_scores + case_scores) / 2),

                      flush=True)

        death_pred = np.hstack(death_pred)

        case_pred = np.hstack(case_pred)

        return death_scores, case_scores, death_pred, case_pred



    countries = [g.split('_')[0] for g in geo_data.index]

    countries = pd.factorize(countries)[0]



    country_ids_base = countries.reshape((-1, 1))

    ohe = OneHotEncoder(sparse=False)

    country_ids_base = 0.2 * ohe.fit_transform(country_ids_base)

    country_ids_base.shape



    geo_ids_base = np.arange(num_geo).reshape((-1, 1))

    ohe = OneHotEncoder(sparse=False)

    geo_ids_base = 0.1 * ohe.fit_transform(geo_ids_base)

    geo_ids_base.shape



    def val_score(true, pred):

        pred = np.log1p(np.round(np.expm1(pred) - 0.2))

        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))



    def val_score(true, pred):

        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))







    start_lag_death, end_lag_death = 14, 6,

    num_train = 5

    num_lag_case = 14

    lag_period = max(start_lag_death, num_lag_case)



    def get_oof(start_val_delta=0):   

        start_val = min_test_val_day + start_val_delta

        last_train = start_val - 1

        num_val = max_test_val_day - start_val + 1

        print(dates[start_val], start_val, num_val)

        train_data = get_dataset(last_train, num_train, lag_period)

        valid_data = get_dataset(start_val, 1, lag_period)

        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 

                                                            start_lag_death, end_lag_death, num_lag_case, num_val)



        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()

        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)

        pred_deaths = pred_deaths.stack().reset_index()

        pred_deaths.columns = ['geo', 'day', 'Fatalities']

        pred_deaths



        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()

        pred_cases.iloc[:, :] = np.expm1(val_case_preds)

        pred_cases = pred_cases.stack().reset_index()

        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']

        pred_cases



        sub = train[['Date', 'Id', 'geo', 'day']]

        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])

        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])

        #sub = sub.fillna(0)

        sub = sub[sub.day >= start_val]

        sub = sub[['Id', 'ConfirmedCases', 'Fatalities']].copy()

        return sub





    if save_oof:

        for start_val_delta, date in zip(range(3, -8, -3),

                                  ['2020-03-22', '2020-03-19', '2020-03-16', '2020-03-13']):

            print(date, end=' ')

            oof = get_oof(start_val_delta)

            oof.to_csv('../submissions/cpmp-%s.csv' % date, index=None)



    def get_sub(start_val_delta=0):   

        start_val = min_test_val_day + start_val_delta

        last_train = start_val - 1

        num_val = max_test_val_day - start_val + 1

        print(dates[last_train], start_val, num_val)

        train_data = get_dataset(last_train, num_train, lag_period)

        valid_data = get_dataset(start_val, 1, lag_period)

        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 

                                                            start_lag_death, end_lag_death, num_lag_case, num_val)



        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()

        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)

        pred_deaths = pred_deaths.stack().reset_index()

        pred_deaths.columns = ['geo', 'day', 'Fatalities']

        pred_deaths



        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()

        pred_cases.iloc[:, :] = np.expm1(val_case_preds)

        pred_cases = pred_cases.stack().reset_index()

        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']

        pred_cases



        sub = test[['Date', 'ForecastId', 'geo', 'day']]

        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])

        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])

        sub = sub.fillna(0)

        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]

        return sub

        return sub





    known_test = train[['geo', 'day', 'ConfirmedCases', 'Fatalities']

              ].merge(test[['geo', 'day', 'ForecastId']], how='left', on=['geo', 'day'])

    known_test = known_test[['ForecastId', 'ConfirmedCases', 'Fatalities']][known_test.ForecastId.notnull()].copy()

    known_test



    unknow_test = test[test.day > max_test_val_day]

    unknow_test



    def get_final_sub():   

        start_val = max_test_val_day + 1

        last_train = start_val - 1

        num_val = max_test_day - start_val + 1

        print(dates[last_train], start_val, num_val)

        train_data = get_dataset(last_train, num_train, lag_period)

        valid_data = get_dataset(start_val, 1, lag_period)

        (_, _, val_death_preds, val_case_preds

        ) = train_model(train_data, valid_data, start_lag_death, end_lag_death, num_lag_case, num_val, score=False)



        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()

        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)

        pred_deaths = pred_deaths.stack().reset_index()

        pred_deaths.columns = ['geo', 'day', 'Fatalities']

        pred_deaths



        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()

        pred_cases.iloc[:, :] = np.expm1(val_case_preds)

        pred_cases = pred_cases.stack().reset_index()

        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']

        pred_cases

        print(unknow_test.shape, pred_deaths.shape, pred_cases.shape)



        sub = unknow_test[['Date', 'ForecastId', 'geo', 'day']]

        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])

        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])

        #sub = sub.fillna(0)

        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]

        sub = pd.concat([known_test, sub])

        return sub



    if save_public_test:

        sub = get_sub()

    else:

        sub = get_final_sub()

    return sub
sub = get_cpmp_sub()

sub['ForecastId'] = sub['ForecastId'].astype('int')



sub
sub.to_csv('submission.csv', index=False)