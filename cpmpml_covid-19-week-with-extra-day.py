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

pd.options.display.max_rows = 500

pd.options.display.max_columns = 500



import matplotlib.pyplot as plt

%matplotlib inline



from scipy.signal import savgol_filter

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.linear_model import LinearRegression, Ridge



import datetime

import gc

from tqdm import tqdm
train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

us_before = pd.read_csv('../input/jhu-covid19-data-with-us-state-data-prior-to-mar-9/covid19_train_data_us_states_before_march_09_new.csv')

update = (train['Country_Region'] == 'US') & (train.Date <= '2020-03-09')

df = train[update]

us_before = us_before[(us_before['Country.Region'] == 'US') & (us_before.Date <= '2020-03-09')]

us_before = us_before[['Province.State', 'Country.Region', 'Date', 'ConfirmedCases', 'Fatalities']]

us_before.columns = ['Province_State', 'Country_Region', 'Date', 'ConfirmedCases_new', 'Fatalities_new']

df.shape, us_before.shape

df = df.merge(us_before, how='left', on=['Province_State', 'Country_Region', 'Date']).fillna(0)

train.loc[update, 'ConfirmedCases'] = df['ConfirmedCases_new'].values

train.loc[update, 'Fatalities'] = df['Fatalities_new'].values



train['Province_State'].fillna('', inplace=True)

train['Date'] = pd.to_datetime(train['Date'])

train['day'] = train.Date.dt.dayofyear

#train = train[train.day <= 85]

train['geo'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]

# keep one extra dasy only

train = train[train.Date <= '2020-04-15'].copy()

train


test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

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



data = pd.concat([train,

                  test[test.day > max_test_val_day][train.columns]

                 ]).reset_index(drop=True)





dates = data[data['geo'] == 'France_'].Date.values



region_meta = pd.read_csv('../input/covid19-forecasting-metadata/region_metadata.csv')

region_meta['Province_State'].fillna('', inplace=True)

region_meta['geo'] = ['_'.join(x) for x in zip(region_meta['Country_Region'], region_meta['Province_State'], )]

region_meta



set(data.geo.unique()) - set(region_meta.geo.unique())



region_meta = data[['geo']].merge(region_meta, how='left', on='geo') #.fillna(0)

region_meta = region_meta.groupby('geo').first()





population = np.log1p(region_meta[['population']])

#_ = plt.hist(population.population, bins=100)



#population = population.pivot(index='geo', columns='day', values='population').values

population = population[['population']].values

population.shape



lockdown_date = pd.read_csv('../input/covid19-lockdown-dates-by-country/countryLockdowndates.csv')

lockdown_date['Province'].fillna('', inplace=True)

lockdown_date['Date'] = pd.to_datetime(lockdown_date['Date'], dayfirst=True)

lockdown_date['lock_day'] = lockdown_date.Date.dt.dayofyear

lockdown_date['lock_day'] -= day_min

lockdown_date['geo'] = ['_'.join(x) for x in zip(lockdown_date['Country/Region'], lockdown_date['Province'])]

lockdown_date



lockdown_date = data[['geo', 'day']].merge(lockdown_date[['geo', 'lock_day']], how='left', on=['geo'])

lockdown_date['locked'] = 1 * (lockdown_date['day'] > lockdown_date['lock_day'])

lockdown_date = lockdown_date.pivot(index='geo', columns='day', values='locked').values

lockdown_date



geo_data = data.pivot(index='geo', columns='day', values='ForecastId')

num_geo = geo_data.shape[0]

geo_data



geo_id = {}

for i,g in enumerate(geo_data.index):

    geo_id[g] = i

num_geo



ConfirmedCases = data.pivot(index='geo', columns='day', values='ConfirmedCases')

Fatalities = data.pivot(index='geo', columns='day', values='Fatalities')



cases = np.log1p(ConfirmedCases.values)

deaths = np.log1p(Fatalities.values)



def get_dataset(start_pred, num_train, lag_period, continents_ids_base, country_ids_base, cases, deaths, 

                 population, time_cases, time_deaths, lockdown_date):

    days = np.arange( start_pred - num_train + 1, start_pred + 1)

    lag_cases = np.vstack([cases[:, d - lag_period : d] for d in days])

    lag_deaths = np.vstack([deaths[:, d - lag_period : d] for d in days])

    target_cases = np.vstack([cases[:, d : d + 1] for d in days])

    target_deaths = np.vstack([deaths[:, d : d + 1] for d in days])

    continents_ids = np.vstack([continents_ids_base for d in days])

    country_ids = np.vstack([country_ids_base for d in days])

    population = np.vstack([population for d in days])

    time_case = np.vstack([time_cases[:, d - 1: d ] for d in days])

    time_death = np.vstack([time_deaths[:, d - 1 : d ] for d in days])

    lockdown = [get_lockdown(lockdown_date, d) for d in days]

    lockdown_case = np.vstack([l[0] for l in lockdown])

    lockdown_death = np.vstack([l[1] for l in lockdown])

    start_pred = np.hstack([d * np.ones((cases.shape[0], )) for d in days]).astype('int')

    return (lag_cases, lag_deaths, target_cases, target_deaths, 

            continents_ids, country_ids, population, time_case, time_death, 

            lockdown_case, lockdown_death, start_pred, days)



def update_valid_dataset(data, pred_death, pred_case, cases, deaths):

    (lag_cases, lag_deaths, target_cases, target_deaths, 

     continents_ids, country_ids, population, time_case, time_death, 

     lockdown_case, lockdown_death, start_pred, days) = data

    day = days[-1] + 1

    new_lag_cases = np.hstack([lag_cases[:, 1:], pred_case])

    new_lag_deaths = np.hstack([lag_deaths[:, 1:], pred_death]) 

    new_target_cases = cases[:, day:day+1]

    new_target_deaths = deaths[:, day:day+1] 

    new_continents_ids = continents_ids  

    new_country_ids = country_ids  

    new_population = population  

    new_time_death, new_time_case = update_time(time_death, time_case, pred_death, pred_case)

    new_lockdown_case = lockdown_case

    new_lockdown_death = lockdown_death

    new_start_pred = 1 + start_pred

    new_days = 1 + days

    return (new_lag_cases, new_lag_deaths, new_target_cases, new_target_deaths, 

            new_continents_ids, new_country_ids, new_population, 

            new_time_case, new_time_death, new_lockdown_case, new_lockdown_death, 

            new_start_pred, new_days)



def train_model(train, valid, start_lag_death, end_lag_death, num_lag_case, num_val, score, cases, deaths,):

    alpha = 3

    lr_death = Ridge(alpha=alpha, fit_intercept=True)

    lr_case = Ridge(alpha=alpha, fit_intercept=False)

        

    (train_death_score, train_case_score, train_pred_death, train_pred_case,

    ) = fit_eval(lr_death, lr_case, train, start_lag_death, end_lag_death, num_lag_case, fit=True, score=score,

                 cases=cases, deaths=deaths)

    

    death_scores = []

    case_scores = []

    

    death_pred = []

    case_pred = []

    

    for i in range(num_val):



        (valid_death_score, valid_case_score, valid_pred_death, valid_pred_case,

        ) = fit_eval(lr_death, lr_case, valid, start_lag_death, end_lag_death, num_lag_case, 

                     fit=False, score=score,

                 cases=cases, deaths=deaths)

        death_scores.append(valid_death_score)

        case_scores.append(valid_case_score)

        death_pred.append(valid_pred_death)

        case_pred.append(valid_pred_case)

        

        if 0:

            print('val death: %0.3f' %  valid_death_score,

                  'val case: %0.3f' %  valid_case_score,

                  'val : %0.3f' %  np.mean([valid_death_score, valid_case_score]),

                  flush=True)

        valid = update_valid_dataset(valid, valid_pred_death, valid_pred_case, cases, deaths)

    

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



def get_country_ids(last_train, case_threshold):

    countries = [g.split('_')[0] for g in geo_data.index]

    countries = pd.factorize(countries)[0]

    countries[cases[:, :last_train+1].max(axis=1) < np.log1p(case_threshold)] = -1

    countries = pd.factorize(countries)[0]

    



    country_ids_base = countries.reshape((-1, 1))

    ohe = OneHotEncoder(sparse=False)

    country_ids_base = 0.2 * ohe.fit_transform(country_ids_base)

    return country_ids_base



def get_continent_ids():

    continents = region_meta['continent']

    continents = pd.factorize(continents)[0]

    continents_ids_base = continents.reshape((-1, 1))

    ohe = OneHotEncoder(sparse=False)

    continents_ids_base = ohe.fit_transform(continents_ids_base)

    return continents_ids_base



def val_score(true, pred):

    pred = np.log1p(np.round(np.expm1(pred) - 0.2))

    return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))



def val_score(true, pred):

    return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))







def fit_eval(lr_death, lr_case, data, start_lag_death, end_lag_death, num_lag_case, fit, score, cases, deaths):

    (lag_cases, lag_deaths, target_cases, target_deaths, 

     continents_ids, country_ids, population, time_case, time_death, 

     lockdown_case, lockdown_death, start_pred, days)  = data

    idx = np.arange(lag_cases.shape[0])

    X_trend = predict(lag_deaths, num_lag_death_trend, eps_trend, w_trend, window_trend)

    X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], 

                         lag_deaths[:, -num_lag_death:], 

                         country_ids,

                         continents_ids,

                          population,

                         time_death,

                         #lockdown_death[idx, start_pred].reshape((-1, 1)),

                        ])

    y_death = target_deaths

    y_death_prev = lag_deaths[:, -1:]

    if fit:

        lr_death.fit(X_death, y_death)

    y_pred_death = lr_death.predict(X_death)

    if 0:

        w_t = lag_deaths[:, -1:]  

        w_t = (0 + w_t) / (0 + np.max(w_t))

        w_t = w_t**0.7

        w_t = 0.6 * w_t * (lag_deaths[:, -num_lag_death_trend:1-num_lag_death_trend]  > np.log1p(death_trend_thr))

    w_t = 0.4 * (lag_deaths[:, -num_lag_death_trend:1-num_lag_death_trend]  > np.log1p(death_trend_thr))

    y_pred_death = ((1 - w_t) *  y_pred_death + w_t * X_trend)

    y_pred_death = np.maximum(y_pred_death, y_death_prev)

    

    

    X_trend = predict(lag_cases, num_lag_trend, eps_trend, w_trend, window_trend)

    X_case = np.hstack([lag_cases[:, -num_lag_case:], 

                        country_ids, 

                        continents_ids,

                        population,

                        time_case,

                        lockdown_case[idx, start_pred].reshape((-1, 1)),

                       ])

    #print(X_case.shape, start_pred.shape, lockdown_case.shape, lockdown_case[idx, start_pred].shape)

    y_case = target_cases

    y_case_prev = lag_cases[:, -1:]

    if fit:

        lr_case.fit(X_case, y_case)

    y_pred_case = lr_case.predict(X_case)

    w_t = lag_cases[:, -1:]  

    w_t = (0 + w_t) / (0 + np.max(w_t))

    w_t = np.clip(1.1 * w_t**0.7, 0, 1)

    

    #w_t = w_t * (lag_cases[:, -num_lag_trend:1-num_lag_trend]  > np.log1p(case_trend_thr))

    

    y_pred_case = ((1 - w_t) *  y_pred_case + w_t * X_trend)

    y_pred_case = np.maximum(y_pred_case, y_case_prev)

    

    if score:

        death_score = val_score(y_death, y_pred_death)

        case_score = val_score(y_case, y_pred_case)

    else:

        death_score = 0

        case_score = 0

        

    return death_score, case_score, y_pred_death, y_pred_case



def update_time(time_death, time_case, pred_death, pred_case):

    new_time_death = np.expm1(time_death) + c_death * (pred_death >= np.log1p(t_death))

    new_time_death = 1 *np.log1p(new_time_death) 

    new_time_case = np.expm1(time_case) + c_case * (pred_case >= np.log1p(t_case))

    new_time_case = 1 *np.log1p(new_time_case) 

    return new_time_death, new_time_case



c_case = 1

t_case = 1000

c_death = 1

t_death = 100



time_cases = c_case * (cases >= np.log1p(t_case)) 

time_cases = np.cumsum(time_cases, axis=1)

time_cases = 1 * np.log1p(time_cases) 

time_cases.shape



time_deaths = c_death * (cases >= np.log1p(t_death))

time_deaths = np.cumsum(time_deaths, axis=1)

time_deaths = 1 *np.log1p(time_deaths) 

time_deaths.shape



def predict(tdata, num_lag, eps, w, window):

    num_pred = 1

    num_geo = tdata.shape[0]

    days = np.arange(num_lag)

    pred = np.zeros((num_geo, num_pred))

    start_pred = tdata.shape[1]

    x0 = np.arange(start_pred).reshape((-1, 1))

    w_delta = (window - 1)/2

    x1 = np.arange(start_pred + w_delta, start_pred + 1 + w_delta).reshape((-1, 1))

    w0 = np.array([w** i for i in range(num_lag)])

    w0 = w0 / w0.mean()

    for i in range(num_geo):

        y0 = pd.Series(tdata[i, :])

        y = y0.rolling(window=window).mean().fillna(method='bfill')

        y = y.diff().fillna(0)

        y = np.clip(y.values, 0, 1e6)

        if y.max() == 0:

            #print(0, geo)

            pred[i, :] = tdata[i, -1]

            continue

        x = x0

        filter_ = (y > 0)

        x = x[y > 0]

        y = y[y > 0]

        y = y[-num_lag:]

        x = x[-num_lag:]

        pad = num_lag - len(y)

        if pad > 0:

            y = np.hstack((np.zeros(pad), y))

            xmin = x.min()

            x = np.vstack((np.arange(xmin - pad, xmin).reshape((-1, 1)), x))

        w = w0[-len(y):]

        if len(y) < num_lag:

            #print(1, geo, y)

            pred[i, :] = tdata[i, -1]

            continue

        y = np.log(y + eps)

        lr = Ridge(fit_intercept=True)

        #lr = HuberRegressor()

        #print(w)

        lr.fit(x, y, w)

        if 0 and lr.coef_[0] > 0:

            print('up', lr.coef_, lr.intercept_)

            #lr.coef_[0] = 0

            #lr.intercept_ = y.mean()

        y_pred = lr.predict(x1)

        #print(y, y.mean(), y_pred)



        y_pred = np.clip(y_pred, -10, np.average(y, weights=w))

        y_pred = np.exp(y_pred) - eps

        y_pred = np.clip(y_pred, 0, 5)

        y_pred = np.cumsum(y_pred)

        y_pred = y_pred + tdata[i, -1]

        pred[i, :] = y_pred

    return pred



def get_lockdown(lockdown_date, day):

    lockdown = lockdown_date.copy()

    # only consider lockdown kinown before val period

    lockdown[:, day:] = 0

    time_since_lockdown = np.cumsum(lockdown, axis=1)

    lockdown_cases = np.log1p(1e1*np.clip(time_since_lockdown - 12, 0, 100))

    lockdown_deaths = np.log1p(1e1*np.clip(time_since_lockdown- 14, 0, 100))

    return lockdown_cases, lockdown_deaths



start_lag_death, end_lag_death = 15, 5,

num_train = 7

num_lag_death = 6

num_lag_case = 12

lag_period = 15

case_threshold = 30





(num_lag_trend, eps_trend, w_trend, window_trend) = (num_lag_case, 1e-5, 1.2, 5)

num_lag_death_trend = 12

death_trend_thr = 1

case_trend_thr = 1

scores = []

for start_val_delta in range(0, -11, -3):

    start_val = min_test_val_day + start_val_delta

    last_train = start_val - 1

    num_val = max_test_val_day - start_val + 1

    first_train = last_train + 1 - (num_train ) 

    #num_lag_case = num_val

    keep_cases = cases

    keep_deaths = deaths

    print(dates[last_train], '%3d %3d' % (start_val, num_val), end=' ')

    country_ids_base = get_country_ids(last_train, case_threshold)

    continents_ids_base = get_continent_ids()

    train_data = get_dataset(last_train, num_train, lag_period, 

                             continents_ids_base, country_ids_base, keep_cases, keep_deaths, 

                             population, time_cases, time_deaths, lockdown_date)

    valid_data = get_dataset(start_val, 1, lag_period, 

                             continents_ids_base, country_ids_base, keep_cases, keep_deaths, 

                             population, time_cases, time_deaths, lockdown_date)

    death_scores, case_scores, _, _ = train_model(train_data, valid_data, start_lag_death, end_lag_death, 

                                                  num_lag_case, num_val, True, keep_cases, keep_deaths,

                                                  )

    scores.append((death_scores + case_scores) / 2)

scores = np.mean(scores)

print('mean score: %0.4f' % scores)


def get_sub(start_val_delta=0):   

    start_val = min_test_val_day + start_val_delta

    last_train = start_val - 1

    num_val = max_test_val_day - start_val + 1

    first_train = last_train + 1 - (num_train ) 

    #num_lag_case = num_val

    keep_cases = cases

    keep_deaths = deaths

    print(dates[last_train], '%3d %3d' % (start_val, num_val), end=' ')

    country_ids_base = get_country_ids(last_train, case_threshold)

    continents_ids_base = get_continent_ids()

    train_data = get_dataset(last_train, num_train, lag_period, 

                             continents_ids_base, country_ids_base, keep_cases, keep_deaths, 

                             population, time_cases, time_deaths, lockdown_date)

    valid_data = get_dataset(start_val, 1, lag_period, 

                             continents_ids_base, country_ids_base, keep_cases, keep_deaths, 

                             population, time_cases, time_deaths, lockdown_date)

    _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, start_lag_death, end_lag_death, 

                                                  num_lag_case, num_val, True, keep_cases, keep_deaths,

                                                  )



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



sub = get_sub()
sub
#sub.to_csv('submission.csv', index=None)
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

    first_train = last_train + 1 - (num_train ) 

    #num_lag_case = num_val

    keep_cases = cases

    keep_deaths = deaths

    print(dates[last_train], '%3d %3d' % (start_val, num_val), end=' ')

    country_ids_base = get_country_ids(last_train, case_threshold)

    continents_ids_base = get_continent_ids()

    train_data = get_dataset(last_train, num_train, lag_period, 

                             continents_ids_base, country_ids_base, keep_cases, keep_deaths, 

                             population, time_cases, time_deaths, lockdown_date)

    valid_data = get_dataset(start_val, 1, lag_period, 

                             continents_ids_base, country_ids_base, keep_cases, keep_deaths, 

                             population, time_cases, time_deaths, lockdown_date)

    _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, start_lag_death, end_lag_death, 

                                                  num_lag_case, num_val, True, keep_cases, keep_deaths,

                                                  )



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

    sub['ForecastId'] = sub['ForecastId'] .astype('int')

    return sub



sub = get_final_sub()
sub
sub.to_csv('submission.csv', index=None)