import math

import numpy as np

import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



# Suppress warnings

import warnings

warnings.filterwarnings("ignore")



PUBLIC_PRIVATE = 1 # 0 for public leaderboard, 1 for private



# Listing files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train    = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

df_test     = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

sub_example = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

regr_ensemble_sub = pd.read_csv('/kaggle/input/regressors-ensemble-submission-week-4/submission (4).csv')

regr_ensemble_sub.head()
df_train['ForecastId'] = -1

df_train['DataType'] = 0

df_test['DataType'] = 2

df_test['ConfirmedCases'] = -1

df_test['Fatalities'] = -1

df_test['Id'] = df_test['ForecastId'] + df_train['Id'].max()

df_intersection = df_train[df_train['Date'] >= df_test['Date'].min()]

df_intersection['DataType'] = 1

df_intersection['ForecastId'] = df_test[df_test['Date'] <= df_train['Date'].max()]['ForecastId'].values

df_train = df_train[df_train['Date'] < df_test['Date'].min()]

df_test = df_test[df_test['Date'] > df_intersection['Date'].max()]

if not PUBLIC_PRIVATE:

    df_intersection['ConfirmedCases'] = -1

    df_intersection['Fatalities'] = -1

    df_full = pd.concat([df_train, df_intersection, df_test], sort=False, axis=0)

else:

    df_full = pd.concat([df_train, df_intersection, df_test], sort=False, axis=0)
# Joining Country_Region and Province_State cols

df_full['Province_State'] = df_full['Province_State'].fillna('')

df_full['Location'] = ['_'.join(x) for x in zip(df_full['Country_Region'], df_full['Province_State'])]

df_full.drop(columns=['Province_State', 'Country_Region'], inplace=True)



df_full.shape
import time

from datetime import datetime

df_full['Date'] = pd.to_datetime(df_full['Date'])

df_full['Date'] = df_full['Date'].apply(lambda s: time.mktime(s.timetuple()))

min_timestamp = np.min(df_full['Date'])

df_full['Date'] = df_full['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
import random

from scipy.optimize.minpack import curve_fit

from sklearn.metrics import r2_score

from scipy.special import expit



def Gompertz(a, c, t, t0):    

    Q = a * np.exp(-np.exp(-c*(t-t0)))

    return Q



ref = df_full[(df_full['DataType'].isin([0,1])) & (df_full['Location'] == 'China_Anhui')]

ref['ConfirmedCases'] /= ref['ConfirmedCases'].max()

ref['Fatalities'] /= ref['Fatalities'].max()



def getMultiplier (date):

    try:

        return 1.0 / ref[ref['Date'] == date]['ConfirmedCases'].values[0]

    except:

        return 3.5



locations = list(set(df_full['Location']))

location_sample = ['Brazil_']#random.sample(locations, 1)



train = df_full[df_full['DataType'] == 0]

valid = df_full[df_full['DataType'] == 1]

test =  df_full[df_full['DataType'] == 2]



for location in tqdm(locations):

    _train = train[train['Location'] == location]

    _valid = valid[valid['Location'] == location]

    _test  = test [test ['Location'] == location]

            

    n_train_days = _train.Date.nunique()

    n_valid_days = _valid.Date.nunique()

    n_test_days  = _test.Date.nunique()

    x_train      = range(n_train_days)

    x_test       = range(n_train_days + n_valid_days + n_test_days + 200)

    y_train_f    = _train['Fatalities']

    y_train_c    = _train['ConfirmedCases']

    

    # ConfirmedCases

    the_first_one = _train[_train['ConfirmedCases'] > 0.05 * _train['ConfirmedCases'].max()]['Date'].min()

    first_cases = _train[_train['ConfirmedCases'] > 0.4 * _train['ConfirmedCases'].max()]['Date'].min()

    if math.isnan(the_first_one):

        the_first_one = _train['Date'].max()

        first_cases = _train['Date'].max() + 8

    current = _train['Date'].max() - first_cases

    if location.startswith('China'):

        lower_c = [0, 0.02, 0]

        upper_c = [2*y_train_c.max()+1, 0.15, 25]

    else:

        lower_c = [0, 0.02, the_first_one]

        upper_c = [getMultiplier(current)*np.max(y_train_c)+1, 0.15, first_cases + 28]

    popt_c, pcov_c = curve_fit(Gompertz, x_train, y_train_c, method='trf', bounds=(lower_c,upper_c))

    a_max_c, estimated_c_c, estimated_t0_c = popt_c

    y_predict_c = Gompertz(a_max_c, estimated_c_c, x_test, estimated_t0_c)

    y_predict_c_at_t0 =  Gompertz(a_max_c, estimated_c_c, estimated_t0_c, estimated_t0_c)



    # Fatalities

    the_first_one = _train[_train['Fatalities'] > 0.05 * _train['Fatalities'].max()]['Date'].min()

    first_cases = _train[_train['Fatalities'] > 0.37 * _train['Fatalities'].max()]['Date'].min()

    if math.isnan(the_first_one):

        the_first_one = _train['Date'].max()

        first_cases = _train['Date'].max() + 8

    current = _train['Date'].max() - first_cases

    if location.startswith('China'):

        lower = [0, 0.02, 0]

        upper = [2*y_train_f.max()+1, 0.15, 25]

    else:

        lower = [0, 0.02, the_first_one]

        upper = [getMultiplier(current)*np.max(y_train_f)+1, 0.15, first_cases + 28]

    popt_f, pcov_f = curve_fit(Gompertz, x_train, y_train_f, method='trf', bounds=(lower,upper))

    a_max, estimated_c, estimated_t0 = popt_f

    y_predict_f = Gompertz(a_max, estimated_c, x_test, estimated_t0)

    y_predict_f_at_t0 =  Gompertz(a_max, estimated_c, estimated_t0, estimated_t0)

    

    

    from datetime import datetime, timedelta

    initial_date = datetime (2020, 1, 22)

    

    dates_train = list(x_train)

    dates_test = list(x_test)

    for i in range(len(dates_train)):

        dates_train[i] = initial_date + timedelta(days=dates_train[i])

    for i in range(len(dates_test)):

        dates_test[i] = initial_date + timedelta(days=dates_test[i])

        

#     plt.figure(figsize=(14,8))

#     plt.title('COVID-19 cases on Brazil')

#     plt.xlabel('Date')

#     plt.ylabel('Number')

#     plt.plot(dates_train, y_train_c, linewidth=2, color='#ff9933')

#     plt.plot(dates_test , y_predict_c, linewidth=2, color='#e67300', linestyle='dashed')

#     legend = []

#     legend.append('{} confirmed cases'.format('Brazil'))

#     legend.append('{} predicted cases'.format('Brazil'))

#     plt.legend(legend)

#     plt.show()



#     plt.figure(figsize=(14,8))

#     plt.title('COVID-19 cases on Brazil')

#     plt.xlabel('Date')

#     plt.ylabel('Number')

#     plt.plot(dates_train, y_train_f, linewidth=2, color='#ff9933')

#     plt.plot(dates_test , y_predict_f, linewidth=2, color='#e67300', linestyle='dashed')

#     legend = []

#     legend.append('{} fatalities'.format('Brazil'))

#     legend.append('{} fatalities'.format('Brazil'))

#     plt.legend(legend)

#     plt.show()



    values = y_predict_c[df_full[df_full['DataType'] == 2]['Date'].astype(int).min():df_full[df_full['DataType'] == 2]['Date'].astype(int).max() + 1]

    df_full.loc[(df_full['DataType'] == 2) & (df_full['Location'] == location), 'ConfirmedCases'] = values

    values = y_predict_f[df_full[df_full['DataType'] == 2]['Date'].astype(int).min():df_full[df_full['DataType'] == 2]['Date'].astype(int).max() + 1]

    df_full.loc[(df_full['DataType'] == 2) & (df_full['Location'] == location), 'Fatalities'] = values

def plotStatus (location):

    from datetime import datetime

    plt.figure(figsize=(14,8))

    plt.title('COVID-19 cases on {}'.format(location))

    _train = df_full.loc[(df_full['DataType'] == 0) & (df_full['Location'] == location)]

    _test = df_full.loc[(df_full['DataType'] == 1) & (df_full['Location'] == location)]

    _valid = df_full.loc[(df_full['DataType'] == 2) & (df_full['Location'] == location)]

    idx = test[test['Location'] == location].index

    legend = []

    plt.xlabel('Date')

    plt.ylabel('Number')

    plt.plot(_train['Date'], _train['ConfirmedCases'], linewidth=2)

    plt.plot(_test['Date'], _test['ConfirmedCases'], linewidth=2)

    plt.plot(_valid['Date'], _valid['ConfirmedCases'], linewidth=2)

    legend.append('{} train cases'.format(location))

    legend.append('{} test cases'.format(location))

    legend.append('{} validation cases'.format(location))

    plt.legend(legend)

    plt.show()

    legend = []

    plt.figure(figsize=(14,8))

    plt.title('COVID-19 fatalities on {}'.format(location))

    plt.xlabel('Date')

    plt.ylabel('Number')

    plt.plot(_train['Date'], _train['Fatalities'], linewidth=2)

    plt.plot(_test['Date'], _test['Fatalities'], linewidth=2)

    plt.plot(_valid['Date'], _valid['Fatalities'], linewidth=2)

    legend.append('{} train fatalities'.format(location))

    legend.append('{} test fatalities'.format(location))

    legend.append('{} validation fatalities'.format(location))

    plt.show()



location_sample = random.sample(locations, 10)

for location in location_sample:

    plotStatus(location)
submission = df_full[df_full['ForecastId'] > 0][['ForecastId', 'ConfirmedCases', 'Fatalities']].sort_values('ForecastId')

submission
merged_sub = pd.DataFrame()

WEIGHT_CC_REGR = .3

WEIGHT_CC_GOMP = .7

WEIGHT_FT_REGR = .3

WEIGHT_FT_GOMP = .7

merged_sub['ForecastId'] = submission['ForecastId']

merged_sub['ConfirmedCases'] = (regr_ensemble_sub['ConfirmedCases'].values * WEIGHT_CC_REGR + submission['ConfirmedCases'].values * WEIGHT_CC_GOMP)

merged_sub['Fatalities'] = (regr_ensemble_sub['Fatalities'].values * WEIGHT_FT_REGR + submission['Fatalities'].values * WEIGHT_FT_GOMP)

merged_sub
merged_sub.to_csv('submission.csv', index=False)