# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression

from datetime import datetime

import math

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")

ForecastId = pd.DataFrame({'ForecastId': submission['ForecastId']})

ForecastId.head(5)
df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

df.head(5)
df['Country_Region'].nunique()
region_list = df[['Province_State', 'Country_Region']].drop_duplicates()

region_list
def select_data(train):

    confirm = train[['Date','ConfirmedCases']]

    confirm = confirm.rename(columns={'ConfirmedCases': 'y_data'})

    fatalty = train[['Date','Fatalities']]

    fatalty = fatalty.rename(columns={'Fatalities': 'y_data'})

    confirm.reset_index(drop=True, inplace=True)

    confirm.index = confirm.index + 1

    fatalty.reset_index(drop=True, inplace=True)

    fatalty.index = fatalty.index + 1

    return confirm, fatalty
def exponential_growth_model(t, k, P0):

    t=t/365

    return P0*np.exp(k*t)
def logistic_growth_model(t, K, P0, r):

    # t:time(yr)   t0:initial time    P0:initial_value    K:capacity  r:increase_rate

    t0=0

    t=t/365

    exp_value=np.exp(r*(t-t0))

    return (K*exp_value*P0)/(K+(exp_value-1)*P0)
def linear_growth_model(t, k, P0):

    t=t/365

    return k*t+P0
def RMSLE(predict, actual):

    rmsle = np.sqrt(np.mean(np.log((predict+1)/(actual+1))))

    return rmsle
def cal_date_interval(d_start, d_end):

    date_format = "%Y-%m-%d"

    a = datetime.strptime(d_start, date_format)

    b = datetime.strptime(d_end, date_format)

    delta = b - a

    return delta.days
def fit_pred(train):

    t_train = np.linspace(1, len(train['Date']), num=len(train['Date']))

    popt_exp, pcov_exp = curve_fit(exponential_growth_model, t_train, train['y_data'])

    popt_log, pcov_log = curve_fit(logistic_growth_model, t_train, train['y_data'], maxfev=2500)

    

#     Using RMSLE score to decide choose Exponential or Logistic growth model

    t_start = train[train['Date'].isin(['2020-04-02'])].index.tolist()[0]

    t_public_end = t_start + 12

    duration = 13

    t_public_pred = np.linspace(t_start, t_public_end, duration)

    y_actual = train['y_data'][train['Date']>='2020-04-02']

    

    y_exp_pred = exponential_growth_model(t_public_pred, *popt_exp)

    y_log_pred = logistic_growth_model(t_public_pred, *popt_log)

    eval_exp = RMSLE(y_exp_pred, y_actual)

    eval_log = RMSLE(y_log_pred, y_actual)



#     To get final prediction

    t_end = t_start + 42

    t_pred = np.linspace(t_start, t_end, 43)

    if eval_exp < eval_log:

        y_pred = exponential_growth_model(t_pred, *popt_exp)

        eval_public = eval_exp

    else:

        y_pred = logistic_growth_model(t_pred, *popt_log)

        eval_public = eval_log

        

#     plot(train['y_data'], logistic_growth_modellogistic_model(np.linspace(1, 107, 107), *popt))

    return y_pred, eval_public
def unqualify_data(train):

    '''

    When report errors 

    RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 2500.

    Using Linear Regression or Exponential to predict unqualified data

    '''

    

    if (train['y_data'][train.Date == '2020-04-02'].values > 0)[0] == True:

        train = train[train['y_data'] > 0]

    else:

        train = train[train['Date']>='2020-04-02']

    train.reset_index(drop=True, inplace=True)

    train.index = train.index + 1



    t_train = np.linspace(1, len(train['Date']), num=len(train['Date']))

    popt_exp, pcov_exp = curve_fit(exponential_growth_model, t_train, train['y_data'])

    popt_lin, pcov_lin = curve_fit(linear_growth_model, t_train, train['y_data'])

    

#     Using RMSLE score to decide choose Exponential or Linear growth model

    t_start = train[train['Date'].isin(['2020-04-02'])].index.tolist()[0]

    t_public_end = train.index.values[-1]

    duration = t_public_end - t_start + 1

    t_public_pred = np.linspace(t_start, t_public_end, duration)

    y_actual = train['y_data'][train['Date']>='2020-04-02']

    

    y_exp_pred = exponential_growth_model(t_public_pred, *popt_exp)

    y_lin_pred = linear_growth_model(t_public_pred, *popt_lin)

    eval_exp = RMSLE(y_exp_pred, y_actual)

    eval_lin = RMSLE(y_lin_pred, y_actual)



#     To get final prediction

    interval = cal_date_interval('2020-04-02', '2020-05-14')

    t_pred = np.linspace(t_start, t_start+interval, interval+1)

    if eval_exp < eval_lin:

        y_pred = exponential_growth_model(t_pred, *popt_exp)

        eval_public = eval_exp

    else:

        y_pred = linear_growth_model(t_pred, *popt_lin)

        eval_public = eval_lin

    return y_pred, eval_public   
def plot(actual, pred):

    plt.plot(np.linspace(1, len(actual), len(actual)), actual)

    plt.plot(np.linspace(1, len(pred), len(pred)), pred)

    plt.show()
c_pred_global = pd.DataFrame(columns=None)

f_pred_global = pd.DataFrame(columns=None)

c_eval = pd.DataFrame(columns=None)

f_eval = pd.DataFrame(columns=None)



for i in range(len(region_list)):

#     train = df[np.logical_and(np.logical_or(pd.isnull(df['Province_State']), df['Province_State']==region_list.iloc[i][0]), 

#                           df['Country_Region']==region_list.iloc[i][1])]

    train = df.iloc[84*i:84*(i+1), :]

    confirm, fatalty = select_data(train)

    

    try: 

        confirm_pred, confirm_eval = fit_pred(confirm)

    except:

        confirm_pred, confirm_eval = unqualify_data(confirm)

#     c_eval = c_eval.append(pd.DataFrame([confirm_eval],columns={'RMSLE'}, index=i))

    c_eval = c_eval.append(pd.DataFrame([confirm_eval],columns={'RMSLE'}))

    c_pred_global = c_pred_global.append(pd.DataFrame(confirm_pred), ignore_index=True)



    try: 

        fatalty_pred, fatalty_eval = fit_pred(fatalty)

    except:

        fatalty_pred, fatalty_eval = unqualify_data(fatalty)

#     f_eval = f_eval.append(pd.DataFrame([fatalty_eval],columns={'RMSLE'}, index=i))

    f_eval = f_eval.append(pd.DataFrame([fatalty_eval],columns={'RMSLE'}))

    f_pred_global = f_pred_global.append(pd.DataFrame(fatalty_pred), ignore_index=True)
# Evalutaion of public data

c_eval.sort_values(by=['RMSLE'])
f_eval.sort_values(by=['RMSLE'])
c_pred_global = c_pred_global.rename(columns={0:'ConfirmedCases'})

c_pred_global
f_pred_global = f_pred_global.rename(columns={0:'Fatalities'})

f_pred_global
# when axis=1, concat horizontaly

output = pd.concat([c_pred_global, f_pred_global], axis=1) 

output = pd.concat([ForecastId, output], axis=1) 

output
output.to_csv('submission.csv', header=True, index=False) 