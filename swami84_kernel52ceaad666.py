

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from xgboost import XGBRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):



    for filename in filenames:

        print(os.path.join(dirname, filename))

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

# test['Province_State'].fillna('', inplace = True)

# train['Province_State'].fillna('', inplace = True)

test['Date'] = pd.to_datetime(test['Date'])

train['Date'] = pd.to_datetime(train['Date'])

train['Province_State'] = train['Province_State'].fillna('None')

train['unique_id'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)

train.head(3)

from sklearn.linear_model import LinearRegression

days = 0

n_days = 7

len_test = test['Date'].nunique()

y_true_c,y_predict_c,y_true_f,y_predict_f = [],[],[],[]



train_US_State = train[train['unique_id'].str.contains('US')]

us_tot = train_US_State.groupby('Date')['ConfirmedCases'].sum().values.flatten()

train_Canada_State = train[train['unique_id'].str.contains('Canada')]

canada_tot = train_Canada_State.groupby('Date')['ConfirmedCases'].sum().values.flatten()



for uid in train['unique_id'].unique():

    

    df = train[train['unique_id'] == uid]

    y_conf = df.set_index('Date')['ConfirmedCases'].values.flatten()

    y_death = df.set_index('Date')['Fatalities'].values.flatten()

    #y_conf = np.ediff1d(y_conf)

    #y_conf = np.append(np.zeros(1), y_conf)

    #y_death = np.ediff1d(y_death)

    #y_death = np.append(np.zeros(1), y_death)

    lin_reg = LinearRegression()

    slopes  = []

    

    intercept = []

    idx = len(y_conf)-days

    y_train_conf = y_conf[:idx]

    y_test_conf = y_conf[idx:]

    y_train_death = y_death[:idx]

    y_test_death = y_death[idx:]

    y_train_conf = np.ediff1d(y_train_conf)

    y_train_conf = np.append(np.zeros(1), y_train_conf)

    y_train_death = np.ediff1d(y_train_death)

    y_train_death = np.append(np.zeros(1), y_train_death)

    for i in range(0,int(len(y_train_conf)/n_days)):

        y_train = y_train_conf[i*n_days:(i+1)*n_days]

        x = np.arange(i*n_days,(i+1)*n_days,1).reshape(-1,1)

        lin_reg.fit(x,y_train)

        slope = (lin_reg.coef_)

        slopes.append(slope)

        intercept.append(lin_reg.intercept_)

    n = len(y_train_conf)

    x = np.arange(n,n+days+len_test,1)

    x = np.arange(0,len_test + days,1)

    y_pred_conf = lin_reg.coef_[0]*x + lin_reg.intercept_

    y_pred_conf = np.append(y_conf[:idx], y_pred_conf)

    y_pred_conf = np.cumsum(y_pred_conf)

    y_pred_conf = y_pred_conf[-(len_test+days):]

    y_pred_conf[y_pred_conf < 0] =0

    for i in range(0,int(len(y_train_death)/n_days)):

        y_train = y_train_death[i*n_days:(i+1)*n_days]

        x = np.arange(i*n_days,(i+1)*n_days,1).reshape(-1,1)

        lin_reg.fit(x,y_train)

        slope = (lin_reg.coef_)

        slopes.append(slope)

        intercept.append(lin_reg.intercept_)

    n = len(y_train_conf)

    x = np.arange(n,n+days+len_test,1)

    y_pred_death = lin_reg.coef_[0]*x + lin_reg.intercept_

    y_pred_death = np.append(y_death[:idx], y_pred_death)

    y_pred_death = np.cumsum(y_pred_death)

    y_pred_death = y_pred_death[-(len_test+days):]

    y_pred_death[y_pred_death < 0] =0

    

    y_true_c.append(y_test_conf)

    y_true_f.append(y_test_death)

    y_predict_c.append(y_pred_conf)

    y_predict_f.append(y_pred_death)

    
y_true_c = [item for sublist in y_true_c for item in sublist]

y_predict_c = [item for sublist in y_predict_c for item in sublist]

y_predict_c = [round(elem, 2) for elem in y_predict_c]

y_true_f = [item for sublist in y_true_f for item in sublist]

y_predict_f = [item for sublist in y_predict_f for item in sublist]

y_predict_f = [round(elem, 2) for elem in y_predict_f]
submission = pd.DataFrame({'ForecastId': test['ForecastId'], 

                           'ConfirmedCases': y_predict_c, 

                           'Fatalities': y_predict_f})

submission_int = submission.round(0)

submission_int = submission_int.astype(int)



submission_int.to_csv('submission.csv', index = False)

len(submission)