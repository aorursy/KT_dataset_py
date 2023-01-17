import os, gc, pickle, copy, datetime, warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn import metrics

pd.set_option('display.max_columns', 100)

warnings.filterwarnings('ignore')
df_test = pd.read_csv("../input/my-covid-pred/test_week2.csv")

df_test.head()
df_week4 = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

df_week4.head()
df_test2 = pd.merge(df_test, df_week4, on=['Province_State', 'Country_Region', 'Date'], how='left')

df_test2.head()
df_test2['Date'] = pd.to_datetime(df_test2['Date'])

df_test2['day'] = df_test2['Date'].apply(lambda x: x.dayofyear).astype(np.int16)

df_test2.head()
# check the last day of existing true data

tmp = df_test2[pd.isna(df_test2['ConfirmedCases'])==False]['Date'].max()

print("last day of existing true data: {}".format(tmp))
df_sub_osciiart_bug = pd.read_csv("../input/my-covid-pred/submission1.csv") # my final submission with bug

df_sub_osciiart_fixed = pd.read_csv("../input/my-covid-pred/submission_osciiart_fixed.csv") # my fixed submission

df_sub_kaz = pd.read_csv("../input/my-covid-pred/submission_Kaz.csv") # 1st place solution

df_sub_osciiart_bug.head()
def calc_score(y_true, y_pred):

    y_true[y_true<0] = 0

    score = metrics.mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5

    return score



def calc_private_score(df_sub):

    day_before_private = 92

    period = (pd.isna(df_test2['ConfirmedCases'])==False) & (df_test2['day']>day_before_private)



    y_true = df_test2['ConfirmedCases'][period].values

    y_pred = df_sub['ConfirmedCases'][period].values

    score1 = calc_score(y_true, y_pred)

    y_true = df_test2['Fatalities'][period].values

    y_pred = df_sub['Fatalities'][period].values

    score2 = calc_score(y_true, y_pred)

    score = (score1+score2)/2

    return score
print("df_sub_osciiart_bug: {:.5f}".format(calc_private_score(df_sub_osciiart_bug)))

print("df_sub_osciiart_fixed: {:.5f}".format(calc_private_score(df_sub_osciiart_fixed)))

print("df_sub_kaz: {:.5f}".format(calc_private_score(df_sub_kaz)))