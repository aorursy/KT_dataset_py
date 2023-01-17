import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

df_test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

df_sub = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')
df_train.describe
country = df_train['Country/Region']
country = set(country)
country
date = df_train['Date']

len(date)
for i in range(len(date)):

    dt = date[i]

    mm = dt[5:7]

    dd = dt[8:10]

    mm= int(mm)

    dd = int(dd)

    if(mm==1):

        day = dd

    elif(mm==2):

        day = 31+dd

    elif(mm==3):

        day = 31+29+dd

    date[i] = day
df_train['Date'] = date
df_train.describe