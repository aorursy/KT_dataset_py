# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import math



from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout



import matplotlib.pyplot as plt

from scipy.stats import kstest

from scipy.signal import argrelextrema

import seaborn as sns

import datetime

%matplotlib inline

%config InlineBackend.figure_format = 'svg'



pd.set_option("display.max_columns",40)
del __

__ = pd.read_csv("../input/user_balance_table.csv")

__ = pd.merge(__,pd.read_csv("../input/mfd_day_share_interest.csv"),left_on="report_date", right_on="mfd_date",how='left')

__ = pd.merge(__,pd.read_csv("../input/mfd_bank_shibor.csv"),left_on="report_date", right_on="mfd_date",how='left')

__ = __.sort_values(by=["report_date","user_id"])

__ = __.groupby('report_date').sum().reset_index()

__ = __[__.report_date>=20140401].reset_index()

__['report_date'] = pd.to_datetime(__['report_date'].astype(str),format="%Y-%m-%d")

__['dayofweek'] = __['report_date'].dt.dayofweek

date_x = __['report_date']

figure2, ax2 = plt.subplots(1,1,figsize=(20,9))

plt.plot(date_x, __['consume_amt'],label = 'consume_amt')

plt.plot(date_x, __['transfer_amt'],label = 'transfer_amt')

plt.plot(date_x, __['tftocard_amt'],label = 'tftocard_amt')

plt.legend()

plt.show()
__ = __.set_index("report_date")

__ = pd.DataFrame(

    __,

    index=pd.date_range(

        start=__.index[0],

        periods=len(__.index) + 30,

        freq=__.index.freq

    )

)



             
from scipy.interpolate import spline

def plot(power,label=None):

    T=np.arange(len(power))

    xnew = np.linspace(T.min(),T.max(),300)

    power_smooth = spline(T,power,xnew)

    plt.plot(xnew,power_smooth,label=label)

def other_features_generator(df):

    festival_date = [405,406,407,

                     501,502,503,

                     531,601,602,

                     906,907,908,

                     101,102,103,104,105,106,107]

    festival_days_vector = {

        0:[0,0,0],

        2:[0,0,1],

        3:[0,1,0],

        7:[1,0,0]

    }

    other_features = [[] for i in range(10)]

    

    for v in df.index:

        dayofweek = v.dayofweek

        

        month_day = int(v.strftime("%Y%m%d"))-20140000

        

        

        #假期

        if month_day in festival_date or (dayofweek in (5,6) and month_day not in (504,928)):

            other_features[0].append(1)

        else:

            other_features[0].append(0)

            

        #放假几天

        if month_day in (404,430,530,905):

            other_features[1].append(festival_days_vector[3])

        elif month_day==930:

            other_features[1].append(festival_days_vector[7])

        elif dayofweek==4 and month_day!=502:

            other_features[1].append(festival_days_vector[2])

        else:

            other_features[1].append(festival_days_vector[0])

        

        #周日补班

        if month_day in (504,928):

            other_features[2].append(1)

        else:

            other_features[2].append(0)

            

        #放假前最后一天上班

        if month_day in (404,430,530,905,930) or (dayofweek==4 and month_day!=502):

            other_features[3].append(1)

        else:

            other_features[3].append(0)

        

        #放假后第一天上班

        if month_day in (408,504,603,909) or (dayofweek==0 and month_day not in (407,602,908)):

            other_features[4].append(1)

        else:

            other_features[4].append(0)

        

        #每月第一天

        if month_day%100 == 1:

            other_features[5].append(1)

        else:

            other_features[5].append(0)

            

        #月初、中、末

        if month_day%100 <= 10:

            other_features[6].append([0,0,1])

        elif month_day%100 <= 20:

            other_features[6].append([0,1,0])

        else:

            other_features[6].append([1,0,0])

        

        #放假第一天

        if month_day in (405,501,531,906) or (dayofweek==5 and month_day not in (503,)):

            other_features[7].append(1)

        else:

            other_features[7].append(0)

        

        #上班前放了几天假

        if month_day in (408,504,603,909):

            other_features[8].append([1,0])

        elif dayofweek==0 and month_day not in (407,502,908):

            other_features[8].append([0,1])

        else:

             other_features[8].append([0,0])

        #几天的假

        if month_day in festival_date:

            other_features[9].append([1,0])

        elif dayofweek in (5,6):

            other_features[9].append([0,1])

        else:

            other_features[9].append([0,0])

        

        

        

        

    return pd.concat([

        pd.DataFrame(other_features[0],columns = ['isFestival']),

        pd.DataFrame(other_features[1],columns = ['feArg1','feArg2','feArg3']),

        pd.DataFrame(other_features[2],columns = ['buban']),

        pd.DataFrame(other_features[3],columns = ['workBeforeFe']),

        pd.DataFrame(other_features[4],columns = ['firstdayWorkFestival']),

        pd.DataFrame(other_features[5],columns = ['firstdayMonth']),

        pd.DataFrame(other_features[6],columns = ['Mon1','Mon2','Mon3']),

        pd.DataFrame(other_features[7],columns = ['firstdayFestival']),

        pd.DataFrame(other_features[8],columns = ['howDaysAfterFe1','howDaysAfterFe2']),

        pd.DataFrame(other_features[9],columns = ['howDaysWenFe1','howDaysWenFe2']),

    ],axis=1

        

    )

            

        



def get_feature_vector(start='20140801',periods=60):

    df = pd.DataFrame(pd.date_range(start=start,periods=periods),columns=['date'])

    df['dayofweek'] = df['date'].dt.dayofweek

    df['date']= df['date'].dt.strftime("%Y%m%d")

    other_features = other_features_generator(df, 'date', 'dayofweek')

    for u,v in other_features.iterrows():

        yield v.values
purchase_scaler = MinMaxScaler(feature_range=(-1, 1))

__['purchase'] = purchase_scaler.fit_transform(__[['total_purchase_amt']].values)

redeem_scaler = MinMaxScaler(feature_range=(-1, 1))

__['redeem'] = redeem_scaler.fit_transform(__[['total_redeem_amt']].values)
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

from sklearn.model_selection import GridSearchCV



def FM(X, y, plot=False, max_depth=7,n_estimators=100,max_features='auto'):

    

    regr = RandomForestRegressor(max_depth=max_depth, random_state=0,n_estimators=n_estimators,max_features=max_features)

    

    

    regr.fit(X[:-30], y[:-30])

    predict = regr.predict(X)

    if plot:

        f, ax = plt.subplots(1,1,figsize=(16,9))

        plt.plot(__.index,predict,label="predict")

        plt.plot(__.index[:-30],y[:-30],label="actual")

        plt.legend()

        plt.show()

    return predict, regr.score(forest_purchase_X[:-30], forest_purchase_y[:-30])

    



forest_purchase_X = other_features_generator(__).values

forest_purchase_y = __['purchase'].values



forest_purchase_predict,forest_purchase_score = FM(

    forest_purchase_X, 

    forest_purchase_y, 

    plot=False, 

    max_depth=8,

    n_estimators=100

    )

print(forest_purchase_score)

forest_redeem_X = other_features_generator(__).values

forest_redeem_y = __['redeem'].values





forest_redeem_predict,forest_redeem_score = FM(

    forest_redeem_X, 

    forest_redeem_y, 

    plot=False, 

    max_depth=8,

    n_estimators=100,

    max_features=None)

print(forest_redeem_score)



forest_purchase_predict = purchase_scaler.inverse_transform(forest_purchase_predict.reshape(-1,1)).flatten()

forest_redeem_predict = redeem_scaler.inverse_transform(forest_redeem_predict.reshape(-1,1)).flatten()



pd.concat([

    pd.Series(pd.date_range(start="20140901",end="20140930")).dt.strftime("%Y%m%d"),

    pd.Series(forest_purchase_predict[-30:]),

    pd.Series(forest_redeem_predict[-30:]),

],axis=1).astype(int).to_csv("tc_comp_predict_table.csv", header=False, index=False)

!cat tc_comp_predict_table.csv
def LM(X, y, plot=False):

    scaler = MinMaxScaler(feature_range=(-1, 1))

    

    lr = LinearRegression()

    lr.fit(X[:-30], y[:-30])

    

    



    date_range = pd.date_range('20140401','20140930')

    

    predict = lr.predict(X)

    if plot:

        f, ax = plt.subplots(1, 1, figsize=(16,9))

        plt.plot(date_range,predict,label="predict")

        plt.plot(date_range[:-30],y[:-30],label="actual")



        plt.legend()

        plt.show()

    return (predict,lr.score(X[:-30], y[:-30]))



from itertools import combinations

max_purchase_li = None

max_purchase_score = 0

max_redeem_li = None

max_redeem_score = 0

for li in combinations([i for i in range(16)],16):

    purchase_X = other_features_generator(__).values[:,li]

    purchase_y = __['total_purchase_amt'].values

     

    purchase_predict, purchase_score = LM(purchase_X, purchase_y, True)

    if purchase_score>max_purchase_score:

        max_purchase_score = purchase_score

        max_purchase_li = li

for li in combinations([i for i in range(16)],16):

    redeem_X = other_features_generator(__).values[:,li]

    redeem_y = __['total_redeem_amt'].values



    redeem_predict, redeem_score = LM(redeem_X, redeem_y, True)

    if redeem_score>max_redeem_score:

        max_redeem_score = redeem_score

        max_redeem_li = li

print(max_purchase_li,max_purchase_score)

print(max_redeem_li,max_redeem_score)

'''

(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) 0.6807225271179145

(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) 0.5313783833240393

'''
pd.concat([

    pd.Series(pd.date_range(start="20140901",end="20140930")).dt.strftime("%Y%m%d"),

    pd.Series(purchase_predict[-30:]),

    pd.Series(redeem_predict[-30:]),

],axis=1).astype(int).to_csv("tc_comp_predict_table.csv", header=False, index=False)

!cat tc_comp_predict_table.csv
from statsmodels.tsa.seasonal import seasonal_decompose

STL_data = __[['total_purchase_amt','total_redeem_amt']]

STL_purchase = STL_data['total_purchase_amt']

STL_redeem = STL_data['total_redeem_amt']

purchase_result = seasonal_decompose(STL_purchase[:-30], model='multiplicative')

redeem_result = seasonal_decompose(STL_redeem[:-30], model='multiplicative')

purchase_result.plot()

plt.title('purchase')

plt.show()



redeem_result.plot()

plt.title('redeem')

plt.show()
def LM_STL(X, y):

    lr = LinearRegression()

    lr.fit(X[3:-33], y[3:-33])

    



    f, ax = plt.subplots(1, 1, figsize=(16,9))

    date_range = pd.date_range('20140404','20140930')

    print(lr.score(X[3:-33], y[3:-33]))

    predict = lr.predict(X[3:])

    print(predict.shape)

    plt.plot(date_range,predict,label="predict")

    plt.plot(date_range[:-33],y[3:-33],label="actual")

    

    plt.legend()

    plt.show()

    return predict
def append_df(df):

    return pd.DataFrame(

    df,

    index=pd.date_range(

        start=df.index[0],

        periods=len(df.index) + 30,

        freq=df.index.freq

    )

)



purchase_trend_predict = LM_STL(purchase_X, append_df(purchase_result.trend))

redeem_trend_predict = LM_STL(redeem_X, append_df(redeem_result.trend))
purchase_resid_predict = LM_STL(purchase_X, append_df(purchase_result.resid))

redeem_resid_predict = LM_STL(redeem_X, append_df(redeem_result.resid))
redeem_result.resid *redeem_result.seasonal*redeem_result.trend