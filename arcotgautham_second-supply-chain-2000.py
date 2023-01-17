import os

import itertools

import time

import json

import tqdm



import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from pathlib import Path

%matplotlib inline



from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.model_selection import KFold, train_test_split



# Feature scaling, required for non-tree-based models

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from scipy.stats.mstats import winsorize



# Encoding categorical data for non-tree-based models

from sklearn.preprocessing import OneHotEncoder



from fbprophet import Prophet



from tqdm import tqdm as tqdm

goodsale = pd.read_csv('../input/goodsale.csv', thousands = ',')

submitfile = pd.read_csv("../input/submit_example.csv")

# full_date = pd.read_csv("../input/full_date.csv")
#date_time

goodsale["data_date"] = goodsale.data_date.map(lambda data:str(data))

goodsale["data_date"] = pd.to_datetime(goodsale.data_date)
#Full Date Generate

full_date = goodsale[['data_date']] #筛选date

full_date = full_date.drop_duplicates() #去重

full_date = full_date.sort_values(by = 'data_date') #排序

full_date["data_date"] = pd.to_datetime(full_date.data_date)

full_date



#Full Date Generate

%%time



sim_goodsale = submitfile.merge(goodsale, on = 'sku_id')

sim_goodsale = sim_goodsale[['sku_id', 'data_date', 'goods_num']]
sim_goodsale = sim_goodsale.sort_values(by = ['data_date'])
sim_goodsale = sim_goodsale.rename(columns = {'data_date': 'ds','goods_num': 'y'})

sim_goodsale


#grouped_goodsale

grouped_goodsale = sim_goodsale.groupby(sim_goodsale['sku_id'])
# grouped_goodsale = goodsale.groupby(goodsale['sku_id'])

grouped_goodsale.get_group('SKCRtFMV')
full_date = full_date.rename(columns = {'data_date': 'ds'})
def result_sum(item_forecast, sku_id):

    

    week1 = item_forecast[(item_forecast['ds'] >= '2018-05-01') & (item_forecast['ds'] <= '2018-05-07')]

    week1 = week1[['yhat']]

    week1 = week1.apply(sum)

    

    week2 = item_forecast[(item_forecast['ds'] >= '2018-05-08') & (item_forecast['ds'] <= '2018-05-14')]

    week2 = week2[['yhat']]

    week2 = week2.apply(sum)

    

    week3 = item_forecast[(item_forecast['ds'] >= '2018-05-15') & (item_forecast['ds'] <= '2018-05-21')]

    week3 = week3[['yhat']]

    week3 = week3.apply(sum)

    

    week4 = item_forecast[(item_forecast['ds'] >= '2018-05-22') & (item_forecast['ds'] <= '2018-05-28')]

    week4 = week4[['yhat']]

    week4 = week4.apply(sum)

    

    week5 = item_forecast[(item_forecast['ds'] >= '2018-05-29') & (item_forecast['ds'] <= '2018-06-04')]

    week5 = week5[['yhat']]

    week5 = week5.apply(sum)

    

    d = {'sku_id': [sku_id], 'week1': [week1[0]], 'week2': [week2[0]], 'week3': [week3[0]],

        'week4': [week4[0]], 'week5': [week5[0]]}

    result_df = pd.DataFrame(data=d)

    return result_df

    
# frame = {'sku_id': [], 'week1': [], 'week2': [], 'week3': [],

#     'week4': [], 'week5': []}

# final_df = pd.DataFrame(data=frame)

final_df = pd.DataFrame(columns=['sku_id', 'week1', 'week2', 'week3', 'week4', 'week5'])

final_df
item_df = grouped_goodsale.get_group('SKpLKkIS')

item_df

# %%time



# #full_data_date = 



# list_submit = list(submitfile['sku_id'])[:2]

# #list_submit = ['SKpLKkIS', 'SKDtK67I', 'SKMF0WaA', 'SKAP7e14']



# for item in tqdm(list_submit):





#     item_df = grouped_goodsale.get_group(item)



#     item_full_time_df = full_date.merge(item_df, how = 'outer')

#     item_full_time_df.fillna(0, inplace=True)

# #     item_full_time_df['sku_id'] = item

# #     item_full_time_df

    

    

#     train = item_full_time_df[["ds", "y"]]

    

#     #creat model

#     model = Prophet(weekly_seasonality=True, yearly_seasonality=True)

#     model.fit(train)

    

#     future = model.make_future_dataframe(periods=80)

#     #future.tail()

    

#     forecast = model.predict(future)

#     #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    

# #     final_df = final_df.append(result_sum(forecast, item))

#     final_df = pd.concat([final_df, result_sum(forecast, item)])



#     #item_df
%%time



final_df = pd.DataFrame(columns=['sku_id', 'week1', 'week2', 'week3', 'week4', 'week5'])

list_submit = list(submitfile['sku_id'])[:2000]

# list_submit = ['SKzY7aeL']

all_day_sale = 30509.255263

summit_day_sale = 123479.722222



for item in tqdm(list_submit):



    item_df = grouped_goodsale.get_group(item)



    item_full_time_df = full_date.merge(item_df, how = 'outer')

    item_full_time_df.fillna(0, inplace=True)

    

    train = item_full_time_df[["ds", "y"]]

    

    ####

    train.loc[(train['ds'] >= '2017-11-5') & 

          (train['ds'] <= '2017-11-27'),'y'] *= all_day_sale / summit_day_sale * 2

    ####

    

    #creat model

    model = Prophet(weekly_seasonality=False, yearly_seasonality=True )

    model.fit(train)

    

    future = model.make_future_dataframe(periods=80)

    #future.tail()

    

    forecast = model.predict(future)

    #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    

#     final_df = final_df.append(result_sum(forecast, item))

    final_df = pd.concat([final_df, result_sum(forecast, item)])



fig1 = model.plot(forecast)
final_df
final_df.to_csv('final_df_2000.csv', index = False)