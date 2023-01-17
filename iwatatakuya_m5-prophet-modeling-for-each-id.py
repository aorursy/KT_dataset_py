import gc

import os

import random

import csv

import sys

import json

import datetime



import lightgbm as lgb

import numpy as np

import pandas as pd

import seaborn as sns

from collections import Counter

from numba import jit

pd.set_option('display.max_rows', 1000)

pd.set_option('display.max_columns', 1000)



from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn import metrics

import statsmodels.api as sm

from statsmodels.tsa.arima_model import ARIMA

from fbprophet import Prophet

from tqdm import tqdm



plt.style.use("seaborn")

sns.set(font_scale=1)
df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")

price = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")
WRMSSE = 0.0

df_all = df.copy()

if os.path.exists("/kaggle/working/sub.csv"):

    print("Read existing file.")

    sub = pd.read_csv("/kaggle/working/sub.csv")

    res = pd.read_csv("/kaggle/working/res.csv")

    for i in range(len(res)):

        WRMSSE += res.weight[i] * res.rmsse[i]

else:

    print("Make a new file.")

    sub = pd.DataFrame()

    res = pd.DataFrame()

    ids = df.id.unique()
df = pd.melt(df,id_vars = df.columns[df.columns.str.endswith("id")],value_vars = df.columns[df.columns.str.startswith("d_")])

# df["day"] = df.variable.str[2:6].astype(int)

df = df.rename(columns = {"value" : "sales"})

df
df = df.set_index("id")
df_store_item = df.loc["HOBBIES_1_001_TX_1_validation"]

df_store_item.head(10)
df_store_item = df_store_item.merge(calendar, left_on  = "variable", right_on = "d",how = "left")

df_store_item = df_store_item.merge(price,on = ["store_id", "item_id", "wm_yr_wk"], how = "left")

df_store_item = df_store_item.dropna(subset = ["sell_price"])

df_store_item.head(10)
df_store_item['date'] = pd.to_datetime(df_store_item['date'])

# df_store_item = df_store_item.rename()
df_train = df_store_item.iloc[1:len(df_store_item)-28]

df_val = df_store_item.iloc[len(df_store_item)-28:len(df_store_item)]

df_train.head(10)
model = Prophet(weekly_seasonality = True, yearly_seasonality = True)

model.fit(df_train.loc[:,["date","sales"]].rename(columns = {"date" : "ds", "sales" : "y"}))

future = model.make_future_dataframe(28)

forecast = model.predict(future)

forecast

model.plot(forecast)

plt.show()
forecast
def RMSSE(pred, act, train):

    one_day_ago = train.loc[:,["date","sales"]].copy()

    one_day_ago["date"] = one_day_ago["date"]+datetime.timedelta(days=1)

    denom = train.loc[:,["date","sales"]].merge(one_day_ago, on = "date", how = "inner")

    denom = (denom["sales_x"] - denom["sales_y"]) * (denom["sales_x"] - denom["sales_y"])

    denom = denom.mean()

    pred = pred.merge(act, left_on = "ds", right_on = "date", how = "inner")

    mole = (pred["yhat"] - act["sales"].reset_index()["sales"])* (pred["yhat"] - act["sales"].reset_index()["sales"])

    mole = mole.mean()

    return mole/denom
RMSSE(forecast,df_val,df_train)
d_val = ["d_1913","d_1912","d_1911","d_1910","d_1909","d_1908","d_1907",

        "d_1906", "d_1905", "d_1904", "d_1903", "d_1902", "d_1901", "d_1900",

        "d_1899", "d_1898", "d_1897", "d_1896", "d_1895", "d_1894", "d_1893",

        "d_1892", "d_1891", "d_1890", "d_1889", "d_1888", "d_1887", "d_1886"]
df_val = df.loc[df.variable.isin(d_val)]

w = df_val.groupby(["id"])["sales"].sum()

tot = w.sum()

w = w/tot

w
if os.path.exists("/kaggle/working/sub.csv"):

    df = df[~df.index.isin(sub.id)]

    ids = df.index.unique()
i = 1

for store_item_id in tqdm(ids):

    df_store_item = df.loc[store_item_id]

    df_store_item = df_store_item.merge(calendar, left_on  = "variable", right_on = "d",how = "left")

    df_store_item = df_store_item.merge(price,on = ["store_id", "item_id", "wm_yr_wk"], how = "left")

    df_store_item = df_store_item.dropna(subset = ["sell_price"])

    df_store_item['date'] = pd.to_datetime(df_store_item['date'])

    df_train = df_store_item.iloc[1:len(df_store_item)-28]

    df_val = df_store_item.iloc[len(df_store_item)-28:len(df_store_item)]

    model = Prophet(weekly_seasonality = True, yearly_seasonality = True)

    model.fit(df_train.loc[:,["date","sales"]].rename(columns = {"date" : "ds", "sales" : "y"}))

    future = model.make_future_dataframe(28)

    forecast = model.predict(future)

    

    res = res.append(pd.DataFrame(data = {"id" : store_item_id,

                                         "weight" : w.loc[store_item_id],

                                         "rmsse" : RMSSE(forecast,df_val,df_train)},index=['i',]))

    WRMSSE += w.loc[store_item_id] * RMSSE(forecast,df_val,df_train)

#     print(i,w.loc[store_item_id],RMSSE(forecast,df_val,df_train))



    model = Prophet(weekly_seasonality = True, yearly_seasonality = True)

    model.fit(df_store_item.loc[:,["date","sales"]].rename(columns = {"date" : "ds", "sales" : "y"}))

    future = model.make_future_dataframe(28)

    forecast = model.predict(future)

    

    sub_id = pd.DataFrame(data = {"pred" : forecast.tail(28).yhat,

                           "col" : ["F" + str(i+1) for i in range(28)]})

    sub_id = sub_id.pivot_table(values=['pred'],columns=['col'], aggfunc='sum').reset_index()

    sub_id["id"] = store_item_id

    sub_id.drop(["index"], axis = 1, inplace = True)

    l = ["F" + str(i+1) for i in range(28)]

    l.insert(0,"id")

    sub = sub.append(sub_id[l])

    

    i = i + 1

    # comment here if you forecast sales for all id

    if i == 5:

        break

    sub.to_csv("sub.csv", index=False)

    res.to_csv("res.csv", index=False)    
submission = submission.loc[:,["id"]].merge(sub, on = "id", how = "left").fillna(0)

submission.to_csv("submission.csv", index=False)