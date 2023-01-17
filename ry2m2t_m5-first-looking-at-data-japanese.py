import os

import numpy as np

import pandas as pd
data_dir = "/kaggle/input/m5-forecasting-accuracy/"
!ls $data_dir
df_cal = pd.read_csv(data_dir + "calendar.csv")
df_cal
cols = ["d", "date"]

df_cal[cols].head()
df_cal[cols].tail()
s_cal_date = df_cal["date"].pipe(pd.to_datetime)
s_d = df_cal["d"].str.extract("(\d+)$", expand=False).astype(int)
s_d.groupby(s_cal_date.dt.year).agg(["min", "max"])
s_cal_month = s_cal_date.dt.to_period("M").dt.to_timestamp()

s_d.groupby(s_cal_month).agg(["min", "max"])
df_cal[["weekday", "wday"]]
cols_snap = ["snap_CA", "snap_TX", "snap_WI"]

df_cal[cols_snap].groupby(s_cal_month).sum()
df_cal[cols_snap].groupby(s_cal_date.dt.year).sum()
df_prices = pd.read_csv(data_dir + "sell_prices.csv")
df_prices.shape
df_prices.iloc[0:10]
df_sales_train_validation = pd.read_csv(data_dir + "sales_train_validation.csv")
df_sales_train_validation.shape
df_sales_train_validation.head(20)
cols = ["store_id", "state_id"]

df_stores = df_sales_train_validation[cols]
df_stores.groupby(cols).size()
cols = ["dept_id", "cat_id"]

df_depts = df_sales_train_validation[cols]
df_depts.groupby(cols).size()
df_depts.groupby(cols).size() / df_depts.shape[0]
df_sample_submission = pd.read_csv(data_dir + "sample_submission.csv")
df_sample_submission