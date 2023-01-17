# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np

import pandas as pd 
date_info = pd.read_csv("../input/date_info.csv")

date_info.head(10)
date_info.info()
# Convert vars to datetime

date_info['calendar_date'] = pd.DatetimeIndex(date_info['calendar_date'])

date_info.info()
colname = 'calendar_date'

date_info[colname+"_year"] = date_info[colname].dt.year

date_info[colname+"_month"] = date_info[colname].dt.month

date_info[colname+"_day"] = date_info[colname].dt.day 

date_info[colname+"_weekday"] = date_info[colname].dt.weekday # monday:0 , tuesday: 1 , ...

date_info[colname+"_hour"] = date_info[colname].dt.hour

date_info[colname+'_weekend'] = [1 if weekday >= 5 else 0 for weekday in date_info[colname+"_weekday"]]

date_info.head(5).T