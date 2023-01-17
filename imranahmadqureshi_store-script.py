# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv", low_memory=False, 

                    parse_dates=['date'], index_col=['date'])

test = pd.read_csv("../input/test.csv", low_memory=False, 

                   parse_dates=['date'], index_col=['date'])

sample_sub = pd.read_csv("../input/sample_submission.csv")
def expand_df(df):

    data = df.copy()

    data['day'] = data.index.day

    data['month'] = data.index.month

    data['year'] = data.index.year

    data['dayofweek'] = data.index.dayofweek

    data['dayofyear']=data.index.dayofyear

    data['weekofyear']=data.index.weekofyear

    return data
data = expand_df(train)
def slightly_better(test, submission):

    submission[['sales']] = submission[['sales']].astype(np.float64)

    for _, row in test.iterrows():

        dow, month, year = row.name.dayofweek, row.name.month, row.name.year

        item, store = row['item'], row['store']

        base_sales = store_item_table.at[store, item]

        mul = month_table.at[month, 'sales'] * dow_table.at[dow, 'sales']

        pred_sales = base_sales * mul * annual_growth(year)

        submission.at[row['id'], 'sales'] = pred_sales

    return submission
store_item_table = pd.pivot_table(data, index='store', columns='item',

                                  values='sales', aggfunc=np.mean)
grand_avg = data.sales.mean()



# Monthly pattern

month_table = pd.pivot_table(data, index='month', values='sales', aggfunc=np.mean)

month_table.sales /= grand_avg
dow_table = pd.pivot_table(data, index='dayofweek', values='sales', aggfunc=np.mean)

dow_table.sales /= grand_avg
year_table = pd.pivot_table(data, index='year', values='sales', aggfunc=np.mean)

year_table /= grand_avg

years = np.arange(2013, 2019)

annual_sales_avg = year_table.values.squeeze()
p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))

p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))
annual_growth = p2
slightly_better_pred = slightly_better(test, sample_sub.copy())

slightly_better_pred.to_csv("sbp_float.csv", index=False)
sbp_round = slightly_better_pred.copy()

sbp_round['sales'] = np.round(sbp_round['sales']).astype(int)

sbp_round.to_csv("sbp_round.csv", index=False)
years = np.arange(2013, 2019)

annual_sales_avg = year_table.values.squeeze()

weights = np.exp((years - 2018)/6)

annual_growth = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2, w=weights[:-1]))

print(f"2018 Relative Sales by Weighted Fit = {annual_growth(2018)}")
def weighted_predictor(test, submission):

    submission[['sales']] = submission[['sales']].astype(np.float64)

    for _, row in test.iterrows():

        dow, month, year = row.name.dayofweek, row.name.month, row.name.year

        item, store = row['item'], row['store']

        base_sales = store_item_table.at[store, item]

        mul = month_table.at[month, 'sales'] * dow_table.at[dow, 'sales']

        pred_sales = base_sales * mul * annual_growth(year)

        submission.at[row['id'], 'sales'] = pred_sales

    return submission
weighted_pred = weighted_predictor(test, sample_sub.copy())

wp_round = weighted_pred.copy()

wp_round['sales'] = np.round(wp_round['sales']).astype(int)

wp_round.to_csv("weight_predictor_2.csv", index=False)