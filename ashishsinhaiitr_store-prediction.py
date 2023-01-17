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
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 12)

PATH = "../input"
train = pd.read_csv(f"{PATH}/train.csv", low_memory=False, 
                    parse_dates=['date'], index_col=['date'])
test = pd.read_csv(f"{PATH}/test.csv", low_memory=False, 
                   parse_dates=['date'], index_col=['date'])
sample_sub = pd.read_csv(f"{PATH}/sample_submission.csv")

train.head()
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
data.head()
data.item.unique()
data.store.unique()
gbs=data.groupby('store')

#Changes y Year
agg_year_item = pd.pivot_table(data, index='year', columns='item',
                               values='sales', aggfunc=np.mean).values
agg_year_store = pd.pivot_table(data, index='year', columns='store',
                                values='sales', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_year_item / agg_year_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_year_store / agg_year_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.show()
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
# Day of week pattern
dow_table = pd.pivot_table(data, index='dayofweek', values='sales', aggfunc=np.mean)
dow_table.sales /= grand_avg

# Yearly growth pattern
year_table = pd.pivot_table(data, index='year', values='sales', aggfunc=np.mean)
year_table /= grand_avg

years = np.arange(2013, 2019)
annual_sales_avg = year_table.values.squeeze()
p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))
p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))

# We pick the quadratic fit
annual_growth = p2

slightly_better_pred = slightly_better(test, sample_sub.copy())
slightly_better_pred.to_csv("sbp_float.csv", index=False)

# Round to nearest integer (if you want an integer submission)
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

# Round to nearest integer
wp_round = weighted_pred.copy()
wp_round['sales'] = np.round(wp_round['sales']).astype(int)
wp_round.to_csv("weight_predictor_2.csv", index=False)
os.listdir('../output')
sub1=pd.read_csv('../output/weight_predictor_1.csv')
sub2=pd.read_csv('../output/weight_predictor_2.csv')
sub3=pd.read_csv('../output/weight_predictor_3.csv')
sub4=pd.read_csv('../output/weight_predictor_4.csv')
sub1.to_csv('weight_predictor_1.csv',index=False)
sub2.to_csv('weight_predictor_2.csv',index=False)
sub3.to_csv('weight_predictor_3.csv',index=False)
sub4.to_csv('weight_predictor_4.csv',index=False)
sub5=sub1
sub5.sales=(sub1.sales*0.5+sub2.sales*0.5)
sub5.head()
sub6=sub1
sub6.sales=sub1.sales*0.4+sub2.sales*0.5+sub3.sales*0.1
sub6.head()
sub5.to_csv('mean_weight_predictor_5.csv',index=False)
sub6.to_csv('weight_predictor_6.csv',index=False)
