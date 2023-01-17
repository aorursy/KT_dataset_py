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
path="../input/"

df=pd.read_csv(path+'train.csv',parse_dates=['date'],index_col='date')
df.head()
import matplotlib.pyplot as plt
plt.plot(df.index,df['sales'])
def expand_date_field(store_df):

    store_df['day']=store_df.index.day

    store_df['month']=store_df.index.month

    store_df['year']=store_df.index.year

    store_df['days_of_week']=store_df.index.dayofweek

    return store_df
ngen_dataframe=expand_date_field(df)
ngen_dataframe.store.unique()
pd.set_option('display.max_rows', 12)
grounp_df=ngen_dataframe.groupby('store')
import numpy as np
agg_year_item = pd.pivot_table(ngen_dataframe, index='year', columns='item',

                               values='sales', aggfunc=np.mean).values

agg_year_store = pd.pivot_table(ngen_dataframe,index='year', columns='store',

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
grand_avg = ngen_dataframe.sales.mean()



# Monthly pattern

month_table = pd.pivot_table(ngen_dataframe, index='month', values='sales', aggfunc=np.mean)

month_table.sales /= grand_avg
# Day of week pattern

dow_table = pd.pivot_table(ngen_dataframe, index='days_of_week', values='sales', aggfunc=np.mean)

dow_table.sales /= grand_avg
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
store_item_table = pd.pivot_table(ngen_dataframe, index='store', columns='item',

                                  values='sales', aggfunc=np.mean)

store_item_table
# Yearly growth pattern

year_table = pd.pivot_table(ngen_dataframe, index='year', values='sales', aggfunc=np.mean)

year_table /= grand_avg



years = np.arange(2013, 2019)

annual_sales_avg = year_table.values.squeeze()
p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))

p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))
annual_growth = p2
#test.csv



sample_sub=pd.read_csv(path+'sample_submission.csv')#,parse_dates=['date'],index_col='date')

test=pd.read_csv(path+'test.csv',parse_dates=['date'],index_col='date')
slightly_better_pred = slightly_better(test, sample_sub.copy())

slightly_better_pred.to_csv("result.csv", index=False)