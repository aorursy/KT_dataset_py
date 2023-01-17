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

import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARMA

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import warnings

warnings.filterwarnings('ignore')
def clean_date(df,date_field, format = None):

    if format:

        df[date_field] = pd.to_datetime(df[date_field], format = format)

    else:

        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format = True)

    df['Month'] = df[date_field].dt.month

    df['Year'] = df[date_field].dt.year

    df['Week'] = df[date_field].dt.week
def filter_shop(df, store_id):

    return df.loc[df['shop_id'] == store_id]
def filter_item(df, item_id):

    return df.loc[df['item_id'] == item_id]
def filter_month_year(df, month = None, year = None):

    if month and year:

        return df.loc[(df['Month'] == month) & (df['Year'] == year)]

    elif month:

        return df.loc[df['Month'] == month]

    else:

        return df.loc[df['Year'] == year]
def plot_sales(df, x_field, y_field, size = None):

    if size:

        plt.figure(figsize = size)

    else:

        plt.figure(figsize = (10,5))

    plt.plot(df[x_field], df[y_field])

    plt.grid()

    plt.legend()

    plt.title('Plot of '+ y_field + ' vs ' + x_field)

    plt.xticks(rotation = 45)

    plt.show()
def plot_bar(df, x_field, y_field):

    plt.figure(figsize = (10,5))

    sns.barplot(x = x_field, y = y_field, data = df)

    plt.show()
items_df = pd.read_csv('../input/items.csv')
item_categories_df = pd.read_csv('../input/item_categories.csv')
sales_train_df = pd.read_csv('../input/sales_train.csv')
sales_train_df.head()
sales_train_df.shape[0]
clean_date(sales_train_df,'date','%d.%m.%Y')
sales_train_df.head()
item_counts = pd.DataFrame(sales_train_df['item_id'].value_counts()).reset_index()

item_counts.rename(columns = {'index':'item_id','item_id':'count'},inplace = True)
plot_bar(item_counts.head(10),'item_id','count')
shop_counts = pd.DataFrame(sales_train_df['shop_id'].value_counts()).reset_index()

shop_counts.rename(columns = {'index':'shop_id','shop_id':'count'},inplace = True)
plot_bar(shop_counts.head(10),'shop_id','count')
sales_top = filter_item(filter_shop(sales_train_df, 31),20949)
plot_sales(sales_top.sort_values(by = 'date'),'date','item_cnt_day')
sales_top_monthly = sales_top.groupby(['Month','Year']).agg({'item_cnt_day':'sum'}).reset_index()

sales_top_monthly['Month-Year'] = pd.to_datetime(sales_top_monthly['Month'].apply(lambda x: str(x)) +'-'+ sales_top_monthly['Year'].apply(lambda x: str(x)),format = '%m-%Y' )
plot_sales(sales_top_monthly.sort_values(by='Month-Year'),'Month-Year','item_cnt_day')
sales_top_weekly = sales_top.groupby(['Week','Year']).agg({'item_cnt_day':'sum'}).reset_index()

sales_top_weekly['Week-Year']= pd.to_datetime('1-' + sales_top_weekly['Week'].apply(lambda x:str(x)) + '-' + sales_top_weekly['Year'].apply(lambda x: str(x)), format = '%w-%W-%Y')
plot_sales(sales_top_weekly.sort_values('Week-Year'),'Week-Year','item_cnt_day', (15,10))
sales_top_weekly.sort_values(by='Week-Year',inplace = True)
plot_acf(sales_top_weekly['item_cnt_day'])
plot_pacf(sales_top_weekly['item_cnt_day'])
sales_top_weekly_diff = sales_top_weekly

sales_top_weekly_diff['Diff sales'] = sales_top_weekly_diff['item_cnt_day'].diff(4)
plot_sales(sales_top_weekly_diff,'Week-Year','Diff sales',size =(15,10))
plot_acf(sales_top_weekly_diff['Diff sales'].dropna())
plot_pacf(sales_top_weekly_diff['Diff sales'].dropna())
monthly_sales = sales_train_df.groupby(['date_block_num','shop_id','item_id']).agg({'date':['min','max'], 'item_price':'mean','item_cnt_day':'sum'}).reset_index()
monthly_sales.head()
item_cat_count = items_df.groupby('item_category_id')['item_id'].agg({'item_id':'count'}).reset_index()
item_cat_count.sort_values(by = 'item_id', ascending = False, inplace = True)
plt.figure()

sns.barplot(item_cat_count.iloc[0:10]['item_category_id'], item_cat_count.iloc[0:10]['item_id'])

plt.title('Items in top 10 categories')

plt.show()