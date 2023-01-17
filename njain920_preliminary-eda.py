# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Importing all the required modules
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
import matplotlib.dates as mdates
#Reading the data
cal_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
print(cal_data.shape)
print(prices.shape)
print(sales.shape)
#Viewing the first five rows of sales data
sales.head() 
print('There are {0} items '.format(len(sales['item_id'].unique())))
print('There are {0} depts'.format(len(sales['dept_id'].unique())))
print('There are {0} categories'.format(len(sales['cat_id'].unique())))
print('There are {0} stores'.format(len(sales['store_id'].unique())))
print('There are {0} states'.format(len(sales['state_id'].unique())))
#Copying the sales dataframe so that modifications can be made and the original dataframe be kept intact
sales_df = sales.copy()
date_list = [d.strftime('%Y-%m-%d') for d in pd.date_range(start = '2011-01-29', end = '2016-04-24')]
#Renaming days to dates
sales_df.rename(columns=dict(zip(sales_df.columns[6:], date_list)),inplace=True)
sales_df.head()
#Aggregating by mean the sales by department
dept_mean = sales_df.groupby(['dept_id']).mean().T
dept_mean.index = pd.to_datetime(dept_mean.index)

#Aggregating by mean the sales by categories
cat_mean = sales_df.groupby(['cat_id']).mean().T
cat_mean.index = pd.to_datetime(cat_mean.index)

#Aggregating by mean the sales by stores
store_mean = sales_df.groupby(['store_id']).mean().T
store_mean.index = pd.to_datetime(store_mean.index)

#Aggregating by mean the sales by states
state_mean = sales_df.groupby(['state_id']).mean().T
state_mean.index = pd.to_datetime(state_mean.index)

#Function for creating plots
def create_plots(df,freq):
    fig, ax = plt.subplots()
    for i in df.columns:
        df_plot = df[i].resample(freq).sum()
        df_plot.plot(ax=ax)
        fig.set_figheight(7)
        fig.set_figwidth(15)
    plt.grid(True)
    ax.legend(df.columns,loc='best')
#Plotting the mean data
create_plots(dept_mean,'m')
create_plots(cat_mean,'m')
create_plots(store_mean,'m')
create_plots(state_mean,'m')
#To plot data in a particular date range
fig, ax = plt.subplots(figsize=(15,5))
state_mean.plot(xlim=['2012-01-01','2014-01-01'],ax=ax,rot=90)
plt.grid(True)
plt.xlabel('Sales by State')
# set ticks every week
ax.xaxis.set_major_locator(mdates.MonthLocator())
# #set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
cal_data.head(31)
print(cal_data['event_name_1'].notnull().sum())
print(cal_data['event_name_2'].notnull().sum())
print(len(cal_data['event_name_1'].unique()))
print(len(cal_data['event_type_1'].unique()))
prices.head()
prices['sell_price'].hist(bins=50)
plt.xlim(0,25)
#Checking the price range of each department
prices[(prices['item_id'].str.startswith('FOODS_1'))]['sell_price'].hist()
plt.xlabel('FOODS_1')
plt.show()
prices[(prices['item_id'].str.startswith('FOODS_2'))]['sell_price'].hist()
plt.xlabel('FOODS_2')
plt.show()
prices[(prices['item_id'].str.startswith('FOODS_3'))]['sell_price'].hist()
plt.xlabel('FOODS_3')
plt.show()
prices[(prices['item_id'].str.startswith('HOUSEHOLD_1'))]['sell_price'].hist()
plt.xlabel('HOUSEHOLD_1')
plt.show()
prices[(prices['item_id'].str.startswith('HOUSEHOLD_2'))]['sell_price'].hist()
plt.xlabel('HOUSEHOLD_2')
plt.show()
prices[(prices['item_id'].str.startswith('HOBBIES_1'))]['sell_price'].hist()
plt.xlabel('HOBBIES_1')
plt.show()
prices[(prices['item_id'].str.startswith('HOBBIES_2'))]['sell_price'].hist()
plt.xlabel('HOBBIES_2')
plt.show()
#Get the average selling price of each item
avg_price = prices.groupby(['item_id'])['sell_price'].mean()
#Merge it with sales data
merged = pd.merge(sales_df,avg_price, right_index=True, left_on='item_id')
#Group the merged that by id 
id_grouped = merged.groupby(['id']).sum()
#Sum by days to get total quantity
id_grouped['Total_Qty'] = id_grouped.sum(axis=1)
#Get the total amount sold by multiplying the total quantity and selling price
id_grouped['Amount_Sold'] = id_grouped['Total_Qty'] * id_grouped['sell_price']
#Remove duplicate columns to merge data with sales
cols_to_use = id_grouped.columns.difference(sales_df.columns)
#Store the final df in new_sales
new_sales = pd.merge(sales_df,id_grouped[cols_to_use], right_index=True, left_on='id')
new_sales.groupby(['dept_id','store_id'])['Total_Qty'].agg('mean').unstack().plot(kind='bar',figsize=(15,7))
plt.title('Mean Quantity Sold by Department in each store')
new_sales.groupby(['dept_id','store_id'])['Total_Qty'].agg('mean').unstack().T.plot(kind='bar',figsize=(15,7))
plt.title('Mean Quantity Sold by Each Store of each Department')
WI_2 = sales_df[(sales_df['store_id'] == 'WI_2')]
dept_WI2 = WI_2.groupby(['dept_id']).sum().T
dept_WI2.index = pd.to_datetime(dept_WI2.index)
dept_WI2.head()

CA_2 = sales_df[(sales_df['store_id'] == 'CA_2')]
dept_CA2 = CA_2.groupby(['dept_id']).sum().T
dept_CA2.index = pd.to_datetime(dept_CA2.index)
dept_CA2.head()

fig, ax = plt.subplots(figsize=(15,5))
dept_CA2.plot(xlim=['2015-01-01','2016-01-01'],ax=ax,rot=90)
plt.grid(True)
plt.xlabel('Sales by Category')
# set ticks every week
ax.xaxis.set_major_locator(mdates.MonthLocator())
#set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.title('CA Plot 2015-16 ')
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
dept_WI2.plot(xlim=['2012-01-01','2013-01-01'],ax=ax,rot=90)
plt.grid(True)
plt.xlabel('Sales by Category')
# set ticks every week
ax.xaxis.set_major_locator(mdates.MonthLocator())
#set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.title('WI Plot 2012-13 ')
plt.show()
