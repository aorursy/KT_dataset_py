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
path = '/kaggle/input/m5-forecasting-accuracy'

train_sales = pd.read_csv(f'{path}/sales_train_validation.csv')

calendar = pd.read_csv(f'{path}/calendar.csv')

sub = pd.read_csv(f'{path}/sample_submission.csv')

sell_prices = pd.read_csv(f'{path}/sell_prices.csv')
train_sales
train_sales['dept_id'].unique()   #7 types of item departments
train_sales['item_id'].nunique()
train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_001']
sell_prices.loc[sell_prices['item_id'] == 'HOBBIES_1_001']
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (18,4))

for i in range(10):

    plt.plot(train_sales[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 6:].rolling(30).mean().values, 

             label=train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 5])

plt.title('HOBBIES_1_002 sales,rolling mean over 30 days ')

plt.legend();

#rolling mean to see the mean sales of product over a period

plt.figure(figsize = (18,4))

for i in range(10):

    plt.plot(train_sales[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 6:].rolling(60).mean().values, 

             label=train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 5])

plt.title('HOBBIES_1_002 sales,rolling mean over 60 days')

plt.legend();



plt.figure(figsize = (18,4))

for i in range(10):

    plt.plot(train_sales[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 6:].rolling(90).mean().values, 

             label=train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 5])

plt.title('HOBBIES_1_002 sales,rolling mean over 90 days')

plt.legend();
#the price of item HOBBIES_1_001



s1 = sell_prices.loc[sell_prices['item_id'] == 'HOBBIES_1_003']

for n in s1['store_id'].unique():

    small_df = s1.loc[s1['store_id'] == n]

    plt.plot(s1['wm_yr_wk'], s1['sell_price'], label = n)

plt.legend()
item_prices = sell_prices.loc[sell_prices['item_id'] == 'HOBBIES_2_001']

item_prices['store_id'].unique()
# in a single store

CA_1 = train_sales.loc[train_sales['store_id'] == 'CA_1']

CA_1
CA_1_price = sell_prices.loc[sell_prices['store_id'] == 'CA_1']

CA_1_price
pd.crosstab(CA_1['cat_id'], CA_1['dept_id'])
plt.figure(figsize = (12, 4))

for dep in CA_1['dept_id'].unique():

    store_sales = CA_1.loc[CA_1['dept_id'] == dep]

    store_sales.iloc[:, 6:].sum().rolling(30).mean().plot(label = dep)

plt.title('CA_1 sales by department, 30 days mean')

plt.legend(loc = (1.0, 0.5))
CA_1_prices = sell_prices.loc[sell_prices['store_id'] == 'CA_1']

CA_1_prices

# all prices of all items in CA_1
CA_1_prices['dept_id'] = CA_1_prices['item_id'].apply(lambda x: x[:-4]) #the dept ids from the item ids

CA_1_prices['dept_id']
#price plot for each dept

plt.figure(figsize = (12, 6))

for dep2 in CA_1_prices['dept_id'].unique():

    df_dept_price = CA_1_prices.loc[CA_1_prices['dept_id'] == dep2]

    grouped = df_dept_price.groupby(['wm_yr_wk'])['sell_price'].mean()

    plt.plot(grouped.index, grouped.values, label = dep2)

plt.legend(loc =  (1.0, 0.5))

plt.title('CA_1 mean sell price by departemnt by week')
# all info of a single dept

hob1_sales = train_sales.loc[train_sales['dept_id'] == 'HOBBIES_1']

hob1_sales
hob1_sales['item_id'].nunique()
hob1_sell_price = sell_prices.loc[sell_prices['item_id'].str.contains('HOBBIES_1')] #sell_prices only has the item ids

hob1_sell_price
plt.figure(figsize = (12, 6))

for n in hob1_sales['store_id'].unique():

    store_sales = hob1_sales.loc[hob1_sales['store_id'] == n]

    store_sales.iloc[:, 6:].sum().rolling(30).mean().plot(label = n)

plt.title('HOBBIES_1 sale by stores, 30')

plt.legend(loc = (1.0, 0.5))
sell_prices
hob1_sell_price = sell_prices.loc[sell_prices['item_id'].str.contains('HOBBIES_1')]

plt.figure(figsize = (12, 6))

for n in hob1_sell_price['store_id'].unique():

    hob1df = hob1_sell_price.loc[hob1_sell_price['store_id'] == n]

    grouped = hob1df.groupby(['wm_yr_wk'])['sell_price'].mean()

    plt.plot(grouped.index, grouped.values, label = n)

plt.legend(loc = (1.0,0.5))

plt.title('HOBBIES_1 mean sell prices by store')
state_CA = train_sales.loc[train_sales['state_id'] == 'CA']

state_CA
for col in ['item_id', 'dept_id', 'store_id']:

    print(f"{col} has {train_sales.loc[train_sales['state_id'] == 'CA', col].nunique()} unique values for CA state")
CA_sales = train_sales.loc[train_sales['state_id'] == 'CA']

plt.figure(figsize = (12, 6))

for n in CA_sales['store_id'].unique():

    store_sales = CA_sales.loc[CA_sales['store_id'] == n]

    store_sales.iloc[:, 6: ].sum().rolling(30).mean().plot(label = n)

plt.title('CA sales by store, rolling mean 30 days')

plt.legend(loc = (1.0, 0.5))

plt.figure(figsize = (12, 6))

for n in CA_sales['store_id'].unique():

    store_sales = CA_sales.loc[CA_sales['store_id'] == n]

    store_sales.iloc[:, 6: ].sum().rolling(60).mean().plot(label = n)

plt.title('CA sales by store, rolling mean 60 days')

plt.legend(loc = (1.0, 0.5))
#relation between sales number and weekdays, events, snaps

#train_sales and calendar

calendar
#See the HOBBIES_1 dept sales in CA_1

sales_CA1_by_wd = (train_sales[(train_sales['store_id'] == 'CA_1' )&(train_sales['dept_id'] == 'HOBBIES_1') ].iloc[:, 6:]).T

sales_CA1_by_wd

#need a reverse transformation, col to row, combine the two dfs

sum_of_day = sales_CA1_by_wd.sum(axis = 1)

sum_of_day.shape
# sales_CA1_by_wd2 = (train_sales[(train_sales['store_id'] == 'CA_1' )&(train_sales['item_id'] == 'HOBBIES_1_001') ].iloc[:, 6:]).T

# sales_CA1_by_wd2 # one particular item in one particular store sale 

#                  # Can we make a plot out of this?
sum_H1_001_Sat = sales_CA1_by_wd2[:].sum()

sum_H1_001_Sat
wds = calendar[['weekday']].iloc[:1913].reset_index(drop = True)



sales_CA1_by_wd = sales_CA1_by_wd.append(sum_of_day, ignore_index = True)
#weekday in calendar

wds.index = sales_CA1_by_wd.index

sales_weekdays =pd.concat([wds, sales_CA1_by_wd], axis = 1, ignore_index = True)

sales_weekdays #

# Store CA_1, dept HOBBIES_1 sales, with no. of days and day of the week
s_index = sales_weekdays.index

for n in s_index:

    sum_of_day_1 = sales_weekdays[n].sum(axis = 1)
# sales_weekdays 

#    tmp = calendar[['weekday']].iloc[:1913].copy()   to get rid of the ori index, copy()

#    tmp.reset_index(drop = True)
sales_weekdays.index
sales_weekdays[0] #index are dropped
sales_weekdays[sales_weekdays[0]=='Saturday'].shape
# #Sales plot of HOBBIES_1 in CA_1

# plt.figure(figsize = (12, 6))

# for n in sales_weekdays.index:  #d_1 to d_1913

    

#         sat = sales_weekdays[sales_weekdays[0]== 'Saturday' or sales_weekdays[0]== 'Sunday'].iloc[:, 1:].sum().plot(label = n)

# # plt.title('Saturday sales of HOBBIES_1 in CA_1')



    
from sklearn.model_selection import StratifiedKFold