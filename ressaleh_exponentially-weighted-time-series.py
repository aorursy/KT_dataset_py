# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime # manipulating date formats
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
%matplotlib inline 

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
DATA_FOLDER = '../input/'

transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
test_data    = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
submission    = pd.read_csv(os.path.join(DATA_FOLDER, 'sample_submission.csv'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
#formatting the date column correctly
transactions.date=transactions.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
sales = pd.merge(test_data, transactions, how='left', on=['shop_id', 'item_id'])
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales of the 42 shops')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts);
N = 11
months = [33,32,31,30,22,29,28,27,26,25,24]
temp3 = test_data
temp3.head()
for month_blk in months:
    #Grab the next month of data (work backwards from present=33 to beginning=0)
    train_df = transactions[transactions['date_block_num'] == month_blk]
    #Replace item_cnt_day by item_cnt_month because thats what we need later
    train_df = train_df.rename(columns={'item_cnt_day':'item_cnt_month'+str(month_blk)})
    #create a dummny result to initialize the inner loop
    result = pd.DataFrame({'shop_id': 0, 'item_id': 5037, 'item_cnt_month'+str(month_blk): 0.0 }, index=[0])
    print(month_blk,result)
    #loop through all shops from 0 to 59
    for i in range(60):
        #Grab all data for shop i in current month
        temp1 = train_df[train_df['shop_id'] == i]
        #Add up all the items for that shop for each item_id so now it's a montly total
        df = temp1[['item_id', 'item_cnt_month'+str(month_blk)]].groupby(['item_id'], as_index=False).sum()
        #Backfill the shop_id so that it's consistent with test_data
        df['shop_id'] = i
        #Concatenate the results of this shop with shops processed so far
        frames = [result, df]
        result = pd.concat(frames,sort=True)
    #Look at the results
    result.head()
    #Construct a temp3 dataFrame with the results and fill in NaN with 0.0
    temp3 = pd.merge(temp3, result, how='left', on=['shop_id', 'item_id']).fillna(0.15)
    temp3.describe()
    #Clip the results between [0,20] to improve accuracy by removing outliers
    temp3.loc[(temp3['item_cnt_month'+str(month_blk)] > 20.0),'item_cnt_month'+str(month_blk)] = 20.0
    temp3.loc[(temp3['item_cnt_month'+str(month_blk)] < 0.0),'item_cnt_month'+str(month_blk)] = 0.0
alpha = np.zeros(N)
for i in range(N):
    alpha[N-i-1] = 2.0**i/(2.0**N-1)
print(alpha)

temp3['item_cnt_month'] = 0.0
i = 0
for month_blk in months:
    temp3['item_cnt_month'] = temp3['item_cnt_month'] + temp3['item_cnt_month'+str(month_blk)]*alpha[i]
    i += 1
submission = temp3.drop(['shop_id','item_id'], axis=1)
for month_blk in months:
    submission = submission.drop('item_cnt_month'+str(month_blk), axis=1)
submission.head()
submission.describe()
submission['item_cnt_month'] = submission['item_cnt_month']*0.77
submission.describe()
submission['item_cnt_month'].sum()
submission[['ID','item_cnt_month']].to_csv('sample_submission1.csv', index=False)
submission.shape