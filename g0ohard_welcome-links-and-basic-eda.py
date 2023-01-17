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
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
test.head()
train.head()
import calendar
import datetime
train['YEAR_MONTH'] = train.OPERATION_TIME.str.slice(start = 0, stop = 7)
train.groupby('YEAR_MONTH')['OPERATION_CODE'].nunique().plot(kind='bar', figsize=(16, 9), fontsize=20);
train_users = train.USER_ID.unique()
len(train_users)
test_users = test.USER_ID.unique()
len(test_users)
train_test_users = np.intersect1d(test_users, train_users)
len(train_test_users)
train[train.USER_ID.isin(train_test_users)].groupby('YEAR_MONTH')['OPERATION_CODE'].nunique().plot(kind='bar', 
                                                                                                   figsize=(16, 9), 
                                                                                                   title='Количество операций выполненных пользователями, которые есть и в трейне и в тесте', 
                                                                                                   fontsize = 20);
train[train.USER_ID.isin(train_test_users)].groupby('PLACE_ID')['OPERATION_CODE'].nunique().plot(kind='bar', 
                                                                                                   figsize=(16, 9), 
                                                                                                   title='Количество операций выполненных в разных "местах"');
train[train.USER_ID.isin(train_test_users)].groupby('GROUP_LVL1')['OPERATION_CODE'].nunique().plot(#kind='bar', 
                                                                                                   figsize=(16, 9), 
                                                                                                   title='Количество операций выполненных с разными группами товаров');
train[train.USER_ID.isin(train_test_users)].groupby('MONEY')['OPERATION_CODE'].nunique().plot(kind='bar', 
                                                                                                   figsize=(16, 9), 
                                                                                                   title='Количество операций выполненных на различную сумму');
train.USER_ID.value_counts().head()
user_id = 12383
train[train.USER_ID == user_id].groupby('YEAR_MONTH')['OPERATION_CODE'].nunique().plot(kind='bar', 
                                                                                       figsize=(16, 9), 
                                                                                       title='В какие месяца пользователь ' + str(user_id) + ' делал операции', 
                                                                                       fontsize = 20);
top_users = train.USER_ID.value_counts().head(100)
train[train.USER_ID.isin(top_users)].groupby('YEAR_MONTH')['OPERATION_CODE'].nunique().plot(kind='bar', 
                                                                                       figsize=(16, 9), 
                                                                                       title='В какие месяца топ пользователи делали операции', 
                                                                                       fontsize = 20);
#2000-12-23 15:20:00
train['TIME'] = pd.to_datetime(train['OPERATION_TIME'], format='%Y-%m-%d %H:%M:%S')
train[(train.TIME.dt.year < 2002)].groupby('YEAR_MONTH')['OPERATION_CODE'].nunique().plot(kind='bar', 
                                                                                       figsize=(16, 9), 
                                                                                       title='Операции в 2001\' году', 
                                                                                       fontsize = 20);
train.ITEM_CODE.value_counts().head(15)
item_id = 61936
train[train.ITEM_CODE == item_id].groupby('YEAR_MONTH')['OPERATION_CODE'].nunique().plot(kind='bar', 
                                                                                       figsize=(16, 9), 
                                                                                       title='В какие месяца товар ' + str(item_id) + ' покупался', 
                                                                                       fontsize = 20);
train[(train.TIME.dt.month == 12)].ITEM_CODE.value_counts().head(15)
item_id = 16411
train[train.ITEM_CODE == item_id].groupby('YEAR_MONTH')['OPERATION_CODE'].nunique().plot(kind='bar', 
                                                                                       figsize=(16, 9), 
                                                                                       title='В какие месяца товар ' + str(item_id) + ' покупался', 
                                                                                       fontsize = 20);
item_id = 33652
train[train.ITEM_CODE == item_id].groupby('YEAR_MONTH')['OPERATION_CODE'].nunique().plot(kind='bar', 
                                                                                       figsize=(16, 9), 
                                                                                       title='В какие месяца товар ' + str(item_id) + ' покупался', 
                                                                                       fontsize = 20);
addiotion_predictions = ' '.join([str(x) for x in list(train.ITEM_CODE.value_counts().head(25).index.values)])
addiotion_predictions
predictions_from_sample_sub = ' '.join([str(x) for x in list(sample_submission.head(1).ITEM_CODES.value_counts().head(25).index.values)])
predictions_from_sample_sub
sumple_sub_exted = predictions_from_sample_sub + ' ' + addiotion_predictions
sumple_sub_exted
sample_submission_extended = sample_submission.copy()
sample_submission_extended.ITEM_CODES = sumple_sub_exted
sample_submission_extended.head()
sample_submission_extended.to_csv('sample_submission_extended.csv', index=False)