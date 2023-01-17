import calendar

import datetime

import numpy as np

import pandas as pd

import os

import itertools
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

users = pd.read_csv("../input/users.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")
train['YEAR_MONTH'] = train.OPERATION_TIME.str.slice(start = 0, stop = 7)

train['MONTH'] = train.OPERATION_TIME.str.slice(start = 5, stop = 7)

train['OPERATION_TIME'] = pd.to_datetime(train['OPERATION_TIME'])
train['MONTH'] = train.YEAR_MONTH.str.slice(start = 5, stop = 7) 

train['YEAR'] = train.YEAR_MONTH.str.slice(start = 0, stop = 4) 

train.drop('YEAR_MONTH', 1) 

winter = ['12', '01', '02'] 

wint_list = train.loc[train['MONTH'].isin(winter)] 

win_list = wint_list["ITEM_CODE"].sort_values().unique().tolist() 

nondecember = train.loc[~train['MONTH'].isin(winter)] 

nondec_list = nondecember["ITEM_CODE"].sort_values().unique().tolist() 

winter_only = set(win_list) - set(nondec_list)

len(winter_only)
sample_submission.USER_ID.tolist()

users_list = np.intersect1d(sample_submission.USER_ID.tolist(),

                 train.USER_ID.drop_duplicates().tolist())



items_list = train[train['USER_ID'].isin(users_list)].sort_values(by=['USER_ID']).ITEM_CODE.drop_duplicates().tolist()

len(items_list)
items_list = set(items_list) - set(winter_only)

items_list = list(items_list)

len(items_list)
users_dataframe = pd.DataFrame(np.intersect1d(sample_submission.USER_ID

                                                               .tolist(),

                                                                train.USER_ID

                                                                     .drop_duplicates()

                                                                     .tolist()))



items_dataframe = pd.DataFrame(items_list, columns = ['ITEM_CODE'])

#items_dataframe = set(items_dataframe) - set(items_list)



users_dataframe['key'] = 1

items_dataframe['key'] = 1
items_dataframe.count()
some_data = pd.merge(users_dataframe,

                     items_dataframe,

                     on = 'key').drop(columns=['key'])
some_data.columns = ['USER_ID','ITEM_CODE']

some_data

some_data.to_csv('users_items.csv', index=False)