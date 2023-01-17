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
train = pd.read_csv('../input/train.csv')
target = train.groupby('USER_ID')['ITEM_CODE'].nth(-1)
train['val'] = 1
pt = pd.pivot_table(train, index='USER_ID', columns='ITEM_CODE', values='val')
pt.head()
train['TIME'] = pd.to_datetime(train.OPERATION_TIME, format="%Y-%m-%d %H:%M:%S")
from datetime import datetime

datetime_object = datetime.strptime('2002-06-01 00:00:00', "%Y-%m-%d %H:%M:%S")
jun = train[train.TIME >= datetime_object]
valid_ans = jun.groupby('USER_ID')['ITEM_CODE'].unique()
valid_ans = pd.DataFrame(valid_ans.apply(lambda x: ' '.join([str(i) for i in x])))
valid_ans = valid_ans.reset_index()
valid_ans = valid_ans.rename(index=str, columns={'ITEM_CODE':'ITEM_CODES'})
valid_ans.head()
valid_ans.shape
sam_sub = pd.read_csv('../input/sample_submission.csv')
sub = sam_sub.join(valid_ans.set_index('USER_ID'), on='USER_ID', rsuffix='_right').copy()
sub.head()
sub.ITEM_CODES = sub['ITEM_CODES_right'].fillna(sub['ITEM_CODES'])
sub.head()
sub[['USER_ID','ITEM_CODES']].to_csv('sub.csv', index=False)

