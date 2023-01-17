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
path = "../input/"

print('load train...')
train_df = pd.read_csv(path+"train.csv",usecols=[ 'click_time'])
print('load test...')
test_df = pd.read_csv(path+"test.csv", usecols=['click_time'])

train_df.click_time.str[11:13].value_counts().sort_index()
test_df.click_time.str[11:13].value_counts().sort_index()
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df = train_df[(train_df.hour == 4)|(train_df.hour == 5)|(train_df.hour == 9)|(train_df.hour == 10)|(train_df.hour == 13)|(train_df.hour == 14)]