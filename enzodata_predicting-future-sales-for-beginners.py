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
# train and test

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
print(train.shape)

print(test.shape)
# I would like to see the top 10 values

train.head(10)
# I would like to see the last 10 training values

train.tail(10)
# I would like to see the top 10 test values

test.head(10)
# I would like to see the last 10 test values

test.tail(10)
# shop.csv

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv');

print(shops.shape)

shops.head()
# item.csv

items= pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv');

print(items.shape)



items.head(10)

# item_categories.csv

item_categories= pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv');

print(item_categories.shape)

item_categories.head()

# sample_submission.csvの

sample_submission= pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv');

print(sample_submission.shape)

sample_submission.head(10)