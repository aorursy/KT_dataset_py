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
items_df=pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

shops_df=pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

sales_train_df=pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test_df=pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

sample_sub_df=pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

item_categ_df=pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
items_df
shops_df.head()
shops_df.shape
test_df
item_categ_df
sales_train_df
train_corr=sales_train_df.corr()

train_corr

sales_train_df.describe()