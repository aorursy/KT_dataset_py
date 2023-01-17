# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
shops= pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

items= pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

item_categories=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

train_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')





print(shops.info())

print(shops.head())

print(shops.describe())

print(items.info())

print(items.head())

print(items.describe())



print(item_categories.info())

print(item_categories.head())

print(item_categories.describe())



print(train_data.info())

print(train_data.head())

print(train_data.describe())
print('# Null values')

print(train_data.isnull().sum())