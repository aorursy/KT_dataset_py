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
merchants = pd.read_csv('../input/merchants.csv', index_col='merchant_id')
merchants.head()
print(merchants.info())
print(merchants.memory_usage(deep=True))
float_cols = merchants.select_dtypes(include=['float'])
#print(float_cols.dtypes)


for cols in float_cols.columns:
	merchants[cols] = pd.to_numeric(merchants[cols], downcast ='float')

print(merchants.info())
merchants.head()
int_cols = merchants.select_dtypes(include=['int'])


for i in int_cols.columns:
	merchants[i] = pd.to_numeric(merchants[i], downcast ='integer')

print(merchants.info())
merchants.head()
for cols in ['category_1', 'category_4', 'most_recent_purchases_range', 'most_recent_sales_range']:
	merchants[cols] = merchants[cols].astype('category')

print(merchants.info())
print(merchants.memory_usage(deep=True))
