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
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
submission_data = pd.read_csv("../input/sample_submission.csv")
train_data.head
train_data.head()
train_data.describe()
train_data.SalePrice.mean()
submission_data.head()
submission_data['SalePrice'] = train_data.SalePrice.mean()
submission_data.to_csv('avg_submission.csv', index=False)
year_data = train_data.groupby(['YearBuilt']).mean()['SalePrice'].reset_index()
year_data
year_data['lagPrice'] = year_data['SalePrice'].shift(-1)
year_data.head()
year_data['percentage_change'] = (year_data['SalePrice'] - year_data['lagPrice']) / year_data['lagPrice']
year_data.head()
year_data['avg_price'] = (year_data['SalePrice']) + (year_data['lagPrice'] * year_data['percentage_change']/100)
year_data.head()
# Merge two column based on inner join
combinedYear = pd.merge(train_data, year_data, how='inner', on = 'YearBuilt')

# Drop any column form the dataframe
combinedYear.head()
combinedYear.shape
submission_data.shape
year_data.shape
submission_data['SalePrice'] = combinedYear.SalePrice_y
submission_data.head()
submission_data.shape
submission_data.to_csv('avg_submission_03.csv', index=False)