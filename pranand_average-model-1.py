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
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
train_data = pd.read_csv('../input/train.csv')
pp = train_data.groupby(['YearBuilt']).mean()['SalePrice'].reset_index()
pp.head()
pp1 = pp.sort_values(by='YearBuilt', ascending=False)
pp1.head()
test_data = pd.read_csv('../input/test.csv')
submisison_file = pd.read_csv('../input/sample_submission.csv')
test_data.head()
train_data['SalePrice'].mean()
submisison_file.head()
submisison_file['SalePrice'] = train_data.SalePrice.mean()
submisison_file.to_csv('first_submission_avg.csv', index=False)
def new_rules(poolarea, salecondition, lotarea, mssubclass, basement):
    if (basement<2200) and (mssubclass<50) and (lotarea<6000):
        return 'condition_1'
    else:
        return 'condition_2'
print(train_data.columns)
train_data['newCol'] = train_data[['PoolArea', 'SaleCondition', 
                                   'LotArea', 'MSSubClass',
                                   'BsmtFinSF1']].apply(lambda x:new_rules(*x), axis=1)
op = train_data.groupby('newCol').mean()['SalePrice'].reset_index()
op
op_dict = dict(zip(op['newCol'], op['SalePrice']))
test_data['newCols'] = test_data[['PoolArea', 'SaleCondition', 
                                   'LotArea', 'MSSubClass',
                                   'BsmtFinSF1']].apply(lambda x:new_rules(*x), axis=1)
test_data['price'] = test_data['newCols'].map(lambda x: op_dict[x])
test_data_dict = dict(zip(test_data['Id'], test_data['price']))
submisison_file['SalePrice'] = submisison_file['Id'].map(lambda x: test_data_dict[x])
submisison_file.to_csv('submission_2.csv', index=False)
