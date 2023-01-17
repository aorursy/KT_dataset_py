# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
train_data.head()
train_data.info()
test_data.info()
sns.pairplot(train_data)
train_data.columns
dataset = train_data.pivot_table(index = ['shop_id','item_id'],columns = ['date_block_num'],values = ['item_cnt_day'],fill_value = 0)
dataset.reset_index(inplace = True)
dataset = pd.merge(test_data,dataset,how = 'left',on = ['shop_id','item_id'])
dataset.head()
dataset.fillna(0,inplace = True)

dataset.head()
submission_pfs = dataset.iloc[:,36]
submission_pfs.clip(0,20,inplace = True)

submission_pfs = pd.DataFrame({'ID':test_data['ID'],'item_cnt_month':submission_pfs.ravel()})
submission_pfs.head()
submission_pfs.to_csv('pfs.csv',index = False)
g = pd.read_csv('pfs.csv')
g.head()