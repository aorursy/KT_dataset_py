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

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from zipfile import ZipFile

zf0 = ZipFile('../input/instacart-market-basket-analysis/order_products__train.csv.zip')
zf1 = ZipFile('../input/instacart-market-basket-analysis/aisles.csv.zip')
zf2 = ZipFile('../input/instacart-market-basket-analysis/orders.csv.zip')
zf3 = ZipFile('../input/instacart-market-basket-analysis/sample_submission.csv.zip')
zf4 = ZipFile('../input/instacart-market-basket-analysis/departments.csv.zip')
zf5 = ZipFile('../input/instacart-market-basket-analysis/products.csv.zip')
zf6 = ZipFile('../input/instacart-market-basket-analysis/order_products__prior.csv.zip')



order_products__train = pd.read_csv(zf0.open('order_products__train.csv'))
aisles                = pd.read_csv(zf1.open('aisles.csv'))
orders                = pd.read_csv(zf2.open('orders.csv'))
sample_submission     = pd.read_csv(zf3.open('sample_submission.csv'))
departments           = pd.read_csv(zf4.open('departments.csv'))
products              = pd.read_csv(zf5.open('products.csv'))
order_products__prior = pd.read_csv(zf6.open('order_products__prior.csv'))


order_products__train.head()


                                    
len(order_products__train)
#order_products__train.groupby('order_id').count()
#1. how many records
#2. how many records by order_id --> 131209 orders #orders by same account cannot be concluded#
#3. Combined with table'orders', it can be conculded that subset of 131209 recrods with 'eval_set = train' can be joint with table 
orders.head()

cnt_srs = orders.eval_set.value_counts()
print(cnt_srs)
#1. counts by eval_set
#*1. link order_product__train to get 'recordered' by order_id *
#subset of orders#
order_train = orders[orders['eval_set'].str.contains('train')]
order_train.head()
#len(order_train)
order_t_wide= pd.merge(order_train,order_products__train,how= 'left', on= ['order_id'])
order_t_wide.head()
len(order_t_wide)
#???
sample_submission.head()

departments.head()
products.head()
aisles.head()
order_products__prior.head()