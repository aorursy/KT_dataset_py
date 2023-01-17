import matplotlib.pyplot as plt

import seaborn as sns

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
sp = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

cl = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

stv = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
stv
stv.groupby('store_id').count()[['id']]
stv.groupby('item_id').count()[['id']]
stv[stv['store_id']=='CA_1'].groupby('dept_id').count()[['id']]
sns.countplot(data=stv[stv['store_id']=='CA_1'],y='dept_id')
stv[stv['store_id']=='CA_1'].groupby('cat_id').count()[['id']]
sns.countplot(data=stv[stv['store_id']=='CA_1'],y='cat_id')