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
def spcl(item_id):

    result = sp[sp['item_id'].isin(item_id)]

    result = pd.merge(result,cl,on='wm_yr_wk')

    result['date'] = pd.to_datetime(result['date'])

    return result



spcl(['HOBBIES_1_001','HOBBIES_1_002'])
fig, ax = plt.subplots(figsize=(20, 10))

sns.lineplot(data=spcl(['HOBBIES_1_001']),x='date',y='sell_price',hue='store_id',ax=ax)
grouped_sp = sp[['item_id','sell_price']].groupby('item_id')

summary_sp = grouped_sp.agg(['mean','max','min','std','var','count'])['sell_price']

summary_sp
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(20, 10))

for i, column in enumerate(summary_sp.columns):

    sns.distplot(summary_sp[column],ax=axes[i//3,i%3])
summary_sp1 = summary_sp.copy()

summary_sp1['item_id'] = summary_sp.index

summary_sp1[['cat_id','dept','item']] = summary_sp1['item_id'].str.split('_',expand=True)

summary_sp1['dept_id'] = summary_sp1['cat_id'] + '_' + summary_sp1['dept']

summary_sp1
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(20, 10))

for i, column in enumerate(['mean', 'max', 'min', 'std', 'var', 'count']):

    sns.distplot(summary_sp1[summary_sp1['cat_id']=='FOODS'][column],ax=axes[i//3,i%3],label='FOODS')

    sns.distplot(summary_sp1[summary_sp1['cat_id']=='HOBBIES'][column],ax=axes[i//3,i%3],label='HOBBIES')

    sns.distplot(summary_sp1[summary_sp1['cat_id']=='HOUSEHOLD'][column],ax=axes[i//3,i%3],label='HOUSEHOLD')



plt.legend()

fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(20, 10))

for i, column in enumerate(['mean', 'max', 'min', 'std', 'var', 'count']):

    sns.distplot(summary_sp1[summary_sp1['cat_id']=='FOODS'][column],ax=axes[i//3,i%3],label='FOODS')



plt.legend()

fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(20, 10))

for i, column in enumerate(['mean', 'max', 'min', 'std', 'var', 'count']):

    sns.distplot(summary_sp1[summary_sp1['cat_id']=='HOBBIES'][column],ax=axes[i//3,i%3],label='HOBBIES')



plt.legend()

fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(20, 10))

for i, column in enumerate(['mean', 'max', 'min', 'std', 'var', 'count']):

    sns.distplot(summary_sp1[summary_sp1['cat_id']=='HOUSEHOLD'][column],ax=axes[i//3,i%3],label='HOUSEHOLD')



plt.legend()

import matplotlib.pyplot as plt

for dept_id in summary_sp1['dept_id'].unique():

    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(20, 10))

    for i, column in enumerate(['mean', 'max', 'min', 'std', 'var', 'count']):

            sns.distplot(summary_sp1[summary_sp1['dept_id']==dept_id][column],ax=axes[i//3,i%3])

    fig.suptitle(dept_id, fontsize=20)

    plt.show()



import matplotlib.pyplot as plt

def show_distplot(summary_sp1):

    for dept_id in summary_sp1['dept_id'].unique():

        for i, column in enumerate(['mean', 'max', 'min', 'std', 'var', 'count']):

            sns.distplot(summary_sp1[summary_sp1['dept_id']==dept_id][column],ax=axes[i//3,i%3])

    



for cat_id in summary_sp1['cat_id'].unique():

    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(20, 10))

    show_distplot(summary_sp1[summary_sp1['cat_id']==cat_id])

    fig.suptitle(cat_id, fontsize=20)

    plt.show()


