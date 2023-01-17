# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
history = pd.read_csv("../input/historical_transactions.csv", low_memory=True)
history.shape
history['purchase_amount'].hist(bins=100)
history['purchase_amount'].plot.box()
history=history[history['purchase_amount']<5000000]
history['purchase_amount'].hist()
history[history['purchase_amount']>25000]['purchase_amount'].hist()
history[history['purchase_amount']<20000]['purchase_amount'].hist()
history[history['purchase_amount']<2500]['purchase_amount'].hist()
history[history['purchase_amount']<500]['purchase_amount'].hist()
history[history['purchase_amount']<100]['purchase_amount'].hist()
history[history['purchase_amount']<10]['purchase_amount'].hist()
history[history['purchase_amount']<1]['purchase_amount'].hist()
history[history['purchase_amount']<1].shape
history[history['purchase_amount']<1].shape[0]/history.shape[0]
history[history['purchase_amount']>1]['purchase_amount'].hist()
history[(history['purchase_amount']>1) & (history['purchase_amount']<10)]['purchase_amount'].hist()
history[(history['purchase_amount']>10) & (history['purchase_amount']<100)]['purchase_amount'].hist()
history[(history['purchase_amount']>100) & (history['purchase_amount']<1000)]['purchase_amount'].hist()
history[(history['purchase_amount']>10000) & (history['purchase_amount']<100000)]['purchase_amount'].hist()
bins = [-1,1,10,100,1000,10000,100000,1000000,10000000]
history['binned_purchase_amount'] = pd.cut(history['purchase_amount'], bins)
binned_purchase_amount_cnt = history.groupby("binned_purchase_amount")['binned_purchase_amount'].count().reset_index(name='binned_purchase_amount_cnt')
binned_purchase_amount_cnt.columns = ['purchase_amount','binned_purchase_amount_cnt']
binned_purchase_amount_cnt['percent']=binned_purchase_amount_cnt['binned_purchase_amount_cnt']*100/binned_purchase_amount_cnt['binned_purchase_amount_cnt'].sum()
binned_purchase_amount_cnt
binned_purchase_amount_cnt.plot.bar(y='percent', x='purchase_amount')
def log_1(x):
    return math.copysign(1,x)*math.log(1+abs(x))

history['log_purchase_amount']=history['purchase_amount'].apply(lambda x :log_1(x) )
history['log_purchase_amount'].hist()
history['purchase_amount_outliers'] = 0
history.loc[history['purchase_amount'] >1, 'purchase_amount_outliers'] = 1
