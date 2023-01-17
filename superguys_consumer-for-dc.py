# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_agg = pd.read_csv('../input/train//train_agg.csv',sep='\t')
train_log = pd.read_csv('../input/train//train_log.csv',sep='\t',parse_dates=['OCC_TIM'])

test_agg = pd.read_csv('../input/test//test_agg.csv',sep='\t')
test_log = pd.read_csv('../input/test//test_log.csv',sep='\t',parse_dates=['OCC_TIM'])
train_flag = pd.read_csv('../input/train//train_flg.csv',sep='\t')
train = pd.merge(train_agg,train_log,on='USRID',how='left')
test = pd.merge(test_agg,test_log,on='USRID',how='left')
del train_agg,train_log,test_agg,test_log
train = pd.merge(train,train_flag,on='USRID',how='left')
test['FLAG'] = -1
train.head()
train.loc[train['USRID']==0]
train.groupby('USRID')['FLAG'].sum().value_counts()[0]

a = train['EVT_LBL'].str.split('-',expand=True)
train.groupby(train['EVT_LBL'])['FLAG'].mean().plot.bar(figsize=(12,6))
EVT_LBL = df['EVT_LBL'].str.split('-',expand=True)
EVT_LBL.columns = ['meal','voucher','details']
train.shape
train['USRID'].value_counts().sort_values()


