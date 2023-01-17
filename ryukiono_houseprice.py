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
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv",sep= ',')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',sep=',')
sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.loc[train['MSZoning']=='RL','MSZoning_RL']=1
train.loc[train['MSZoning']=='RM','MSZoning_RM']=1
train.loc[train['MSZoning']=='C(all)','MSZoning_C']=1
train.loc[train['MSZoning']=='FV','MSZoning_FV']=1
train.loc[train['MSZoning']=='RH','MSZoning_RH']=1

train['MSZoning_RL']=train.loc[:,'MSZoning_RL'].fillna(0)
train['MSZoning_RM']=train.loc[:,'MSZoning_RM'].fillna(0)
train['MSZoning_C']=train.loc[:,'MSZoning_C'].fillna(0)
train['MSZoning_FV']=train.loc[:,'MSZoning_FV'].fillna(0)
train['MSZoning_RH']=train.loc[:,'MSZoning_RH'].fillna(0)

train.head()
train.drop(['MSZoning'],axis=1,inplace=True)
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv",sep= ',')
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit_transform(train.MSZoning)
pd.concat([train,pd.DataFrame(lb.transform(train.MSZoning),columns=lb.classes_)],axis=1)
