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
import seaborn as sns

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
data = pd.read_csv('../input/train.csv')

data.head()
data.describe()
data_y=data['target']

data_X=data.drop(['target', 'ID_code'], axis=1)
import lightgbm as lgb

from sklearn.model_selection import train_test_split

param = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.01, 'num_rounds': 6000, 'verbose': 1}

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=101)

train_set = lgb.Dataset(X_train, y_train)

val_set = lgb.Dataset(X_test, y_test)

model=lgb.train(param,train_set,valid_sets=val_set,verbose_eval=500)

test = pd.read_csv('../input/test.csv')

test_id=test.ID_code

test_df=test.drop('ID_code',axis=1)

test_pre= model.predict(test_df)

test_final=np.array(test_pre>0.5,dtype='int')
submit = pd.read_csv('../input/sample_submission.csv')

submit['ID_code']=test_id;

submit['target']=test_final;

submit.to_csv('submission.csv',index=False)