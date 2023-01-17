# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/heart.csv')

data.head()
msk=np.random.rand(len(data))<0.8

train_df=data[msk]

test_df=data[~msk]
len(train_df)
len(test_df)
from sklearn import ensemble
train_df_x=train_df.drop(columns=['target'])

train_df_y=train_df.target



test_df_x=test_df.drop(columns=['target'])

test_df_y=test_df.target
regr1=ensemble.RandomForestClassifier()
regr1.fit(train_df_x,train_df_y)
from sklearn.metrics import accuracy_score
train_ac=accuracy_score(regr1.predict(train_df_x),train_df_y)

train_ac
test_ac=accuracy_score(regr1.predict(test_df_x),test_df_y)

test_ac
pd.DataFrame({'Features':train_df_x.columns,'Importance':regr1.feature_importances_}).plot.bar(x='Features')
gbt=ensemble.GradientBoostingClassifier()
gbt.fit(train_df_x,train_df_y)
train_ac=accuracy_score(gbt.predict(train_df_x),train_df_y)

train_ac
test_ac=accuracy_score(gbt.predict(test_df_x),test_df_y)

test_ac
pd.DataFrame({'Features':train_df_x.columns,'Importance':gbt.feature_importances_}).plot.bar(x='Features',color='orange')
pd.DataFrame({'Features':train_df_x.columns,

              'Importance_RF':regr1.feature_importances_,

              'Importance_GBT':gbt.feature_importances_}).plot.bar(x='Features')