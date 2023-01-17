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
import pandas as pd
import numpy as np
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_test.head()
df_train.head()
df_train.columns
df_train.count(0)
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_train.isnull().sum().sum()
df_test.isnull().sum().sum()
numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))
df_train_2 = df_train.drop(categorical_feats, axis=1)
df_test_2 = df_test.drop(categorical_feats, axis=1)
print('missin train values')
total = df_train_2.isnull().sum().sort_values(ascending=False)
percent = (df_train_2.isnull().sum()/df_train_2.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
print('missin test values')
total = df_test_2.isnull().sum().sort_values(ascending=False)
percent = (df_test_2.isnull().sum()/df_test_2.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_train_2['Age'].mean()
df_train_2['Age'].fillna((df_train_2['Age'].mean()), inplace=True)
df_test_2['Age'].fillna((df_test_2['Age'].mean()), inplace=True)
print('missin train values')
total = df_train_2.isnull().sum().sort_values(ascending=False)
percent = (df_train_2.isnull().sum()/df_train_2.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_train_2.isnull().sum().sum()
df_test_2.isnull().sum().sum()
df_test_2['Fare'].fillna((df_test_2['Fare'].mean()), inplace=True)
df_test_2.isnull().sum().sum()
df_train_2.columns
from sklearn.linear_model import LogisticRegression
y_train = df_train_2[["Survived"]]
x_train = df_train_2.drop(["Survived"],axis = 1)
x_train.tail()
model = LogisticRegression()
model.fit(x_train, y_train)
submission = pd.read_csv("../input/gender_submission.csv")
submission.head()
df_test_2.tail()
submission.tail()
for i in range(417):
    submission.loc[i , 'Survived'] = model.predict(df_test_2.iloc[i,0:].values.reshape(1,6))[0]
submission.tail()
submission.to_csv("final_submission.csv", index=False)


