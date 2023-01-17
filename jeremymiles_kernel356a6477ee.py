# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import random as rnd
from os import *
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
test_df  = pd.read_csv("../input/test.csv")

print(train_df.columns)


myvars = ['Sex', 'Survived', 'Pclass']
print(train_df[myvars].groupby(['Pclass']).mean())
print(train_df[['Sex', 'Survived']].groupby(['Sex']).mean())

pclassgraph = sns.barplot(train_df['Pclass'], train_df['Survived']*100,
                          hue = train_df['Sex'])
pclassgraph.set(xlabel='Passenger Class', ylabel='Percent Survived')
plt.show()

train_df['kid'] = (train_df['Age'] < 16).astype(float)
test_df['kid'] = (test_df['Age'] < 16).astype(float)

train_df['Age'] = np.where(train_df['Age'] < 16, 
                          16, train_df['Age'])

test_df['Age'] = np.where(test_df['Age'] < 16, 
                          16, test_df['Age'])


train_df['Age2'] = train_df['Age'] ** 2
test_df['Age2'] = test_df['Age'] ** 2

train_df_no_na = train_df[['Sex', 'Pclass', 'Age', 'Age2', 'kid', 'Survived']].dropna()
train_df_no_na["male"] = (train_df_no_na["Sex"] == "male").astype(float)
test_df['male'] = (test_df["Sex"] == "male").astype(float)

train_df_no_na['Pclass1'] = (train_df_no_na['Pclass'] == 1).astype(float)
train_df_no_na['Pclass2'] = (train_df_no_na['Pclass'] == 2).astype(float)
test_df['Pclass1'] = (test_df['Pclass'] == 1).astype(float)
test_df['Pclass2'] = (test_df['Pclass'] == 2).astype(float)

x_train = train_df_no_na[['Pclass1', 'Pclass2', 'male', 'kid', 'Age', 'Age2']]
y_train = train_df_no_na['Survived']
logit = sm.Logit(y_train, x_train)
result = logit.fit()
print(result.summary2())



test_df['Survived'] = (result.predict(test_df[['Pclass1', 'Pclass2', 'male', 'kid', 'Age', 'Age2']]) > 0.5).astype(int)



submission = test_df[['PassengerId', 'Survived']] 

submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv("sub8.csv", index = None)