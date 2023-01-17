# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(5)
test.head()
train.isna().sum()
train.describe(include='all')
np.mean(train.Age)
num_trains =  len(train)
all = pd.concat([train, test], axis=0)
test_id = test['PassengerId']
all.shape
all = all.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
all['Age'] = all['Age'].fillna(29)
all['Embarked'] = all['Embarked'] .fillna('S')
all['Fare'] = all['Fare'].fillna(0)
all.info()
all['Sex'] = (all.Sex=='M').astype('int')
all = pd.get_dummies(all, ['Embarked'])
train = all[~(all.Survived.isna())] 
test = all[all.Survived.isna()] 
train.shape, test.shape
test = test.drop(['Survived'], axis=1)
dtc = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3)
dtc.fit(train.drop(['Survived'],axis=1), train.Survived)
prediction = dtc.predict(test)
submission = pd.concat([test_id, pd.Series(prediction, index=test.index)],axis=1)
submission.columns=['PassengerId', 'Survived']
submission['Survived'] =submission['Survived'].astype(int)

submission.to_csv("humble_submission.csv", index=False)
submission.describe()
submission.head()
submission.shape
