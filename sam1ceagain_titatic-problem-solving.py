# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train['Sex']=pd.get_dummies(train['Sex'],drop_first=True)
train['Embarked']=pd.get_dummies(train['Embarked'],drop_first=True)
train.head()
train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
X = train.drop('Survived', axis = 1)
y = train['Survived']
sns.heatmap(X.isnull())
X.fillna(X['Age'].mean(), inplace=True)
sns.heatmap(X.isnull())
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X,y)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
X.columns
test_shorted = test[['Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
test_shorted['Sex']=pd.get_dummies(train['Sex'],drop_first=True)
test_shorted['Embarked']=pd.get_dummies(train['Embarked'],drop_first=True)
sns.heatmap(test_shorted.isnull())
test_shorted.fillna(X['Age'].mean(), inplace=True)
y_pred =LR.predict(test_shorted)
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_pred,y)
classification_report