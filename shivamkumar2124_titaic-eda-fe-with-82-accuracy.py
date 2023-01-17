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
train = pd.read_csv('../input/titanic/train.csv')
train.head()
train.isnull().sum()
train.drop(['Name','PassengerId','Ticket'], axis =1, inplace = True)
train.isnull().sum()
train.drop('Cabin', axis =1, inplace = True)
train.corr()
import seaborn as sns
sns.heatmap(train.corr())
train['Age'] = train['Age'].fillna(np.mean(train['Age']))
train.isnull().sum()
train
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.figure(figsize = (12,6))
sns.countplot(train['Survived'])
plt.figure(figsize = (20,8))
sns.heatmap(train.corr(), annot = True)
plt.figure(figsize = (15,8))
sns.scatterplot(train['Survived'],train['Fare'])
print(train['Pclass'].unique())
print(train['Embarked'].unique())
sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)
p = pd.get_dummies(train['Pclass'], drop_first = True)
train = pd.concat([sex,train,embark,p], axis =1)
train.drop(['Pclass','Sex','Embarked'], axis =1, inplace = True)
from sklearn.linear_model import LogisticRegression
x = train.drop('Survived', axis =1)
y = train['Survived']
model = LogisticRegression()
model.fit(x,y)
test = pd.read_csv('../input/titanic/test.csv')
test_x = pd.read_csv('../input/titanic/test.csv')
test['Age'] = test['Age'].fillna(np.mean(test['Age']))
test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))
test.isnull().sum()
sex_t = pd.get_dummies(test['Sex'], drop_first = True)
embark_t = pd.get_dummies(test['Embarked'], drop_first = True)
p_t = pd.get_dummies(test['Pclass'], drop_first = True)
test = pd.concat([sex_t,test,embark_t,p_t], axis =1)
test.drop(['PassengerId','Name','Sex','Pclass','Ticket','Embarked','Cabin'], axis =1, inplace = True)
test
prediction = model.predict(test)
output = pd.DataFrame({'PassengerId' : test_x.PassengerId, 'Survived':prediction})
output.to_csv('Submission.csv', index = False)
output.head()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(20,400,num = 20)]
max_depth = [int(x) for x in np.linspace(1,100,num = 10)]
min_samples_split = [2,3,5,7,8]
min_samples_leaf = [2,3,5,8]
bootstrap = [True,False]
random_para = {'n_estimators' : n_estimators,
               'max_depth' : max_depth,
                'min_samples_split' : min_samples_split,
                'min_samples_leaf' : min_samples_leaf,
                'bootstrap' : bootstrap}
from sklearn.model_selection import RandomizedSearchCV
random = RandomizedSearchCV(estimator = rf, param_distributions = random_para, cv = 3, n_iter = 100, verbose = 5)
random.fit(x,y)
random.best_estimator_
rf = RandomForestClassifier(max_depth=34, min_samples_leaf=3, min_samples_split=3,
                       n_estimators=200)
rf.fit(x,y)
prediction1 = rf.predict(test)
output = pd.DataFrame({'PassengerId' : test_x.PassengerId, 'Survived':prediction1})
output.to_csv('Submission.csv', index = False)
output.head()
