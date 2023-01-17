# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams["patch.force_edgecolor"] = True
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap = 'viridis')
sns.set_style('whitegrid')
sns.countplot(x='Pclass', data= train)
sns.countplot(x='Survived', data = train, hue='Sex')
train.info()
plt.figure(figsize=(12,7))
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
import cufflinks as cs
cs.go_offline()

sample = train[['Age','Pclass']]
train.head()
def fill_age(col):
    age = col[0]
    pclass = col[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age
train['Age'] = train[['Age','Pclass']].apply(fill_age, axis = 1)
plt.figure(figsize=(12,7))
sns.heatmap(train.isnull(), yticklabels=False, cbar = False)
train['Embarked'].isnull().value_counts()
train['Cabin'].isnull().value_counts()
train.drop('Cabin', axis = 1, inplace = True)
train.dropna(inplace=True)
plt.figure(figsize=(12,7))
sns.heatmap(train.isnull(), yticklabels=False, cbar = False)
train.head()
sns.distplot(train['Fare'], kde = False, bins = 50)
#sns.distplot(train['Age'], kde = False, bins = 50)
sample1 = train[['Age','Survived']]
sample1.pivot(columns='Survived', values = 'Age').iplot(kind = 'hist',barmode = 'stacked', bins = 20)
sample1 = train[['Age','Pclass']]
sample1.pivot(columns='Pclass', values = 'Age').iplot(kind = 'hist',barmode = 'stacked', bins = 20)
train.head()
embark = pd.get_dummies(train['Embarked'], drop_first=True)
sex = pd.get_dummies(train['Sex'], drop_first=True)
pclass = pd.get_dummies(train['Pclass'], drop_first=True)
train = pd.concat([train,embark,sex,pclass], axis = 1)
train.head()
train.drop(['Pclass','Embarked','Sex'], axis =1, inplace = True)
modified_train = train.copy(deep = True)
modified_train.head()
modified_train.drop(['Name','Ticket'], axis = 1, inplace=True)

modified_train.set_index('PassengerId', inplace=True)
X_train = modified_train.drop('Survived', axis = 1)
y_train = modified_train['Survived']
modified_train.head()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pd.DataFrame(data = lr.coef_, columns=X_train.columns)
from sklearn.metrics import classification_report
#Test data

test = pd.read_csv('../input/test.csv')
test['Age'] = test[['Age','Pclass']].apply(fill_age, axis = 1)
test.drop('Cabin', axis = 1, inplace = True)
test.info()
test.isnull().sum()
test.at[test['Fare'].isna(),'Fare'] = test[test['Pclass'] == 3]['Fare'].mean()
test.isna().sum()


test.dropna(inplace=True)
embark = pd.get_dummies(test['Embarked'], drop_first=True)
sex = pd.get_dummies(test['Sex'], drop_first=True)
pclass = pd.get_dummies(test['Pclass'], drop_first=True)
test = pd.concat([test,embark,sex,pclass], axis = 1)

test.drop(['Pclass','Embarked','Sex'], axis =1, inplace = True)
modified_test = test.copy(deep = True)
modified_test.drop(['Name','Ticket'], axis = 1, inplace=True)
modified_test.set_index('PassengerId', inplace=True)



#modified_test.head()
#sns.heatmap(modified_test.isnull())
predictions = pd.Series(lr.predict(modified_test))
modified_test.reset_index(inplace=True)
modified_test.head()
first_submission = pd.concat([modified_test['PassengerId'], predictions], axis = 1)
first_submission.columns = ['PassengerId', 'Survived']
first_submission.head()
first_submission.to_csv('First_Submission.csv', index = False)
first_submission.info()
