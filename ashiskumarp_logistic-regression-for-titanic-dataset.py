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


train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()
train.info()
train.describe()
train.isna().sum()
train.Embarked = train.Embarked.fillna(train['Embarked'].mode()[0])
median_age = train.Age.median()

train.Age.fillna(median_age, inplace = True)



train.drop('Cabin', axis = 1,inplace = True)
train['FamilySize'] = train['SibSp'] + train['Parch']+1
train['GenderClass'] = train.apply(lambda x: 'child' if x['Age'] < 15 else x['Sex'],axis=1)
train[train.Age<15].head(2)
train = pd.get_dummies(train, columns=['GenderClass','Embarked'], drop_first=True)
train = train.drop(['Name','Ticket','Sex','SibSp','Parch'], axis = 1)

train.head()
import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy

import seaborn as sns                                              # Provides a high level interface for drawing attractive and informative statistical graphics

%matplotlib inline

sns.set()

sns.pairplot(train[["Fare","Age","Pclass","Survived"]],vars = ["Fare","Age","Pclass"],hue="Survived", dropna=True,markers=["o", "s"])

plt.title('Pair Plot')
corr = train.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr,vmax=.8,linewidth=.01, square = True, annot = True,cmap='YlGnBu',linecolor ='black')

plt.title('Correlation between features')
X = train.loc[:,train.columns != 'Survived']

X.head()
y = train.Survived 
y.head()
print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print(X_train.shape)

print(X_test.shape)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)
y_pred_test = logreg.predict(X_test)  
from sklearn.metrics import accuracy_score

print('Accuracy score for test data is:', accuracy_score(y_test,y_pred_test))
test.head()
test.isnull().sum()
median_age = test.Age.median()

test.Age.fillna(median_age, inplace = True)
test.drop('Cabin', axis = 1,inplace = True)
median_fare = test.Fare.median()

test.Fare.fillna(median_fare, inplace = True)
test['FamilySize'] = test['SibSp'] + test['Parch']+1
test['GenderClass'] = test.apply(lambda x: 'child' if x['Age'] < 15 else x['Sex'],axis=1)
test = pd.get_dummies(test, columns=['GenderClass','Embarked'], drop_first=True)
test_processing = test.drop(['Name','Ticket','Sex','SibSp','Parch'], axis = 1)

test_processing.head()
y_pred_test = logreg.predict(test_processing)  
y_pred_test
y_pred_test.shape
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred_test

    })

submission.to_csv('submission.csv', index=False)