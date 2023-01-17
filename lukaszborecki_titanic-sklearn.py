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
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic.info()
titanic.describe()
titanic.drop('Cabin', inplace=True, axis=1)


titanic['Embarked'].fillna('S', inplace=True)
titanic.Embarked.value_counts()

# Create a groupby object: by_sex_class

by_sex_pclass = titanic.groupby(['Sex','Pclass'])



# Write a function that imputes median

def impute_median(series):

    return series.fillna(series.median())



# Impute age and assign to titanic.age

titanic.Age = by_sex_pclass.Age.transform(impute_median)



np.median(titanic['Age'])
titanic.info()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]

pipeline = Pipeline(steps)
titanic= titanic[['Survived', 'Pclass', 'Sex', 'Age','SibSp','Parch','Fare','Embarked']]

titanic = pd.get_dummies(titanic)
titanic.drop(['Sex_male','Embarked_C'] ,axis=1, inplace=True)


X = titanic.drop('Survived',axis=1)

y= titanic['Survived']
parameters = {'knn__n_neighbors': np.arange(1, 20, 1),

             'knn__leaf_size': np.arange(1,30,2)}
parameters = {"tree__max_depth": [3,2,1, None],

              "tree__max_features": [1,4,7],

              "tree__min_samples_leaf": [2,5,8],

              "tree__criterion": ["gini", "entropy"]}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=55)
cv = GridSearchCV(pipeline, parameters, cv=5)

cv.fit(X, y)

y_pred = cv.predict(X_test)

print(cv.best_params_)

print(cv.best_score_)



pipeline.fit(X,y)
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
titanic_test.head()
titanic_test.Age = by_sex_pclass.Age.transform(impute_median)
np.mean(titanic_test.Fare)
titanic_test['Fare'].fillna('35.62', inplace=True)
titanic_test['Fare']=titanic_test.Fare.astype(float)
titanic_test.info()

titanic_test.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
titanic_test=pd.get_dummies(titanic_test)
titanic_test.drop(['Sex_male','Embarked_C'] ,axis=1, inplace=True)
titanic_test.columns
X.columns
titanic_test_id = titanic_test['PassengerId']
X_test = titanic_test.drop('PassengerId', axis=1)

y_test= cv.predict(X_test)
titanic_test_id.head()
result=pd.DataFrame()

result['PassengerId']=titanic_test_id
result['Survived']=y_test.reshape(-1,1)
result
result.to_csv('result_knn.csv')