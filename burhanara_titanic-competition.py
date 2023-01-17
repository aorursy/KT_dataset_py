from warnings import filterwarnings

filterwarnings('ignore')



# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.model_selection import learning_curve, GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
train_data.head()
train_data.info()
train_data.describe().T
sns.heatmap(train_data.isnull())
sns.countplot(x='Survived', data=train_data)
sns.distplot(train_data['Age'].dropna())
train_data['Fare'].hist(bins=30)
train_data['Age']= train_data[['Age']].fillna(value=train_data['Age'].mean())
sns.heatmap(train_data.isnull())
train_data= train_data.drop(['Cabin','PassengerId','Name','Ticket'], axis=1)
sex=pd.get_dummies(train_data['Sex'], drop_first=True)

embarked= pd.get_dummies(train_data['Embarked'], drop_first=True)
train_data.drop(['Sex','Embarked'], axis=1,inplace=True)
train_data=pd.concat([train_data,sex,embarked], axis=1)
train_data.head()
from sklearn.model_selection import train_test_split

X=train_data.drop(['Survived'], axis=1)

y=train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30,random_state=42)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier().fit(X_train, y_train)
rf_model 
from sklearn.metrics import accuracy_score #works

y_pred = rf_model.predict(X_test)

accuracy_score(y_test, y_pred)
rf_params = {"max_depth": [2,5,8,10],

            "max_features": [2,5,8],

            "n_estimators": [10,500,1000],

            "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()



rf_cv_model = GridSearchCV(rf_model, 

                           rf_params, 

                           cv = 10, 

                           n_jobs = -1, 

                           verbose = 2) 
rf_cv_model.fit(X_train, y_train)
print("Best Params: " + str(rf_cv_model.best_params_))
rf_tuned = RandomForestClassifier(max_depth = 8, 

                                  max_features = 2, 

                                  min_samples_split = 2,

                                  n_estimators = 500)



rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
filter_test_data=test_data.drop(['Cabin','PassengerId','Name','Ticket'], axis=1)

filter_test_data['Age']=filter_test_data[['Age']].fillna(value=filter_test_data['Age'].mean())

sex_test=pd.get_dummies(filter_test_data['Sex'], drop_first=True)

embarked_test= pd.get_dummies(filter_test_data['Embarked'], drop_first=True)

filter_test_data.drop(['Sex','Embarked'], axis=1,inplace=True)

filter_test_data=pd.concat([filter_test_data,sex_test,embarked_test], axis=1)
filter_test_data.isnull().sum()
filter_test_data['Fare']=filter_test_data[['Fare']].fillna(value=filter_test_data['Age'].mean())

final_pred=rf_tuned.predict(filter_test_data)
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": final_pred

    })

 

submission.to_csv('submission.csv', index=False)
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")