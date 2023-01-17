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

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import math





import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams['figure.figsize'] = ((15,5))
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")
# lets look into top 5 rows

print("Top 5 Rows\n",)

train.head()
# lets look into random 5 rows

print("Random 5 Rows\n",)

train.sample(5)
# Info of training data

train.info()
# as we can see there are missing values in training data, lets look using seaborn

sns.heatmap(train.isna(), annot=False, )
print("Average age of entire data:" , math.ceil(train.Age.mean()))

sns.distplot(train.Age, kde=False, )
sns.boxplot(y=train.Age, x= train.Pclass, palette="magma")
# There are multiple approch to fill missing values, we trying one simple manual approach here!



# train.Age=train.Age.fillna(train.Age.mean())

for i in range(len(train.Age)):

      if pd.isnull(train.Age[i]):

        if train.Pclass[i]==1:

            train.Age[i]=37

        elif train.Pclass[i]==2:

            train.Age[i]=30  

        elif train.Pclass[i]==3:

            train.Age[i]=25

            

for i in range(len(test.Age)):

      if pd.isnull(test.Age[i]):

        if test.Pclass[i]==1:

            test.Age[i]=37

        elif test.Pclass[i]==2:

            test.Age[i]=30  

        elif test.Pclass[i]==3:

            test.Age[i]=25            

            

test.Fare.fillna(test.Fare.mean(), inplace=True)            
# Lets see if there is any correlation between any numeric column

sns.heatmap(train.drop(["PassengerId"], axis=1).corr(), annot=True, )

plt.tight_layout()
sns.countplot(train.Sex, palette="magma")
print(train.Survived.value_counts())

sns.countplot(train.Survived)
sns.countplot(x=train.Survived, hue=train.Sex,)
sns.countplot(x=train.Survived, hue=train.SibSp,)
sns.countplot(x=train.Survived, hue=train.Pclass)
plt.figure(figsize=(12,5))

sns.distplot(train.Fare, kde=False, bins=10, )
train.head(2)
# So here, we are applying OHE on Sex and Embarked column

Sex_dummies=pd.get_dummies(train.Sex)

Embarked_dummies=pd.get_dummies(train.Embarked)

train=pd.concat([train,Sex_dummies,Embarked_dummies], axis=1)

train=train.drop(['Sex', 'Embarked'],axis=1)



Sex_dummies=pd.get_dummies(test.Sex)

Embarked_dummies=pd.get_dummies(test.Embarked)

test=pd.concat([test,Sex_dummies,Embarked_dummies], axis=1)

test=test.drop(['Sex', 'Embarked'],axis=1)



train.head()
print("Percentage of NaN value in Canbin:", (train.Cabin.isna().sum()/train.Cabin.isna().count())*100)
PassengerId=test['PassengerId']
# Now lets remove few irrelevant columns, columns that  do not play any crucial role and has too many NaN value.

train.drop(['Name', 'Cabin', 'PassengerId', 'Ticket'], axis=1,inplace=True)

test.drop(['Name', 'Cabin', 'PassengerId', 'Ticket'], axis=1,inplace=True)

train.head(2)
sns.pairplot(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']],)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'],axis=1), train.Survived,

                                                    test_size=0.25)
from sklearn import model_selection

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

def kfold_and_confusion_matrix(model):

    kfold = KFold(n_splits=5)

    model_kfold = model

    results_kfold = model_selection.cross_val_score(model_kfold, X_train, y_train,  cv=kfold)

    print("K Fold Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 

    

    y_pred=model.predict(X_test)

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train, y_train)

model.score(X_test, y_test)



print("Normal Accuracy:",(model.score(X_test, y_test)*100))

kfold_and_confusion_matrix( model)
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler



logregpipe = Pipeline([('scale', StandardScaler()),

                   ('logreg',LogisticRegression())])



# Gridsearch to determine the value of C

param_grid = {'logreg__C':np.arange(0.01,1,30)}

logreg_cv = GridSearchCV(logregpipe,param_grid,cv=5,return_train_score=True)

logreg_cv.fit(X_train,y_train)

print(logreg_cv.best_params_)





bestlogreg = logreg_cv.best_estimator_

bestlogreg.fit(X_train,y_train)

bestlogreg.coef_ = bestlogreg.named_steps['logreg'].coef_

print("Normal Accuracy: %.2f%%" % (bestlogreg.score(X_train,y_train)*100))



kfold_and_confusion_matrix( bestlogreg)
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train, y_train)

print("Normal Accuracy:",(model.score(X_test, y_test)*100))



kfold_and_confusion_matrix( model)
from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

print("Normal Accuracy:",(model.score(X_test, y_test)*100))



kfold_and_confusion_matrix(model)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight="balanced", n_estimators=250)

model.fit(X_train, y_train)

print("Normal Accuracy:",(model.score(X_test, y_test)*100))



kfold_and_confusion_matrix(model)
from sklearn.ensemble import  GradientBoostingClassifier

gb_model = GradientBoostingClassifier()

gb_model.fit(X_train, y_train)

print("Normal Accuracy:",(gb_model.score(X_test, y_test)*100))



kfold_and_confusion_matrix( gb_model)
import lightgbm as lgb

from sklearn import metrics





lg = lgb.LGBMClassifier(silent=False)

param_dist = {"max_depth": [25,50, 75],

              "learning_rate" : [0.01,0.05,0.1],

              "num_leaves": [300,900,1200],

              "n_estimators": [200]

             }

grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)

grid_search.fit(X_train,y_train)

# grid_search.best_estimator_

model= grid_search.best_estimator_

model.fit(X_train,y_train)

print("Normal Accuracy:",(model.score(X_test, y_test)*100))



kfold_and_confusion_matrix( model)

import xgboost as xgb

from sklearn import metrics

# Parameter Tuning

model = xgb.XGBClassifier()

param_dist = {"max_depth": [10,30,50],

              "min_child_weight" : [1,3,6],

              "n_estimators": [200],

              "learning_rate": [0.05, 0.1,0.16],}

grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, 

                                   verbose=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

model=grid_search.best_estimator_

# model = xgb.XGBClassifier(max_depth=50, min_child_weight=1,  n_estimators=200,\

#                           n_jobs=-1 , verbose=1,learning_rate=0.1)

model.fit(X_train,y_train)

print("Normal Accuracy:",(model.score(X_test, y_test)*100))



kfold_and_confusion_matrix( model)
result=pd.DataFrame(columns=["PassengerId", "Survived"])

result["PassengerId"]=PassengerId

result["Survived"]= gb_model.predict(test)

result.to_csv("Submission.csv", index=False)