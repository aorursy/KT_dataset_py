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
import warnings

warnings.filterwarnings('ignore')
## importing the dataset 

titanic = pd.read_csv('/kaggle/input/titanic/train.csv')
## let's check the head

titanic.head()
## let's check the shape

titanic.shape
# let's check info of dataset

titanic.info()
## converted pclass into object type 

titanic['Pclass']=titanic['Pclass'].astype('str')
## check for null values

round(100*(titanic.isnull().sum()/titanic.shape[0]),2)
## cabin has 77% null values hence removing it 

titanic.drop('Cabin',axis=1,inplace=True)
## remove rows don't have Embarked 

titanic = titanic[~titanic['Embarked'].isnull()]
## impute the missing values with mean of the Age 

age_mean = titanic['Age'].mean()

titanic['Age'].fillna(age_mean,inplace=True)
## computing the mean fare 

fare_mean = titanic['Fare'].mean()

## checking after removing null values

round(100*(titanic.isnull().sum()/titanic.shape[0]),2)
## let's impute null values of age using mode/median

## hence check for outliers first

import seaborn as sns

import matplotlib.pyplot as plt

cap_cols = ['Age','SibSp','Parch','Fare']

plt.figure(figsize=(15,5))

for i in enumerate(cap_cols):

    plt.subplot(2,2,i[0]+1)

    sns.boxplot(titanic[i[1]])

plt.tight_layout()

plt.show()

## replacing outliers by some upper range and lower range values

for i in cap_cols:

    q1 = titanic[i].quantile(0.05)

    q3 = titanic[i].quantile(0.95)

    titanic[i][titanic[i]<=q1]=q1

    titanic[i][titanic[i]>=q3]=q3
plt.figure(figsize=(15,5))

for i in enumerate(cap_cols):

    plt.subplot(2,2,i[0]+1)

    sns.boxplot(titanic[i[1]])

plt.tight_layout()

plt.show()
## create dummy variables

dummy = pd.get_dummies(titanic[['Sex','Embarked','Pclass']],drop_first=True)

dummy.head()
titanic = pd.concat([titanic,dummy],axis=1)

## concatinate our dummy sets with main dataframe
titanic.drop(['Sex','Embarked','Ticket','Name','PassengerId','Pclass'],axis=1,inplace=True)
import sklearn

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

titanic[cap_cols] = sc.fit_transform(titanic[cap_cols])
## check head after scaling

titanic.head()
y_train = titanic.pop('Survived')

X_train = titanic
from sklearn.svm import SVC

svc = SVC(C = 1).fit(X_train,y_train)
titanic['Survived_pred'] = svc.predict(X_train)
## check head once 

titanic.head()
# Evaluate the model using confusion matrix 

from sklearn import metrics

metrics.confusion_matrix(y_true=y_train, y_pred=titanic.Survived_pred)
# print other metrics



# accuracy

print("accuracy", metrics.accuracy_score(y_train, titanic.Survived_pred))



# precision

print("precision", metrics.precision_score(y_train, titanic.Survived_pred))



# recall/sensitivity

print("recall", metrics.recall_score(y_train, titanic.Survived_pred))
## let's prepare the test data 

titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
titanic_test[cap_cols] = sc.transform(titanic_test[cap_cols])
## converted pclass into object type 

titanic_test['Pclass']=titanic_test['Pclass'].astype('str')
## create dummy variables

dummy_test = pd.get_dummies(titanic_test[['Sex','Embarked','Pclass']],drop_first=True)

dummy_test.head()
titanic_test = pd.concat([titanic_test,dummy_test],axis=1)

## concatenate our dummy sets
titanic_test.head()
## impute the missing values with mean of the Age 

titanic_test['Age'].fillna(age_mean,inplace=True)
titanic_test['Fare'].fillna(fare_mean,inplace=True) ## impute missing fare values 
cols_pred = ['Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S',

       'Pclass_2', 'Pclass_3']

titanic_test['Survived'] = svc.predict(titanic_test[cols_pred])
titanic_test.head()
titanic_test_final = titanic_test[['PassengerId','Survived']]
titanic_test_final.head()
titanic_test_final.to_csv("prediction_titanic_svm.csv",index=False)
