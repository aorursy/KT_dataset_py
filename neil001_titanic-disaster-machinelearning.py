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
## Importing the required libraries and packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
## Importing the train csv datafile



titanic_train = pd.read_csv('../input/titanic/train.csv')
## Importing the test csv datafile



titanic_test = pd.read_csv('../input/titanic/test.csv')
## Description of every column for train data



titanic_train.info()
## Description of every column for test data



titanic_test.info()
## Checking top 5 rows of the train dataset



titanic_train.head()
## Checking top 5 rows of the test dataset



titanic_test.head()
## Checking for top columns having missing values percentage in train dataset



print ((titanic_train.isnull().sum()/len(titanic_train)*100).round(2).sort_values(ascending = False).head())
## Checking for top columns having missing values percentage in test dataset



print ((titanic_test.isnull().sum()/len(titanic_test)*100).round(2).sort_values(ascending = False).head())
## Dropping the missing value Cabin column from train dataset



titanic_train.drop('Cabin', axis=1, inplace=True)
## Dropping the missing value Cabin column from test dataset



titanic_test.drop('Cabin', axis=1, inplace=True)
## Checking for the Age distribution in training dataset



plt.figure(figsize=(10,5))

sns.distplot(titanic_train['Age'] , bins=30)
## Filling the missing values in Age cloumn with mean values for training dataset



titanic_train['Age'] = (titanic_train['Age'].fillna(titanic_train['Age'].mean())).round()
## Plotting the count of passenger sex based on Pclass for training set



plt.figure(figsize=(10,5))

sns.countplot(x = 'Sex' , data = titanic_train , hue = 'Pclass' , palette = 'rainbow')

plt.title('Passenger Sex based on Pclass')
## Checking for the Age distribution in test dataset



plt.figure(figsize=(10,5))

sns.distplot(titanic_test['Age'] , bins=30)
## Filling the missing values in Age cloumn with mean values for test dataset



titanic_test['Age'] = (titanic_test['Age'].fillna(titanic_test['Age'].mean())).round()
## Plotting the count of passenger sex based on Pclass for test set



plt.figure(figsize=(10,5))

sns.countplot(x = 'Sex' , data = titanic_test , hue = 'Pclass' , palette = 'viridis')

plt.title('Passenger Sex based on Pclass')
## Creating a new feature name Family Size(SibSp+Parch+1) for training set



titanic_train['Family Size'] = titanic_train['SibSp'] + titanic_train['Parch'] + 1
## Creating a new feature name Family Size(SibSp+Parch+1) for test set



titanic_test['Family Size'] = titanic_test['SibSp'] + titanic_test['Parch'] + 1
## Dropping columns which doesn't add up to our predictions in training set



titanic_train.drop(['Name','Ticket','SibSp','Parch'] , axis = 1 , inplace = True)
## Dropping columns which doesn't add up to our predictions in test set



titanic_test.drop(['Name','Ticket','SibSp','Parch'] , axis = 1 , inplace = True)
## Convert categorical variables for training set into dummy variables (i.e. one-hot encoding)



dummies = pd.get_dummies(titanic_train[['Sex','Embarked']],drop_first=True)



titanic_train_dummy = titanic_train.drop(['Sex','Embarked'] , axis = 1)



titanic_train_dummy = pd.concat([titanic_train_dummy , dummies] , axis = 1)



titanic_train_dummy.info()
## Convert categorical variables for test set into dummy variables (i.e. one-hot encoding)



dummies = pd.get_dummies(titanic_test[['Sex','Embarked']],drop_first=True)



titanic_test_dummy = titanic_test.drop(['Sex','Embarked'] , axis = 1)



titanic_test_dummy = pd.concat([titanic_test_dummy , dummies] , axis = 1)



titanic_test_dummy['Fare'] = (titanic_test_dummy['Fare'].fillna(titanic_test_dummy['Fare'].mean())).round()



titanic_test_dummy.info()
## Plotting the count of passenger died and survived



plt.figure(figsize=(10,5))

sns.countplot(titanic_train_dummy['Survived'])

plt.title('Count of passenger died and survived')
## Plotting the count of passenger died and survived based on Pclass



plt.figure(figsize=(10,5))

sns.countplot(x = 'Survived' , data = titanic_train_dummy , hue = 'Pclass')

plt.title('Passenger died and survived based on Pclass')
## Plotting joint relationship between 'Fare' , 'Age' , 'Pclass' & 'Survived' for training dataset



sns.pairplot(titanic_train_dummy[['Fare','Age','Pclass','Survived']] , hue = 'Survived' , height = 4)
## Checking correlation between all the features using heatmap for training dataset



plt.figure(figsize=(10,8))

corr = titanic_train_dummy.corr()

sns.heatmap(corr , annot = True , linecolor = 'black' , linewidth = .01)

plt.title('Correlation between features')
## Checking correlation between all the features using heatmap for test dataset



plt.figure(figsize=(10,8))

corr = titanic_test_dummy.corr()

sns.heatmap(corr , annot = True , linecolor = 'black' , linewidth = .01 , cmap = 'YlGnBu')

plt.title('Correlation between features')
X_train = titanic_train_dummy.drop(['Survived','PassengerId'] , axis =1)
y_train = titanic_train_dummy['Survived']
X_test = titanic_test_dummy.drop("PassengerId", axis=1).copy()
print ('X_train =' , X_train.shape)

print ('y_train =' , y_train.shape) 

print ('X_test =' , X_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_test)
acc_log = (logreg.score(X_train, y_train)*100).round(2)



print ('Accuracy score is:', acc_log)
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train , y_train)
y_pred = svc.predict(X_test)
acc_svm = (svc.score(X_train, y_train)*100).round(2)



print ('Accuracy score is:', acc_svm)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train , y_train)
y_pred = knn.predict(X_test)
acc_knn = (knn.score(X_train, y_train)*100).round(2)



print ('Accuracy score is:', acc_knn)
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train , y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = (gaussian.score(X_train, y_train)*100).round(2)



print ('Accuracy score is:', acc_gaussian)
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train , y_train)
y_pred = random_forest.predict(X_test)
acc_random_forest = (random_forest.score(X_train, y_train)*100).round(2)



print ('Accuracy score is:', acc_random_forest)
model = pd.DataFrame({

    'Model': ['Logistic Regression','Support Vector Machine','K - Nearest Neighbours',

              'Naive Bayes Classifier','Random Forest Classifier'],

    'Score': [acc_log, acc_svm, acc_knn, acc_gaussian, acc_random_forest]})



model.sort_values(by = 'Score' , ascending = False)
final_pred = pd.DataFrame({

        "PassengerId": titanic_test_dummy['PassengerId'],

        "Survived": y_pred

    })
## Predicting the survival and writing to CSV



final_pred.to_csv('gender_submission.csv', index = False)
titanic_survival = pd.read_csv('gender_submission.csv')
titanic_survival.info()
print (titanic_survival['Survived'].value_counts())