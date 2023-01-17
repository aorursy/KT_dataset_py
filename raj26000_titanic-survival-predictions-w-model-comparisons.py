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
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
#Dropping unnecessary features.

data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

print(data_train)



data_val = pd.read_csv("/kaggle/input/titanic/test.csv")



drop_col = ['PassengerId', 'Cabin', 'Ticket']

val_id = data_val['PassengerId']

data_train.drop(drop_col, axis=1, inplace=True)

data_val.drop(drop_col, axis=1, inplace=True)

print(data_train)
#Identifying and Filling up missing values.

print(data_train.isnull().sum())

print(data_val.isnull().sum())



data_train['Age'].fillna(data_train['Age'].median(), inplace=True)

data_train['Embarked'].fillna(data_train['Embarked'].mode()[0], inplace=True)

data_val['Age'].fillna(data_train['Age'].median(), inplace=True)

data_val['Embarked'].fillna(data_train['Embarked'].mode()[0], inplace=True)

data_val['Fare'].fillna(data_val['Fare'].mean(), inplace=True)



print(data_train.isnull().sum())

data_val.isnull().sum()
#Generating some features using existing ones.

data_train['family_size'] = data_train['SibSp'] + data_train['Parch'] + 1

data_train['is_alone'] = 1

data_train['is_alone'].loc[data_train['family_size'] > 1] = 0



data_val['family_size'] = data_val['SibSp'] + data_val['Parch'] + 1

data_val['is_alone'] = 1

data_val['is_alone'].loc[data_val['family_size'] > 1] = 0



#Encoding non-numerical distict features as categorical variables.

label_encoder = LabelEncoder()

data_train['Embarked'] = label_encoder.fit_transform(data_train['Embarked'])

data_train['Sex'] = label_encoder.fit_transform(data_train['Sex'])

data_val['Embarked'] = label_encoder.fit_transform(data_val['Embarked'])

data_val['Sex'] = label_encoder.fit_transform(data_val['Sex'])

print(data_train)
#Splitting given training dataset into train, test sets to evaluate various classifiers before using actual validation data.

X_col = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'family_size', 'is_alone']

X_train, X_test, y_train, y_test = train_test_split(data_train[X_col], data_train['Survived'], random_state=0)



#1. K-Nearest neighbors Classifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))



cv_scores = cross_validate(knn, data_train[X_col], data_train['Survived'], cv=50, return_train_score=True)

print('KNN training score:',cv_scores['train_score'].mean())

print('KNN testing score:',cv_scores['test_score'].mean())

print('KNN fit time:',cv_scores['fit_time'].mean())
#2. Logistic Regression

clf = LogisticRegression(penalty='l2', max_iter=500).fit(X_train, y_train)

print(clf.score(X_test, y_test))



cv_scores = cross_validate(clf, data_train[X_col], data_train['Survived'], cv=40, return_train_score=True)

print('Logistic Regression training score:',cv_scores['train_score'].mean())

print('Logistic Regression testing score:',cv_scores['test_score'].mean())

print('Logistic Regression fit time:',cv_scores['fit_time'].mean())
#3. Linear SVM

clf = SVC(kernel='linear').fit(X_train, y_train)

print(clf.score(X_test, y_test))



cv_scores = cross_validate(clf, data_train[X_col], data_train['Survived'], cv=50, return_train_score=True)

print('Linear SVM training score:',cv_scores['train_score'].mean())

print('Linear SVM testing score:',cv_scores['test_score'].mean())

print('Linear SVM fit time:',cv_scores['fit_time'].mean())
#4. Decision Tree

clf = DecisionTreeClassifier(max_depth=7).fit(X_train, y_train)

print(clf.score(X_test, y_test))



cv_scores = cross_validate(clf, data_train[X_col], data_train['Survived'], cv=40, return_train_score=True)

print('Decision Tree training score:',cv_scores['train_score'].mean())

print('Decision Tree testing score:',cv_scores['test_score'].mean())

print('Decision Tree fit time:',cv_scores['fit_time'].mean())
print(data_val[X_col])

print(data_val.isnull().sum())



#Using Decision Tree Classifier based on comparison of training results.

y_val = clf.predict(data_val[X_col])



print('-'*50)

print('Output labels:')

print(y_val)
#Saving predictions for submission.

labels = pd.DataFrame()

labels['PassengerId'] = val_id

labels['Survived'] = y_val

print(labels)



labels.to_csv('Submission.csv', index = False)

print('Submission has been saved.')