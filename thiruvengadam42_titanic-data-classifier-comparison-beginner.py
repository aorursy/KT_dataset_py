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
df=pd.read_csv('../input/train.csv')
df.head()
df.isnull().sum()
df['Age'].fillna(df['Age'].mode()[0], inplace= True)
df['Embarked']=df['Embarked'].factorize()[0]
df=df.loc[:,['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Embarked']]
df.shape
df=pd.get_dummies(df)
df.shape
col_list=[col for col in df.columns if col not in ['PassengerId', 'Name']]
x=df.loc[:,col_list]
y=df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(x, y, test_size=0.2)
X_train.shape, X_test.shape, y_train.shape,y_test.shape
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifiers = [

    KNeighborsClassifier(3),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]
# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)
for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
df1=pd.read_csv('../input/test.csv')
df1.shape
df1.isnull().sum()
df1['Age'].fillna(df1['Age'].mode()[0], inplace=True)
df1['Embarked']=df1['Embarked'].factorize()[0]
df1['Fare'].fillna(df1['Fare'].mean(), inplace=True)
df1=df1.loc[:,['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare','Embarked']]
df1=pd.get_dummies(df1)
df1.shape
df1=df1.loc[:,X_train.columns]
df1.shape
df1.fillna(0, inplace=True)
candidate_classifier = GradientBoostingClassifier()

candidate_classifier.fit(X_train, y_train)

result = candidate_classifier.predict(df1)
df2=pd.read_csv('../input/test.csv')
submit_file=pd.DataFrame(df2['PassengerId'])
submit_file=pd.DataFrame(df2['PassengerId'])
submit_file['Survived']=result
submit_file