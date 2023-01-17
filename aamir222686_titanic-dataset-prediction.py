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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()
test_df.head()
train_df.info()
train_df = train_df.drop('Cabin', axis=1)
train_df.head()
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass', y='Age', data=train_df, palette='rainbow')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 28

        else:

            return 24

    else:

        return Age
train_df['Age'] = train_df[['Age', 'Pclass']].apply(impute_age, axis=1)
train_df.dropna(inplace=True)
train_df.head()
sex = pd.get_dummies(train_df['Sex'], drop_first=True)

embarked = pd.get_dummies(train_df['Embarked'], drop_first=True)
train_df.drop(train_df[['Name', 'Sex', 'Ticket', 'Embarked']], axis=1, inplace=True)
train_df.head()
train_df = pd.concat([train_df, sex, embarked], axis=1)
train_df.info()
test_df.head()
test_df.info()
test_df['Age'] = test_df[['Age', 'Pclass']].apply(impute_age, axis=1)
test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)
test_df.head()
sex_test = pd.get_dummies(test_df['Sex'], drop_first=True)

embarked_test = pd.get_dummies(test_df['Embarked'], drop_first=True)
test_df.drop(['Name', 'Sex', 'Embarked', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df = pd.concat([test_df, sex_test, embarked_test],  axis=1)
test_df.head()
train_df.head()
X_train = train_df.drop(['Survived', 'PassengerId'], axis=1)

y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

log_pred = log_reg.predict(X_test)

log_score = round(log_reg.score(X_train, y_train) *100, 2)
log_score
from sklearn.tree import DecisionTreeClassifier

d_tree = DecisionTreeClassifier()

d_tree.fit(X_train, y_train)

dt_pred = d_tree.predict(X_test)

dt_score = round(d_tree.score(X_train, y_train) * 100, 2)
dt_score
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

knn_score = round(knn.score(X_train, y_train) * 100, 2)
knn_score
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_score = round(rf.score(X_train, y_train) * 100,2)
rf_score
from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train, y_train)

svm_predict = svm.predict(X_test)

svm_score = round(svm.score(X_train, y_train)*100, 2)
svm_score
from sklearn.naive_bayes import GaussianNB

guassian = GaussianNB()

guassian.fit(X_train, y_train)

gua_pred = guassian.predict(X_test)

gua_score = round(guassian.score(X_train, y_train)*100, 2)
gua_score
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 

              'Decision Tree'],

    'Score': [svm_score, knn_score, log_score, 

              rf_score, gua_score, dt_score]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": dt_pred

    })
submission.to_csv('submission.csv', index=False)