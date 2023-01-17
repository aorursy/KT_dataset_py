# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import classification_report



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv", index_col="PassengerId")

test = pd.read_csv("/kaggle/input/titanic/test.csv", index_col="PassengerId")
train.head()
test.head()
train.isna().sum()
test.isna().sum()
train.describe(include="all")
train.hist(figsize=(12, 9))

plt.show()
test.hist(figsize=(12, 9))

plt.show()
train['Age'].fillna(train['Age'].mean(), inplace=True)

test['Age'].fillna(test['Age'].mean(), inplace=True)

train['Embarked'].fillna("S", inplace=True)

test['Fare'].fillna(test['Fare'].mean(), inplace=True)

train.describe(include="all")
train['Sex_Male'] = (train['Sex'] == "male").astype(int)

train['Embarked_S'] = (train['Embarked'] == "S").astype(int)

test['Sex_Male'] = (test['Sex'] == "male").astype(int)

test['Embarked_S'] = (test['Embarked'] == "S").astype(int)

train.describe(include="all")
X = train[['Pclass', 'Sex_Male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_S']].values

y = train['Survived'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
candidate_models = {

    "Logistic Regression": LogisticRegression(max_iter=1000000),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(),

    "Random Forest": RandomForestClassifier(),

    "Support Vector Machine": SVC()

}
for name, model in candidate_models.items():

    print(name)

    model.fit(X_train, y_train)

    print(classification_report(y_valid, model.predict(X_valid)))
rf = RandomForestClassifier(n_estimators=200, min_samples_split=10)

rf.fit(X_train, y_train)

print(classification_report(y_valid, rf.predict(X_valid)))
final_rf = RandomForestClassifier(n_estimators=200, min_samples_split=10)

final_rf.fit(X, y)
X_test = test[['Pclass', 'Sex_Male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_S']].values

y_pred = final_rf.predict(X_test)
df_submit = pd.DataFrame({'PassengerId': test.index, 'Survived': y_pred})

display(df_submit.head())

df_submit.to_csv("/kaggle/working/submission.csv", header=True, index=False)