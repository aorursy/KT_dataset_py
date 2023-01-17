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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_train.head()
df_test.head()
df_train.info()
df_train.describe()
sex_dict = {'male' : 0, 'female' : 1}
df_train['Sex'] = df_train['Sex'].map(sex_dict)
df_train.head()
df_train.corr()
#sns.pairplot(data=df_train)
plt.figure(figsize=(20,15))
sns.heatmap(data = df_train.isnull())
df_train.Cabin.isnull().sum()
plt.figure(figsize=(10,8))
sns.barplot(x='Pclass', y=('Survived'),  data=df_train)
#plt.figure(figsize=(10,8))
#sns.barplot(x='Fare', y=('Survived'),  data=df_train)
plt.figure(figsize=(10,8))
sns.countplot(x='Pclass',  data=df_train)
plt.figure(figsize=(10,8))
sns.barplot(x='Embarked', y=('Survived'),  data=df_train)
plt.figure(figsize=(10,8))
sns.barplot(x='Pclass', y=('Age'),  data=df_train)
df_train["Age"] = df_train["Age"].fillna(round(df_train.groupby("Pclass")["Age"].transform("mean")))
df_train.dropna(subset = ['Embarked'], inplace = True)
df_train.drop('Cabin',axis=1, inplace = True)
df_train.drop('Name',axis=1, inplace = True)
df_train.drop('Ticket',axis=1, inplace = True)
one_hot = pd.get_dummies(df_train['Embarked'])
df_train.drop('Embarked',axis = 1, inplace = True)
df_train = df_train.join(one_hot)
df_train
plt.figure(figsize=(20,15))
sns.heatmap(data = df_train.isnull())
X = df_train.drop(['Survived','Parch','PassengerId','SibSp','C','Q','S'],axis=1)
y = df_train['Survived']
X
y
X_training, X_testing, y_training, y_testing = train_test_split(X,y, test_size = 0.33)
classifier = LogisticRegression(max_iter = 10000)
history = classifier.fit(X,y)
df_test
df_test.isnull().sum()
df_test["Fare"].mean()
df_test['Sex'] = df_test['Sex'].map(sex_dict)
df_test["Age"] = df_test["Age"].fillna(round(df_train.groupby("Pclass")["Age"].transform("mean")))
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].mean())

df_test.drop('Cabin',axis=1, inplace = True)
df_test.drop('Name',axis=1, inplace = True)
df_test.drop('Ticket',axis=1, inplace = True)

one_hot = pd.get_dummies(df_test['Embarked'])
df_test.drop('Embarked',axis = 1, inplace = True)
df_test = df_test.join(one_hot)

df_T = df_test
df_test = df_test.drop(['Parch','PassengerId','SibSp','C','Q','S'],axis=1)
df_test
predictions = classifier.predict(df_test)
predictions
submission_test = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
submission_test
submission = pd.DataFrame({
        "PassengerId": df_T["PassengerId"],
        "Survived": predictions
    })
submission
submission.to_csv('submission.csv', index=False)
model = RandomForestClassifier(n_estimators=1000)
model.fit(X,y)
preds = model.predict(df_test)
submission_RF = pd.DataFrame({
        "PassengerId": df_T["PassengerId"],
        "Survived": preds
    })
submission_RF
submission_RF.to_csv('submission2.csv', index=False)
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

L1 = LogisticRegression(max_iter=10000, C=10)
L1.fit(X_training, y_training)
P1 = L1.predict(X_testing)

T_accuracy = accuracy_score(P1,y_testing)
T_accuracy
F1 = RandomForestClassifier(n_estimators = 1000)
F1.fit(X_training, y_training)
P2 = F1.predict(X_testing)

T2_accuracy = accuracy_score(P2,y_testing)
T2_accuracy
D1 = DecisionTreeClassifier(random_state=42)
D1.fit(X_training, y_training)
P3 = D1.predict(X_testing)

T3_accuracy = accuracy_score(P3,y_testing)
T3_accuracy
