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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.tail()
df_test.tail()
df_train.info()
print('-----------------------------------------------')
df_test.info()
survived = df_train.groupby(['Survived'])['PassengerId'].count()
survived.plot(kind='bar', figsize=(6, 4), color='steelblue', width=0.8)
plt.xticks(rotation='horizontal')
print(survived)
plt.show()
embarked = df_train.groupby(['Embarked'])['PassengerId'].count()
embarked.plot(kind='bar', figsize=(6, 4), color='steelblue', width=0.8)
plt.xticks(rotation='horizontal')
print(embarked)
plt.show()
gender = df_train.groupby(['Sex'])['PassengerId'].count()
gender.plot(kind='bar', figsize=(6, 4), color='steelblue', width=0.8)
plt.xticks(rotation='horizontal')
print(gender)
plt.show()
pclass = df_train.groupby(['Pclass'])['PassengerId'].count()
pclass.plot(kind='bar', figsize=(6, 4), color='steelblue', width=0.8)
plt.xticks(rotation='horizontal')
print(pclass)
plt.show()
survived_Pclass = df_train.groupby(['Pclass', 'Survived'])['Survived'].count()
survived_Pclass.plot(kind='bar', figsize=(6, 4), color=['red', 'yellowgreen'], width=0.8)
plt.xticks(rotation='horizontal')
print(survived_Pclass)
plt.show()
survived_gender = df_train.groupby(['Sex', 'Survived'])['Survived'].count()
survived_gender.plot(kind='bar', figsize=(6, 4), color=['red', 'yellowgreen'], width=0.8)
plt.xticks(rotation='horizontal')
print(survived_gender)
plt.show()
survived_embarked = df_train.groupby(['Embarked', 'Survived'])['Survived'].count()
survived_embarked.plot(kind='bar', figsize=(6, 4), color=['red', 'yellowgreen'], width=0.8)
plt.xticks(rotation='horizontal')
print(survived_embarked)
plt.show()
g = sns.FacetGrid(df_train, col='Survived', aspect=1.5, height=4)
g.map(plt.hist, 'Age', bins=16)
grid = sns.FacetGrid(df_train, col='Survived', row='Sex', height=3, aspect=1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=16, color='purple')
grid.add_legend();
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', height=3, aspect=1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=16, color='purple')
grid.add_legend();
grid = sns.FacetGrid(df_train, row='Embarked', height=3, aspect=1.5)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=['orange', 'blue'], order=[1, 2, 3], hue_order=['female', 'male'])
grid.add_legend()
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Fare'], axis=1)
df_test = df_test.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Fare'], axis=1)

df_train.shape, df_test.shape
embarked_mode = df_train['Embarked'].mode()[0]
embarked_mode
df_train['Embarked'] = df_train['Embarked'].fillna(embarked_mode)
df_train['Embarked'].count()
df_combine = [df_train, df_test]

for df in df_combine:
    df['Sex'] = df['Sex'].map({'male':0, 'female':1}).astype((int))
    df['Embarked'] = df['Embarked'].map({'C':0, 'Q':1, 'S':2}).astype(int)
    
df_train.head()
for df in df_combine:
    for i in range(0,2):
        for j in range(0,3):
            df_ages = df[(df['Sex'] == i) & (df['Pclass'] == j+1)]['Age'].dropna()
            age_median = df_ages.median()
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), 'Age'] = age_median

df_train.head()
# separate the features
df_features = df_train.drop(['Survived'], axis=1)
scaler = StandardScaler()
cols = list(df_features.columns)
df_train_norm = pd.DataFrame(data = df_features)
df_train_norm[cols] = scaler.fit_transform(df_features[cols])
df_train_norm.head(10)
X_train = df_train_norm
y_train = df_train['Survived']
X_test = df_test.drop('PassengerId', axis=1)

X_train.shape, y_train.shape, X_test.shape
# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train)*100, 2)
print('KNN accuracy: ', acc_knn)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = round(dt.score(X_train, y_train)*100, 2)
print('Decision Tree accuracy: ', acc_dt)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = round(rf.score(X_train, y_train)*100, 2)
print('Random Forest accuracy: ', acc_rf)

# Logistic Regression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
acc_logreg = round(logreg.score(X_train, y_train)*100, 2)
print('Logistic Regression accuracy: ', acc_logreg)

# SVC
svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train)*100, 2)
print('SVC accuracy: ', acc_svc)

# Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
acc_nb = round(nb.score(X_train, y_train)*100, 2)
print('Gaussian Naive Bayes accuracy: ', acc_nb)
classifier = ['KNN', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'SVC', 'Gaussian Naive Bayes']
acc = [acc_knn, acc_dt, acc_rf, acc_logreg, acc_svc, acc_nb]

model = pd.DataFrame({'Model': classifier, 'Accuracy': acc})
model = model.sort_values(['Accuracy'], ascending=False).reset_index(drop=True)
model
submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred_dt})
submission.head(10)
submission.to_csv('submission.csv', index=False)