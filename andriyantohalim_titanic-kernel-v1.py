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
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_combined = pd.concat([df_train, df_test], sort=False)
df_train.head()
df_train.describe()
df_test.head()
df_test.describe()
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by="Survived", ascending=False)
df_train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by="Survived", ascending=False)
df_train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by="Survived", ascending=False)
df_train[['Parch','Survived']].groupby(['Parch'], as_index=False).maean().sort_values(by="Survived", ascending=False)
df_train['Title'] = df_train['Name'].str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(df_train['Title'], df_train['Sex'])
df_test['Title'] = df_test['Name'].str.extract('([A-Za-z]+)\.', expand=False)
df_train['Title'] = df_train['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Jonkheer', 'Sir', 'Rev', 'Dona'], 'Rare')
df_train['Title'] = df_train['Title'].replace(['Lady', 'Countess', 'Mme'], 'Mrs')
df_train['Title'] = df_train['Title'].replace(['Ms', 'Mlle'], 'Miss')
df_test['Title'] = df_test['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Jonkheer', 'Sir', 'Rev', 'Dona'], 'Rare')
df_test['Title'] = df_test['Title'].replace(['Lady', 'Countess', 'Mme'], 'Mrs')
df_test['Title'] = df_test['Title'].replace(['Ms', 'Mlle'], 'Miss')
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
df_train['FamSize'] = df_train['Parch'] + df_train['SibSp'] + 1 
df_test['FamSize'] = df_test['Parch'] + df_test['SibSp'] + 1
df_train.head()
df_train[['FamSize','Survived']].groupby(['FamSize'], as_index=False).mean().sort_values(by="Survived", ascending=False)
df_train['IsAlone'] = df_train["FamSize"].map(lambda s: 1 if s==1 else 0)
df_test['IsAlone'] = df_test["FamSize"].map(lambda s: 1 if s==1 else 0)
df_train[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by="Survived", ascending=False)
df_train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by="Survived", ascending=False)
df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].median())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
df_train['CategoricalFare'] = pd.qcut(df_train['Fare'], 10)
# df_test['CategoricalFare'] = pd.qcut(df_test['Fare'], 10)
print (df_train[['CategoricalFare','Survived']].groupby(['CategoricalFare'], as_index=False).mean().sort_values(by="Survived", ascending=False))
age_median = df_combined['Age'].median()

age_median
age_mean = df_combined['Age'].mean()

age_mean
age_std = df_combined['Age'].std()

age_std
age_null_count = df_train['Age'].isnull().sum()

age_null_count
df_train['Age'] = df_train['Age'].fillna(age_median)
df_train['CategoricalAge'] = pd.cut(df_train['Age'], 8)

print (df_train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
df_test['Age'] = df_test['Age'].fillna(age_median)
# df_test['CategoricalAge'] = pd.cut(df_test['Age'], 8) # no 'cutting' on test data
df_train['Title'] = df_train['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 0}).astype(int)
df_train.head()
df_test['Title'] = df_test['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 0}).astype(int)
df_test.head()
df_train['Sex'] = df_train['Sex'].map({'female' : 0, 'male' : 1}).astype(int)
df_train.head()
df_test['Sex'] = df_test['Sex'].map({'female' : 0, 'male' : 1}).astype(int)
df_test.head()
df_train.loc[df_train['Embarked'] == 'Empty']
df_train['Embarked'] = df_train['Embarked'].fillna("Empty")

df_train['Embarked'] = df_train['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2, 'Empty' : 0}, na_action='ignore').astype(int)

df_train.head()
df_test['Embarked'] = df_test['Embarked'].fillna("Empty")
df_test.loc[df_test['Embarked'] == 'Empty']
df_test['Embarked'] = df_test['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2, 'Empty' : 0}, na_action='ignore').astype(int)

df_test.head()
df_train['CategoricalAge'] = 0

df_train.loc[df_train['Age']  <= 10.368, 'CategoricalAge'] = 0
df_train.loc[(df_train['Age']  > 10.368) & (df_train['Age'] <= 20.315), 'CategoricalAge'] = 1
df_train.loc[(df_train['Age']  > 20.315) & (df_train['Age'] <= 30.263), 'CategoricalAge'] = 2
df_train.loc[(df_train['Age']  > 30.263) & (df_train['Age'] <= 40.21), 'CategoricalAge'] = 3
df_train.loc[(df_train['Age']  > 40.21) & (df_train['Age'] <= 50.158), 'CategoricalAge'] = 4
df_train.loc[(df_train['Age']  > 50.158) & (df_train['Age'] <= 60.105), 'CategoricalAge'] = 5
df_train.loc[(df_train['Age']  > 60.105) & (df_train['Age'] <= 70.052), 'CategoricalAge'] = 6
df_train.loc[df_train['Age']  > 70.052, 'CategoricalAge'] = 7

df_train.head()
df_test['CategoricalAge'] = 0

df_test.loc[df_test['Age']  <= 10.368, 'CategoricalAge'] = 0
df_test.loc[(df_test['Age']  > 10.368) & (df_test['Age'] <= 20.315), 'CategoricalAge'] = 1
df_test.loc[(df_test['Age']  > 20.315) & (df_test['Age'] <= 30.263), 'CategoricalAge'] = 2
df_test.loc[(df_test['Age']  > 30.263) & (df_test['Age'] <= 40.21), 'CategoricalAge'] = 3
df_test.loc[(df_test['Age']  > 40.21) & (df_test['Age'] <= 50.158), 'CategoricalAge'] = 4
df_test.loc[(df_test['Age']  > 50.158) & (df_test['Age'] <= 60.105), 'CategoricalAge'] = 5
df_test.loc[(df_test['Age']  > 60.105) & (df_test['Age'] <= 70.052), 'CategoricalAge'] = 6
df_test.loc[df_test['Age']  > 70.052, 'CategoricalAge'] = 7

df_test.head()
df_train['CategoricalFare'] = 0

df_train.loc[df_train['Fare'] <= 7.55, 'CategoricalFare'] = 0
df_train.loc[(df_train['Fare'] > 7.55) & (df_train['Fare'] <= 7.854 ), 'CategoricalFare'] = 1
df_train.loc[(df_train['Fare'] > 7.854) & (df_train['Fare'] <= 8.05 ), 'CategoricalFare'] = 2
df_train.loc[(df_train['Fare'] > 8.05) & (df_train['Fare'] <= 10.5 ), 'CategoricalFare'] = 3
df_train.loc[(df_train['Fare'] > 10.5) & (df_train['Fare'] <= 14.454 ), 'CategoricalFare'] = 4
df_train.loc[(df_train['Fare'] > 14.454) & (df_train['Fare'] <= 21.679 ), 'CategoricalFare'] = 5
df_train.loc[(df_train['Fare'] > 21.679) & (df_train['Fare'] <= 27.0 ), 'CategoricalFare'] = 6
df_train.loc[(df_train['Fare'] > 27.0) & (df_train['Fare'] <= 39.688 ), 'CategoricalFare'] = 7
df_train.loc[(df_train['Fare'] > 39.688) & (df_train['Fare'] <= 77.958 ), 'CategoricalFare'] = 8
df_train.loc[df_train['Fare'] > 77.958, 'CategoricalFare'] = 9

df_train.head()
df_test['CategoricalFare'] = 0

df_test.loc[df_test['Fare'] <= 7.55, 'CategoricalFare'] = 0
df_test.loc[(df_test['Fare'] > 7.55) & (df_test['Fare'] <= 7.854 ), 'CategoricalFare'] = 1
df_test.loc[(df_test['Fare'] > 7.854) & (df_test['Fare'] <= 8.05 ), 'CategoricalFare'] = 2
df_test.loc[(df_test['Fare'] > 8.05) & (df_test['Fare'] <= 10.5 ), 'CategoricalFare'] = 3
df_test.loc[(df_test['Fare'] > 10.5) & (df_test['Fare'] <= 14.454 ), 'CategoricalFare'] = 4
df_test.loc[(df_test['Fare'] > 14.454) & (df_test['Fare'] <= 21.679 ), 'CategoricalFare'] = 5
df_test.loc[(df_test['Fare'] > 21.679) & (df_test['Fare'] <= 27.0 ), 'CategoricalFare'] = 6
df_test.loc[(df_test['Fare'] > 27.0) & (df_test['Fare'] <= 39.688 ), 'CategoricalFare'] = 7
df_test.loc[(df_test['Fare'] > 39.688) & (df_test['Fare'] <= 77.958 ), 'CategoricalFare'] = 8
df_test.loc[df_test['Fare'] > 77.958, 'CategoricalFare'] = 9

df_test.head()
drop_elements = ['Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'Parch', 'SibSp', 'IsAlone']
df_train = df_train.drop(drop_elements, axis = 1)
df_test = df_test.drop(drop_elements, axis = 1)
df_train.head()
train = df_train.copy()
train = train.drop('PassengerId', axis=1)

train.head()
test = df_test.copy()
test = test.drop('PassengerId', axis=1)

test.head()
from sklearn.preprocessing import StandardScaler
X_train = train.drop('Survived', axis=1).astype(float)
y_train = df_train['Survived']

X_test = test.astype(float)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print ("X_train shape: ", str(X_train.shape))
print ("y_train shape: ", str(y_train.shape))
print ("X_test shape: ", str(X_test.shape))
# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 3)

acc_log
svc = SVC(gamma='scale')
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 3)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 3)

acc_knn
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 3)

acc_gaussian
perceptron = Perceptron(max_iter=1000, tol=0.001)
perceptron.fit(X_train, y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 3)

acc_perceptron
linear_svc = LinearSVC(max_iter=10000)
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 3)

acc_linear_svc
sgd = SGDClassifier(max_iter=10000, tol=0.001)
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 3)

acc_sgd
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 3)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 3)

acc_random_forest
MLP_clf = MLPClassifier(activation='tanh', solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(50,50), random_state=1, warm_start=True)
MLP_clf.fit(X_train, y_train)
Y_pred = MLP_clf.predict(X_test)
acc_MLP_clf = round(MLP_clf.score(X_train, y_train) * 100, 3)

acc_MLP_clf
adaboost = AdaBoostClassifier()
adaboost.fit(X_train, y_train)
Y_pred = adaboost.predict(X_test)
acc_adaboost = round(adaboost.score(X_train, y_train) * 100, 3)

acc_adaboost
gboost = GradientBoostingClassifier()
gboost.fit(X_train, y_train)
Y_pred = gboost.predict(X_test)
acc_gboost = round(gboost.score(X_train, y_train) * 100, 3)

acc_gboost
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'MLPClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree, acc_MLP_clf, acc_adaboost, acc_gboost]})
models.sort_values(by='Score', ascending=False)
Y_pred = random_forest.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)
