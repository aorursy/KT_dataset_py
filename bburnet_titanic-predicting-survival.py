#load in libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import seaborn as sns
sns.set_palette('Set2')
import matplotlib.pyplot as plt
#load in data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
test.shape
train.shape
train.columns
train.head()
train.describe()
train.info()
total_survived = train['Survived'].value_counts()
proportion_survived = total_survived / train.shape[:1]
survival = pd.DataFrame({
    'Count': total_survived,
    'Percentage': proportion_survived})
survival.sort_values(by = 'Percentage', ascending = False)
train = train.drop(['Cabin', 'Ticket', 'Name'], axis = 1)
test = test.drop(['Cabin', 'Ticket', 'Name'], axis = 1)
train.isnull().sum()
train['Age'].fillna(train['Age'].median(), inplace = True)
train['Embarked'].fillna(train['Embarked'].value_counts().index[0], inplace = True)
#change passenger class from 1,2,3 to 1st,2nd,3rd
d = {1:'1st', 2:'2nd', 3:'3rd'}
train['Pclass'] = train['Pclass'].map(d)
#get dummy variables for categorical variables
cat_vars = train[['Pclass', 'Sex', 'Embarked']]
dummies = pd.get_dummies(cat_vars, drop_first = True)

train = train.drop(['Pclass', 'Sex', 'Embarked'], axis = 1)

train = pd.concat([train, dummies], axis = 1)
######      modeling      ######
#set up
X = train.drop(['Survived'], 1)
y = train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
#linear regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logr = LogisticRegression()
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)
acc_log = metrics.accuracy_score(y_pred, y_test)
print("Logistic Regression accuracy score: ", acc_log)
#decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
acc_dt = metrics.accuracy_score(y_pred, y_test)
print('Decision Tree accuracy score: ', acc_dt)
#support vector machine
from sklearn import svm
sv = svm.SVC()
sv.fit(X_train, y_train)
y_pred = sv.predict(X_test)
acc_svm = metrics.accuracy_score(y_pred, y_test)
print('SVM Accuracy Score: ', acc_svm)
#k nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors = 3)
knc.fit(X_train, y_train)
y_pred = knc.predict(X_test)
acc_knn = metrics.accuracy_score(y_pred, y_test)
print('KNN Accuracy Score: ', acc_knn)
a_index = list(range(1, 11))
a = pd.Series()
x = [1,2,3,4,5,6,7,8,9,10]
for i in list(range(1, 11)):
    kcs = KNeighborsClassifier(n_neighbors = i)
    kcs.fit(X_train, y_train)
    y_pred = kcs.predict(X_test)
    a = a.append(pd.Series(metrics.accuracy_score(y_pred, y_test)))
plt.plot(a_index, a)
plt.xticks(x)
knc2 = KNeighborsClassifier(n_neighbors = 4)
knc2.fit(X_train, y_train)
y_pred = knc2.predict(X_test)
acc_knn2 = metrics.accuracy_score(y_pred, y_test)
print('KNN Accuracy Score: ', acc_knn2)
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Support Vector Machines', 'K-Nearest Neighbors'],
    'Score': [acc_log, acc_dt, acc_svm, acc_knn2]})
models.sort_values(by = 'Score', ascending = False)