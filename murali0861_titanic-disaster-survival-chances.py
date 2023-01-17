import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv(os.path.join(dirname, 'train.csv'), sep=",")

train_data.head()
test_data = pd.read_csv(os.path.join(dirname, 'test.csv'), sep=",")

test_data.head()
test_data.shape
train_data.shape
train_data.info()
# ===> unique values count:

train_data.PassengerId.nunique()
train_data.Survived.value_counts()
train_data.Pclass.value_counts()
train_data.Sex.value_counts()
train_data.Cabin.value_counts().head(10)

print(train_data[train_data.Cabin.isnull() == False]['Pclass'].value_counts())

print(train_data[train_data.Cabin.isnull() == False]['Survived'].value_counts())
train_data.Pclass.value_counts()
train_data.Sex.value_counts()
train_data.Ticket.nunique()
t = train_data.Ticket.value_counts()

train_data['TicketCoPassengers'] = train_data.Ticket.apply(lambda x: t[x]-1) 

train_data.head()
t = test_data.Ticket.value_counts()

test_data['TicketCoPassengers'] = test_data.Ticket.apply(lambda x: t[x]-1)
train_data.Ticket.value_counts().head(10)
train_data[train_data.Ticket == '110152']
train_data.groupby('Ticket').TicketCoPassengers.count().head()
train_data.head()
train_data.isnull().sum(axis=0) / train_data.shape[0]
train_data.describe(include=['O'])
train_data.Age.value_counts().head()
sns.distplot(train_data.Fare.dropna(), kde=False, bins=30)
sns.distplot(train_data.TicketCoPassengers.dropna(), kde=False, bins=30)
sns.countplot(train_data.Embarked.fillna('NoClass'))
sns.countplot(train_data.Survived)
sns.countplot(train_data.Parch) 
sns.countplot(hue='Survived', x='Parch', orient='h', data=train_data, )
train_data.info()
sns.countplot(x=train_data.Embarked, hue=train_data.Survived)
sns.countplot(x='Pclass', hue='Survived', data=train_data)
sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.figure()

sns.distplot(train_data[train_data.Survived == 1].Age.dropna(), bins=30)

sns.distplot(train_data[train_data.Survived == 0].Age.dropna(), bins=30)

plt.show()
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
train_data.Age = train_data.Age.fillna(train_data.Age.mean())

test_data.Age = test_data.Age.fillna(test_data.Age.mean())
train_data.Age.isnull().sum(axis=0)
bins = [0, 15, 40, 60, 120]

labels = [1, 2, 3, 4]

train_data['age_bin'] = pd.cut(train_data.Age, bins=bins, labels=labels)

test_data['age_bin'] = pd.cut(test_data.Age, bins=bins, labels=labels)
train_data.head()
plt.figure()

g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Fare', bins=20)

plt.show()
train_data.Fare = train_data.Fare.fillna(train_data.Fare.mean())

test_data.Fare = test_data.Fare.fillna(test_data.Fare.mean())
bins = [-1, 50, 100, 200, 10000]

labels = [1, 2, 3, 4]

train_data['Fare_bins'] = pd.cut(train_data.Fare, bins=bins, labels=labels)

test_data['Fare_bins'] = pd.cut(test_data.Fare, bins=bins, labels=labels)
train_data.head()
train_data.Embarked.value_counts()
train_data[train_data.Embarked.isnull()]
train_data.Embarked = train_data.Embarked.fillna('S')

test_data.Embarked = test_data.Embarked.fillna('S')
train_data.head()
train_data.Sex = train_data.Sex.map({'male': 1, 'female': 0})

test_data.Sex = test_data.Sex.map({'male': 1, 'female': 0})
# ===> 

train_data = pd.get_dummies(train_data, columns=['Pclass', 'Embarked'], drop_first=True)

test_data = pd.get_dummies(test_data, columns=['Pclass', 'Embarked'], drop_first=True)

train_data.head()
train_data.drop(columns=['Age', 'Fare', 'Ticket'], inplace=True)

test_data.drop(columns=['Age', 'Fare', 'Ticket'], inplace=True)
train_data.loc[:, 'Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_data.loc[:, 'Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_data.Title, train_data.Survived)
def replaceTitles(_data):

    _data['Title'] = _data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    _data['Title'] = _data['Title'].replace('Mlle', 'Miss')

    _data['Title'] = _data['Title'].replace('Ms', 'Miss')

    _data['Title'] = _data['Title'].replace('Mme', 'Mrs')

    return _data



train_data = replaceTitles(train_data)

test_data = replaceTitles(test_data)

print(train_data['Title'].value_counts())

print(test_data['Title'].value_counts())
train_data = pd.get_dummies(train_data, columns=['Title'], drop_first=True)

test_data = pd.get_dummies(test_data, columns=['Title'], drop_first=True)
train_data.drop(columns=['Name'], inplace=True)

test_data.drop(columns=['Name'], inplace=True)
train_data.head()
#====> Fill cabin NaN as not assigned

train_data.drop(columns=['Cabin', 'PassengerId'], inplace=True)

test_data_passengers = test_data.PassengerId

test_data.drop(columns=['Cabin', 'PassengerId'], inplace=True)

# train_data.Cabin.fillna('NotAssigned').head()
train_data.head()
test_data.head()
x_train = train_data.drop(columns=['Survived'])

y_train = train_data.Survived
log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(test_data)

log_reg.score(x_train, y_train)
log_reg.coef_[0]
corr = pd.DataFrame({'features': x_train.columns, 'coff': log_reg.coef_[0]}).sort_values(by='coff', ascending=False)

corr
train_data.shape
g_logit = LogisticRegression()

params = {

    "C": np.logspace(-3, 3, 7),

    "penalty": ["l1", "l2"]  # l1 lasso l2 ridge

}

kfold = KFold(n_splits=2, random_state=101, shuffle=True)



grid = GridSearchCV(g_logit,

                    param_grid=params,

                    n_jobs=-1,

                    verbose=True,

                    return_train_score=True,

                    scoring='accuracy',

                    cv=kfold)

grid.fit(x_train, y_train)
grid.best_score_
results = pd.DataFrame(grid.cv_results_)

results.sort_values(by='mean_test_score', ascending=False)
logit_alog = grid.best_estimator_

logit_alog.fit(x_train, y_train)

logit_y_predict = logit_alog.predict(test_data)

print("Accuracy score for the logistic regression::", logit_alog.score(x_train, y_train)*100)
knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)

knn.fit(x_train, y_train)

knn_y_pred = knn.predict(test_data)
knn.score(x_train, y_train)
knn.get_params
g_knn = KNeighborsClassifier()

params = {

    'n_neighbors': range(2, 20),

    'leaf_size': range(4, 40, 4)

}



kfold = KFold(n_splits=2, shuffle=True, random_state=101)



grid = GridSearchCV(g_knn,

                    param_grid=params,

                    cv=kfold,

                    n_jobs=-1,

                    scoring='accuracy',

                    verbose=True,

                    return_train_score=True)

grid.fit(x_train, y_train)
grid.best_estimator_
grid.best_score_
grid.best_params_
results = pd.DataFrame(grid.cv_results_)

results.sort_values(by='mean_test_score', ascending=False)
knn_algo = grid.best_estimator_

knn_algo.fit(x_train, y_train)

knn_y_pred = knn.predict(test_data)

print("Accuracy score for the logistic regression::", knn_algo.score(x_train, y_train)*100)
from sklearn.svm import SVC
svc_alog = SVC()

params = {

    'C': [0.1, 1, 10, 20, 30, 40, 50, 100, 150],

    'kernel': ['rbf']

}

svc_grid = GridSearchCV(svc_alog,

                        param_grid=params,

                        cv=kfold,

                        scoring='accuracy',

                        n_jobs=-1,

                        verbose=True,

                        return_train_score=True)

svc_grid.fit(x_train, y_train)
results = pd.DataFrame(svc_grid.cv_results_)

results.sort_values(by='mean_test_score', ascending=False)
svc_grid.best_score_
svc_alog = svc_grid.best_estimator_

svc_alog.fit(x_train, y_train)

svc_y_pred = svc_alog.predict(test_data)

print("Linear SVM algorithm accuracy score:: ", svc_alog.score(x_train, y_train)*100) 
decision_alog = DecisionTreeClassifier()
decision_alog.fit(x_train, y_train)

decision_alog.score(x_train, y_train)
decision_alog
g_decision_alog = DecisionTreeClassifier()

params = {

    'min_samples_split': range(2, 20, 2),

    'max_leaf_nodes': range(2, 10, 2)

}

decision_tree_grid = GridSearchCV(g_decision_alog,

                                  param_grid=params,

                                  verbose=True,

                                  scoring='accuracy',

                                  return_train_score=True,

                                  cv=kfold,

                                  n_jobs=-1)

decision_tree_grid.fit(x_train, y_train)
decision_tree_grid.best_score_
decision_tree_grid.best_estimator_
result = pd.DataFrame(decision_tree_grid.cv_results_)

result.sort_values(by='mean_test_score', ascending=False)
decision_tree_algo = decision_tree_grid.best_estimator_

decision_tree_algo.fit(x_train, y_train)

dec_tree_y_pred = decision_tree_algo.predict(test_data)

print("Decision Tree alogirthm score:: ", decision_tree_algo.score(x_train, y_train) * 100) 
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier
decision_tree_algo
d_alog = DecisionTreeClassifier(max_leaf_nodes=8, min_samples_split=2)

adaboost = AdaBoostClassifier(d_alog, n_estimators=200)

# adaboost.fit(x_train, y_train)
kfold = KFold(n_splits=2, random_state=101, shuffle=True)

algo = DecisionTreeClassifier(max_leaf_nodes=8, min_samples_split=2)

adaboost = AdaBoostClassifier(base_estimator=algo, n_estimators=200)

params = {

    "base_estimator__criterion": ["gini", "entropy"],

    "base_estimator__splitter": ["best", "random"],

    "n_estimators": range(2, 200, 2)

}

adaboost_grid = GridSearchCV(adaboost,

                             param_grid=params,

                             verbose=True,

                             scoring='accuracy',

                             return_train_score=True,

                             cv=kfold,

                             n_jobs=-1)

adaboost_grid.fit(x_train, y_train)
adaboost_grid.best_score_
result = pd.DataFrame(adaboost_grid.cv_results_)

result.sort_values(by='mean_test_score', ascending=False)
adaboost_grid.best_params_
rfc = RandomForestClassifier()

params = {

    'min_samples_split': range(2, 20, 2),

    'max_leaf_nodes': range(2, 10, 2)

}

rfc_grid = GridSearchCV(rfc,

                        param_grid=params,

                        verbose=True,

                        scoring='accuracy',

                        return_train_score=True,

                        cv=kfold,

                        n_jobs=-1)

rfc_grid.fit(x_train, y_train)
rfc_grid.best_score_
rfc_grid.best_params_
final_model = rfc_grid.best_estimator_

rfc_y_pred = final_model.predict(test_data)
submission = pd.DataFrame({'PassengerId': test_data_passengers, 'Survived': rfc_y_pred})

submission.head()
# submission.to_csv('../output/submission.csv', index=False)