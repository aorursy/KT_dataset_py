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

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head(), train.describe()
test.head(), test.describe()
train.isnull().sum()
test.isnull().sum()
def null_percentage (df):

    for column in df:

        print(column +':', 100 * df[column].isnull().sum()/len(df[column]))
null_percentage(train)
null_percentage(test)
train.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)

test.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
train.Age.fillna(train['Age'].median(), inplace=True), test.Age.fillna(test['Age'].median(), inplace=True)
train.Embarked.fillna(train['Embarked'].mode()[0], inplace=True)
train.Fare.fillna(train['Fare'].median(), inplace=True), test.Fare.fillna(test['Fare'].median(), inplace=True)
train.info(), train.describe()
test.info(), test.describe()
for data in [train, test]:

    data['family_members'] = data['SibSp'] + data['Parch'] + 1

    data['single'] = data['family_members'].map(lambda s: 1 if s == 1 else 0)

    data['small_fam'] = data['family_members'].map(lambda s: 1 if  s == 2  else 0)

    data['med_fam'] = data['family_members'].map(lambda s: 1 if 3 <= s <= 4 else 0)

    data['large_fam'] = data['family_members'].map(lambda s: 1 if s >= 5 else 0)

    data['fare_bin'] = pd.qcut(data['Fare'], 4, labels=False)

    data['age_bin'] = pd.cut(data['Age'].astype(int), 5, labels=False)
sex_dummy_train= pd.get_dummies(train['Sex'], prefix='Sex')

train = pd.concat([train, sex_dummy_train], axis=1)

sex_dummy_test = pd.get_dummies(test['Sex'], prefix='Sex')

test = pd.concat([test, sex_dummy_test], axis=1)
embarked_dummy_train= pd.get_dummies(train['Embarked'], prefix='Embarked')

train = pd.concat([train, embarked_dummy_train], axis=1)

embarked_dummy_test = pd.get_dummies(test['Embarked'], prefix='Embarked')

test = pd.concat([test, embarked_dummy_test], axis=1)
pclass_dummy_train= pd.get_dummies(train['Pclass'], prefix='Pclass')

train = pd.concat([train, pclass_dummy_train], axis=1)

pclass_dummy_test = pd.get_dummies(test['Pclass'], prefix='Pclass')

test = pd.concat([test, pclass_dummy_test], axis=1)
for data in [train, test]:

    data_title = [i.split(",")[1].split(".")[0].strip() for i in data["Name"]]

    data["Title"] = pd.Series(data_title)

    data["Title"].head()

    data["Title"] = data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data["Title"] = data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

    data["Title"] = data["Title"].astype(int)

    data.drop('Name', axis=1, inplace=True)
#train['Cabin'] = pd.Series(i[0] if not pd.isnull(i) else 'X' for i in train['Cabin'])

#train_dummies = pd.get_dummies(train['Cabin'], prefix='cabin')

#train = pd.concat([train, train_dummies], axis=1).drop('Cabin', axis=1)



#test['Cabin'] = pd.Series(i[0] if not pd.isnull(i) else 'X' for i in test['Cabin'])

#test_dummies = pd.get_dummies(test['Cabin'], prefix='cabin')

#test = pd.concat([test, test_dummies], axis=1).drop('Cabin', axis=1)
#ticket = []

#for i in list(train.Ticket):

    #if not i.isdigit() :

        #ticket.append(i.replace('.', '').replace('/', '').strip().split(' ')[0])

    #else:

        #ticket.append('X')

#train['Ticket'] = ticket

#train['Ticket'].head()

#train_dummies = pd.get_dummies(train['Ticket'], prefix='T')

#train = pd.concat([train, train_dummies], axis=1).drop('Ticket', axis=1)



#ticket = []

#for i in list(test.Ticket):

    #if not i.isdigit() :

        #ticket.append(i.replace('.', '').replace('/', '').strip().split(' ')[0])

    #else:

        #ticket.append('X')

#test['Ticket'] = ticket

#test['Ticket'].head()

#test_dummies = pd.get_dummies(test['Ticket'], prefix='T')

#test = pd.concat([test, test_dummies], axis=1).drop('Ticket', axis=1)
train.head(), test.head()
train.head(), test.head()
train.Sex.value_counts()
import seaborn as sns

sns.set(style="darkgrid")

sns.countplot(x='Survived', hue='Sex', data=train)
sns.violinplot(x='Sex', y='Age', hue='Survived', data=train)
train.Pclass.value_counts()
sns.catplot(x='Sex', hue='Pclass', col='Survived', kind='count', data=train)
sns.catplot(x='Embarked', hue='Pclass', col='Survived', kind='count', data=train)
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=train)
train.Age.value_counts()
sns.distplot(train.Age, bins=20, kde=False, rug=True)
sns.pairplot(train[['Survived', 'Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare']])
import matplotlib.pyplot as plt

_ , ax = plt.subplots(figsize =(14, 12))

colormap = sns.diverging_palette(220, 10, as_cmap = True)

_ = sns.heatmap(train.corr(), square=True, annot=True, cmap=colormap)
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

val_test = test.drop('PassengerId', axis=1)

train[['Sex', 'Embarked', 'age_bin', 'fare_bin']] = ordinal_encoder.fit_transform(train[['Sex', 'Embarked', 'age_bin', 'fare_bin']]).astype('int64')

val_test[['Sex', 'Embarked', 'age_bin', 'fare_bin']] = ordinal_encoder.transform(val_test[['Sex', 'Embarked', 'age_bin', 'fare_bin']]).astype('int64')
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

X = train.drop('Survived', axis=1)

y = train['Survived']

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

for train_index, test_index in split.split(X, y):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y.loc[train_index], y.loc[test_index]
train.Survived.value_counts()/len(train)
y_train.value_counts() /len(y_train), y_test.value_counts() / len(y_test)
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

X_train = pd.DataFrame(standard_scaler.fit_transform(X_train), columns = X_train.columns)

X_test = pd.DataFrame(standard_scaler.transform(X_test), columns = X_test.columns)

val_test = pd.DataFrame(standard_scaler.transform(val_test), columns = val_test.columns)
X_train.head(), X_test.head(),
import time

from sklearn import tree

from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC, SVC

import xgboost

random_state=2

models = [tree.DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state), SGDClassifier(random_state=random_state), 

          LinearSVC(random_state=random_state, max_iter=10000), SVC(random_state=random_state, max_iter=10000),

            xgboost.XGBClassifier(random_state=random_state), AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1), 

          ExtraTreesClassifier(random_state=random_state), GradientBoostingClassifier(random_state=random_state)]

columns = ['Name', 'Score', 'RMSE', 'Precision', 'Recall', 'F1 Score']

models_compare = pd.DataFrame(columns=columns)

i=0

for model in models:

    start_time = time.time()

    clf = model

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    models_compare.loc[i, 'Name'] = clf.__class__.__name__

    models_compare.loc[i, 'Score'] = clf.score(X_test, y_test)

    models_compare.loc[i, 'RMSE'] = np.sqrt(mean_squared_error(y_test, predictions))

    models_compare.loc[i, 'Precision'] = precision_score(y_test, predictions)

    models_compare.loc[i, 'Recall'] = recall_score(y_test, predictions)

    models_compare.loc[i, 'F1 Score'] = f1_score(y_test, predictions)

    models_compare.loc[i, 'Execution time'] = time.time()- start_time

    i+=1

models_compare.sort_values(by='Score', ascending=False)
models_compare = models_compare.drop('Execution time', axis=1)

df = models_compare.melt('Name', var_name='Metrics',  value_name='')

import matplotlib.pyplot as plt



g = sns.catplot(x="Name", y="", hue='Metrics', kind='point', aspect=4, markers="o", linestyles= "--", data=df)
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold

models = [tree.DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state), SGDClassifier(random_state=random_state), 

          LinearSVC(random_state=random_state, max_iter=10000), SVC(random_state=random_state, max_iter=10000),

            xgboost.XGBClassifier(random_state=random_state), AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1), 

          ExtraTreesClassifier(random_state=random_state), GradientBoostingClassifier(random_state=random_state)]

columns = ['Name', 'Score', 'RMSE', 'Precision', 'Recall', 'F1 Score']

models_compare_cv = pd.DataFrame(columns=columns)

i=0

kfold = StratifiedKFold(n_splits=10)

for model in models: 

    start_time = time.time()

    clf = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    predictions = cross_val_predict(model, X_test, y_test, cv=5)

    mse = mean_squared_error(y_test, predictions)

    rmse =np.sqrt(mse)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    models_compare_cv.loc[i, 'Name'] = model.__class__.__name__

    models_compare_cv.loc[i, 'Score'] = clf.mean()

    models_compare_cv.loc[i, 'RMSE'] = rmse

    models_compare_cv.loc[i, 'Precision'] = precision

    models_compare_cv.loc[i, 'Recall'] = recall

    models_compare_cv.loc[i, 'F1 Score'] = f1_score(y_test, predictions)

    models_compare_cv.loc[i, 'Execution time'] = time.time()- start_time

    i+=1

models_compare_cv.sort_values(by='Score', ascending=False)
models_compare_cv = models_compare_cv.drop('Execution time', axis=1)

df = models_compare_cv.melt('Name', var_name='Metrics',  value_name='')

g = sns.catplot(x="Name", y="", hue='Metrics', kind='point', aspect=4, markers="o", linestyles= "--", data=df)
models_compare.sort_values('Score', ascending=False).Name.head()
models_compare_cv.sort_values('Score', ascending=False).Name.head()
from sklearn.model_selection import GridSearchCV

forest_grid = {'n_estimators': [10,100,1000], 'max_features':[2,4,6,8], 'bootstrap': [False], 'min_samples_leaf': [2, 4, 6, 8], 'max_depth': [10, 20, 30, 40]},

forest_reg = RandomForestClassifier(random_state=random_state)

grid_search_forest = GridSearchCV(forest_reg, forest_grid, cv=kfold, scoring='accuracy', return_train_score=True, n_jobs=-1)

start_time = time.time()

grid_search_forest.fit(X_train, y_train)

print('Execution time:', time.time() - start_time)
grid_search_forest.best_params_, grid_search_forest.best_score_
forest_be = grid_search_forest.best_estimator_
linear_svc_grid = [

    {'tol': [0.01, 0.1, 0.3, 0.5], 'loss': ['hinge', 'squared_hinge'], 'max_iter':[760000]}

]

linear_svc = LinearSVC(random_state=random_state)

grid_search_linearsvc = GridSearchCV(linear_svc, linear_svc_grid, cv=kfold, scoring='accuracy', return_train_score=True, n_jobs=-1)

start_time = time.time()

grid_search_linearsvc.fit(X_train, y_train)

print('Execution time:', time.time() - start_time)
grid_search_linearsvc.best_params_, grid_search_linearsvc.best_score_
linearsvc_be = grid_search_linearsvc.best_estimator_
gbc = GradientBoostingClassifier(random_state=random_state)

gbc_grid = {'loss': ['deviance'], 'n_estimators' : [100,200,300],

            'learning_rate' : [0.1, 0.05, 0.01], 'max_depth': [4,6,8],

            'min_samples_leaf': [2,3,4], 'max_features': [1.0,0.3, 0.1]}

grid_search_gbc = GridSearchCV(gbc, param_grid = gbc_grid, cv=kfold, scoring='accuracy', return_train_score=True, n_jobs=-1)

start_time = time.time()

grid_search_gbc.fit(X_train, y_train)

print('Execution time:', time.time() - start_time)
grid_search_gbc.best_params_, grid_search_gbc.best_score_
gbc_be = grid_search_gbc.best_estimator_
svc = SVC(probability=True, random_state=random_state)

svc_grid = {'kernel': ['rbf'], 'gamma' : [0.001, 0.01, 0.1], 'C':[1,5,10]}

grid_search_svc = GridSearchCV(svc, param_grid=svc_grid, cv=kfold, scoring='accuracy')

start_time = time.time()

grid_search_svc.fit(X_train, y_train)

print('Execution time:', time.time() - start_time)
grid_search_svc.best_params_, grid_search_svc.best_score_
svc_be = grid_search_svc.best_estimator_
xgb = xgboost.XGBClassifier(random_state=random_state)

xgb_grid = {'max_depth':[2,4,8], 'learning_rate':[0.001, 0.01,0.1,0.5], 'random_state': [random_state]}

grid_search_xgb = GridSearchCV(xgb, param_grid=xgb_grid, cv=kfold, scoring='accuracy')

start_time = time.time()

grid_search_xgb.fit(X_train, y_train)

print('Execution time:', time.time()- start_time)
grid_search_xgb.best_params_, grid_search_xgb.best_score_
xgb_be = grid_search_xgb.best_estimator_
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators= [('rf', forest_be), ('svc', svc_be), ('linear_svc', linearsvc_be), ('gb', gbc_be), ('xgb', xgb_be)], voting='hard')

voting_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

for clf in [forest_be, svc_be, linearsvc_be, gbc_be, xgb_be, voting_clf]:

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
test_final = val_test.copy()

voting_clf.fit(X_train, y_train)

voting_clf.score(test_final, gender_submission['Survived'])
test_final['Survived'] = voting_clf.predict(test_final)
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId'].copy().astype('int')

submission['Survived'] = test_final['Survived']

submission.to_csv('../working/submit.csv', index=False)