import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

IDtest = test["PassengerId"]
# concat all data to analyze

dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

len(dataset)
train_len = len(train)
dataset.head()
dataset.info()
# null includes: Age, Cabin, Embarked

dataset.isnull().sum()
dataset.describe()
dataset[dataset['Survived'] == 1]
plt.figure(figsize=(12, 8))

plt.title('Pearson correlation of Features')

sns.heatmap(train.drop(labels=['PassengerId'], axis=1).corr(), cmap=plt.cm.RdBu, annot=True)
import re

def get_title(name):

    title_search = re.search(r'([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    

    return ''



dataset['Title'] = dataset['Name'].apply(get_title)
dataset['Title'].value_counts()
#replacing all titles with mr, mrs, miss, master, rare

def replace_titles(title):

    if title in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:

        return 'Rare'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



dataset['Title'] = dataset['Title'].apply(replace_titles)
dataset['Title'].value_counts()
# Mapping titles

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

dataset['Title'] = dataset['Title'].map(title_mapping)
sns.factorplot(x='Title', y='Survived', data=dataset, kind='bar')
dataset['HasCabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
sns.factorplot(x='HasCabin', y='Survived', data=dataset, kind='bar')
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['FamilySize'].value_counts()
sns.factorplot(x='FamilySize', y='Survived', data=dataset)
dataset['Embarked'].value_counts()
dataset['Embarked'].fillna('S', inplace=True)
sns.factorplot(x='Embarked', y='Survived', data=dataset, kind='bar')
dataset["Sex"] = dataset["Sex"].map({"male": 1, "female": 0})
sns.factorplot(x='Sex', y='Survived', data=dataset, kind='bar')
plt.figure(figsize=(8, 6))

plt.title('Pearson correlation of Age')

sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(), cmap=plt.cm.RdBu, annot=True)
# Explore Age vs Survived

g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Age")
# Explore Age vs Parch , Pclass and SibSP

g = sns.factorplot(y="Age",x="Pclass", data=dataset,kind="box")

g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box")

g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box")
# Filling missing value of Age 



## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



age_med = dataset["Age"].median()

for i in index_NaN_age :

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) &

                                (dataset['Parch'] == dataset.iloc[i]["Parch"]) &

                                (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med

dataset[:len(train)][['Age', 'Survived']].groupby(by=['Age']).sum()
dataset.head()
dataset['NameParenthesis'] = dataset['Name'].apply(lambda x: 1 if re.search(r'\(.*?\)', x) else 0)
dataset[:train_len][['NameParenthesis', 'Survived']].groupby(by=['NameParenthesis']).sum()
dataset[:train_len]['NameParenthesis'].value_counts()
sns.factorplot(x='NameParenthesis', y='Survived', data=dataset, kind='bar')
dataset['NameLength']= dataset['Name'].apply(len)
sns.factorplot(x='NameLength', y='Survived', data=dataset, kind='bar')
dataset[:train_len][['NameLength', 'NameParenthesis', 'Survived']].groupby(by=['NameLength'], as_index=False).mean()
plt.figure(figsize=(12, 8))

sns.heatmap(dataset.corr(), cmap=plt.cm.RdBu, annot=True)
dataset.head(3)
#dataset = dataset.drop(labels=['PassengerId', 'Name', 'Cabin', 'Parch', 'SibSp', 'Ticket'], axis=1)
dataset = dataset.drop(['Cabin', 'Name', 'PassengerId', 'Ticket', ], axis=1)
dataset.head()
# Mapping Fare

dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

dataset['Fare'] = dataset['Fare'].astype(int)



# Mapping Age

dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

dataset['Age'] = dataset['Age'].astype(int)



# Mapping Embarked

dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# dataset['IsAlone'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)
# sns.factorplot(x='IsAlone', y='Survived', data=dataset, kind='bar')
# Create new feature of family size

dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallFamily'] = dataset['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedFamily'] = dataset['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeFamily'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
# one-hot values Title and Embarked 

dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"])

dataset = pd.get_dummies(dataset, columns = ["Fare"])

dataset = pd.get_dummies(dataset, columns = ["Age"])
dataset.head()
## Separate train dataset and test dataset

train = dataset[:len(train)]

test = dataset[len(train):]

test.drop(labels=["Survived"],axis = 1,inplace=True)
## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)

X_train = train.drop(labels = ["Survived"],axis = 1)

y_train = train["Survived"]
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier, \

                              ExtraTreesClassifier,AdaBoostClassifier,\

                              BaggingClassifier, VotingClassifier)

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from pprint import pformat
random_state = 2
cv = StratifiedShuffleSplit(n_splits=10,)
clf_list = [

    GradientBoostingClassifier(),

    RandomForestClassifier(),

    ExtraTreesClassifier(),

    AdaBoostClassifier(),

    BaggingClassifier(),

    DecisionTreeClassifier(),

    SVC(),

    KNeighborsClassifier(),

    LogisticRegression(),

    GaussianNB(),

    LGBMClassifier(),

    XGBClassifier(),

]

accuracy_dict = {}

for clf in clf_list:

    acc = cross_val_score(clf, X_train, y=y_train, cv=cv, scoring = "accuracy")    

    accuracy_dict[clf.__class__.__name__] = [acc.min(), acc.mean(), acc.max()]
accuracy_df = pd.DataFrame(accuracy_dict).transpose()

accuracy_df
accuracy_df.plot(kind='bar',rot=60)
gbm = LGBMClassifier(num_leaves=20,

                        learning_rate=0.5,

                        n_estimators=100)

gbm.fit(X_train, y_train,

        eval_metric='l1')

print('Feature importances:', list(gbm.feature_importances_))
from sklearn.metrics import accuracy_score

predictions = gbm.predict(X_train)

accuracy_score(predictions, y_train)
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
cv = StratifiedKFold(n_splits=10)
grid_seed = [2]
# GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc_params = {

    'n_estimators': [100, 200, 300], #default=100

    'learning_rate': [0.01, 0.03, 0.1, 0.3], #default=0.1

    'random_state': grid_seed,

    'max_depth': [4, 8],

    'min_samples_leaf': [1, 3, 10],

}



gbc_search = GridSearchCV(gbc, param_grid=gbc_params, n_jobs=-1, cv=cv, verbose=1)

gbc_search.fit(X_train, y_train)

gbc_best = gbc_search.best_estimator_

print(gbc_search.best_params_)

gbc_search.best_score_
# RandomForestClassifier

rf = RandomForestClassifier()

rf_params = {

    "max_features": [1, 3, 10],

    "min_samples_split": [2, 3, 10],

    "min_samples_leaf": [1, 3, 10],

    "bootstrap": [False],

    "n_estimators" :[100, 300],

    "criterion": ["gini"],

    'random_state': grid_seed,

}



rf_search = GridSearchCV(rf, param_grid=rf_params, n_jobs=-1, cv=cv, verbose=1)

rf_search.fit(X_train, y_train)

rf_best = rf_search.best_estimator_

print(rf_search.best_params_)

rf_search.best_score_
# BaggingClassifier

bagging = BaggingClassifier()

bagging_params = {

    'n_estimators': [100, 300], 

    'max_samples': [0.1, 0.3],

    'random_state': grid_seed,

}



bagging_search = GridSearchCV(bagging, param_grid=bagging_params, n_jobs=-1, cv=cv, verbose=1)

bagging_search.fit(X_train, y_train)

bagging_best = bagging_search.best_estimator_

print(bagging_search.best_params_)

bagging_search.best_score_
# AdaBoostClassifier

DTC = DecisionTreeClassifier()

ada = AdaBoostClassifier(DTC)

ada_params = {

    "base_estimator__criterion": ["gini", "entropy"],

    "base_estimator__splitter": ["best", "random"],

    "algorithm" : ["SAMME","SAMME.R"],

    "n_estimators" :[100, 300],

    "learning_rate":  [0.001, 0.003, 0.01, 0.03, 0.1],

    'random_state': grid_seed,

}



ada_search = GridSearchCV(ada, param_grid=ada_params, n_jobs=-1, cv=cv, verbose=1)

ada_search.fit(X_train, y_train)

ada_best = ada_search.best_estimator_

print(ada_search.best_params_)

ada_search.best_score_
# ExtraTreesClassifier

ext = ExtraTreesClassifier()

ext_params = {

    "max_features": [1, 3, 10],

    "min_samples_split": [2, 3, 10],

    "min_samples_leaf": [1, 3, 10],

    "bootstrap": [False],

    "n_estimators": [100, 300],

    "criterion": ["gini"],

    'random_state': grid_seed,

}



ext_search = GridSearchCV(ext, param_grid=ext_params, n_jobs=-1, cv=cv, verbose=1)

ext_search.fit(X_train, y_train)

ext_best = ext_search.best_estimator_

print(ext_search.best_params_)

ext_search.best_score_
# XGBClassifier

xgb = XGBClassifier()

xgb_params = {

    'learning_rate': [0.01, 0.03, 0.1], 

    'max_depth': [1, 3, 4, 6], 

    'n_estimators': [100, 300], 

    'seed': grid_seed 

}



xgb_search = GridSearchCV(xgb, param_grid=xgb_params, n_jobs=-1, cv=cv, verbose=1)

xgb_search.fit(X_train, y_train)

xgb_best = xgb_search.best_estimator_

print(xgb_search.best_params_)

xgb_search.best_score_
# LGBMClassifier

lgbm = LGBMClassifier()

lgbm_params = {

    'num_leaves': [10, 20, 30],

    'learning_rate': [0.001, 0.01, 0.03, 0.1], 

    'max_depth': [1, 3, 6, 10], 

    'n_estimators': [100, 300], 

    'seed': grid_seed 

}



lgbm_search = GridSearchCV(lgbm, param_grid=lgbm_params, n_jobs=-1, cv=cv, verbose=1)

lgbm_search.fit(X_train, y_train)

lgbm_best = lgbm_search.best_estimator_

print(lgbm_search.best_params_)

lgbm_search.best_score_
# KNeighborsClassifier

knn = KNeighborsClassifier()

knn_params = {

    'n_neighbors': [1,2,3,4,5,6,7], #default: 5

    'weights': ['uniform', 'distance'], #default = ‘uniform’

    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

}



knn_search = GridSearchCV(knn, param_grid=knn_params, n_jobs=-1, cv=cv, verbose=1)

knn_search.fit(X_train, y_train)

knn_best = knn_search.best_estimator_

print(knn_search.best_params_)

knn_search.best_score_
### SVC

svc = SVC(probability=True)

svc_params = {'kernel': ['rbf'], 

              'gamma': [ 0.001, 0.01, 0.1, 1],

              'C': [1, 10, 50, 100, 200, 300, 1000]}



svc_search = GridSearchCV(svc, param_grid=svc_params, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)

svc_search.fit(X_train,y_train)



svc_best = svc_search.best_estimator_

print(svc_search.best_params_)

svc_search.best_score_
ensemble_clfs = {

    'gbc': gbc_best,

    'rf': rf_best,

    'ada': ada_best,

    'bag': bagging_best,

    'ext': ext_best,

    'xgb': xgb_best,

    'lgbm': lgbm_best,

    'svc': svc_best,

    'knn': knn_best,

}
pred_dict = {}

for clf in ensemble_clfs.values():

    pred_dict[clf.__class__.__name__] = clf.predict(test)

sns.heatmap(pd.DataFrame(pred_dict).corr(),annot=True)
voting_soft = VotingClassifier(estimators=ensemble_clfs.items(), voting='soft', n_jobs=-1)
voting_soft.fit(X_train, y_train)

accuracy_score(voting_soft.predict(X_train), y_train)
# voting_hard = VotingClassifier(estimators=ensemble_clfs.items(), voting='hard', n_jobs=-1)
# voting_hard.fit(X_train, y_train)

# accuracy_score(voting_hard.predict(X_train), y_train)
# lgbm = LGBMClassifier(num_leaves=20, learning_rate=0.01, n_estimators=300, max_depth=6)

# print(cross_val_score(lgbm, X_train, y_train, cv=cv))

# # print(cross_validate(lgbm, X_train, y_train, cv=cv))

#lgbm.fit(X_train, y_train)
# xgbt = XGBClassifier(learning_rate=0.03, max_depth=3, n_estimators=300)

# cross_val_score(xgbt, X_train, y_train, cv=cv)
#xgbt.fit(X_train, y_train)
test_Survived = pd.Series(voting_soft.predict(test), name="Survived")



results = pd.concat([IDtest, test_Survived],axis=1)



results.to_csv("titanic_with_ensemble.csv",index=False)