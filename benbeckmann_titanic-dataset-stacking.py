import os

import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

from collections import Counter

from scipy import stats



from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn.ensemble import (

    RandomForestClassifier,

    ExtraTreesClassifier,

    AdaBoostClassifier,

    GradientBoostingClassifier

)

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from xgboost import XGBClassifier

from vecstack import stacking



print(os.listdir('../input'))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.columns = train.columns.str.lower()

test.columns = test.columns.str.lower()



test_id = test['passengerid']



dataset = pd.concat([train, test], axis=0)
dataset_backup = dataset.copy(deep=True)
dataset = dataset_backup.copy(deep=True)
pd.DataFrame([dataset.columns, dataset.dtypes, dataset.isnull().sum(), dataset.nunique()]).T
sns.pairplot(train)
numerical = ['age', 'fare', 'sibsp', 'parch']

categorical = ['cabin', 'embarked', 'pclass', 'sex']
fig, ax = plt.subplots(1, 4, figsize=(15, 5))

fig.tight_layout()

ax = iter(ax.flatten())

for feat in numerical:

    sns.boxplot(x='survived', y=feat, data=train, ax=next(ax))

    

# age and age bins for more obvious correlation?

# family_size and is_alone feature?
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

fig.tight_layout()

ax = iter(ax.flatten())

for feat in ['sibsp', 'parch']:

    sns.barplot(x=feat, y='survived', data=train, ax=next(ax))
# some basic cleaning and feature engineering



# probably more realistic if filled separately for train and test sets, since this actually "leaks" test data into train

# maybe fill in per category most correlated with age

mean_age = dataset.age.mean()

std_age = dataset.age.std()

dataset.loc[dataset.age.isnull(), 'age'] = np.random.randint(mean_age - std_age, mean_age + std_age, len(dataset[dataset.age.isnull()]))

# separate into bins

dataset['age_bin'] = pd.cut(dataset.age, bins=list(range(0, int(train.age.max()) + 10, 5)), include_lowest=True)

# dataset['age_bin'] = dataset.age_bin.apply(lambda x: f'>{int(x.left)} - {int(x.right)}').apply(lambda x: f'>0{x[1:]}' if len(x[1:].split(' - ')[0]) == 1 else x ).astype(str)



# fare bin

dataset['fare'] = dataset.fare.fillna(dataset.fare.median()).apply(lambda x: np.log1p(x))

dataset['fare_bin'] = pd.cut(dataset.fare, bins=30, include_lowest=True)



# family size

dataset['family_size'] = dataset.parch + dataset.sibsp + 1



# is alone

dataset['is_alone'] = dataset.family_size == 1
fig, ax = plt.subplots(1, 4, figsize=(15, 5))

ax = iter(ax.flatten())

data = dataset[~dataset.passengerid.isin(test_id)][['age_bin', 'fare_bin', 'family_size', 'is_alone', 'survived']]

for feat in data.columns[:-1]:

    labels = sorted(data[feat].unique())

    g = sns.barplot(x=feat, y='survived', data=data, order=labels, ax=next(ax))

    g.set_xticklabels(labels=labels, rotation=90)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

fig.tight_layout()

ax = iter(ax.flatten())

for feat in categorical[1:]:

    sns.barplot(x=feat, y='survived', data=train, ax=next(ax))

    

# features seems to have a good predictability for the target outcome
# fill na's and feature engineering



dataset['embarked'] = dataset.embarked.fillna(dataset.embarked.mode()[0])



# assuming people with NA cabin has no cabin, check to see if having a cabin increases chance of survival

dataset['has_cabin'] = dataset.cabin.notnull()
sns.barplot(x='has_cabin', y='survived', data=dataset[~dataset.passengerid.isin(test_id)])
# check if title impacts survivability

dataset['title'] = dataset.name.apply(lambda x: x.split(', ')[1].split(' ')[0]).apply(lambda x: x if x in ['Mr.', 'Mrs.', 'Miss.', 'Master.'] else 'Others')
sns.barplot(x='title', y='survived', data=dataset[~dataset.passengerid.isin(test_id)])
train = dataset[~dataset.passengerid.isin(test_id)]

test = dataset[dataset.passengerid.isin(test_id)]
train.columns
fig, ax = plt.subplots(3, 1, figsize=(15, 15))

fig.tight_layout()

ax = iter(ax.flatten())

labels = sorted(train.age_bin.unique())

for feat in ['sex', 'pclass', 'embarked']:

    _ax = next(ax)

#     _ax.grid(False)

    g = sns.barplot(x='age_bin', y='survived', hue=feat, data=train, order=labels, ax=_ax)

    g.set_xticklabels(labels=labels, rotation=15)

    

#     twin_ax = _ax.twinx()

#     twin_ax.grid(False)

#     data = train.age_bin.value_counts().to_frame().reset_index().rename(columns={'age_bin':'count'})

#     sns.barplot(x='index', y='count', data=data, ax=twin_ax, alpha=0.5)



# clearly, male passengers has a lot less chance of surviving than females

# class 1 and 2 passengers are also prioritized

# for some reason, those who embarked from C and Q has a higher chance of surviving
for feat in ['sex', 'pclass', 'embarked', 'is_alone', 'has_cabin']:

    sns.FacetGrid(train, hue='survived', col=feat, height=5, aspect=1).map(sns.distplot, 'fare')

# blue: survived = 0

# orange: survived = 1

# the trend seems to be the same for all, the higher the fare, the higher the chance of survival
feats = ['embarked', 'pclass', 'sex', 'age_bin', 'fare_bin', 'family_size', 'is_alone', 'has_cabin', 'title']

dataset_y = dataset[['survived']]

dataset_X = dataset[feats]

dataset_X = dataset_X.apply(LabelEncoder().fit_transform)
# check correlation before onehot

idx = len(dataset) - len(test_id)

sns.heatmap(pd.concat([dataset_X[:idx], dataset_y[:idx]], axis=1).astype(int).corr(), annot=True)
onehot = ['embarked', 'title']

onehot_df = pd.get_dummies(dataset_X[onehot].astype(object), drop_first=True)

dataset_X = pd.concat([dataset_X.drop(onehot, axis=1), onehot_df], axis=1)
X_train = dataset_X[:idx]

X_test = dataset_X[idx:]

y_train = dataset_y[:idx]
kfold = StratifiedKFold(n_splits=6)
clfs_name = [

    'RandomForest',

    'ExtraTrees',

    'AdaBoost',

    'GradientBoost',

    'XGBoost',

    'SVC'

]



clfs = [

    RandomForestClassifier(random_state=0),

    ExtraTreesClassifier(random_state=0),

    AdaBoostClassifier(random_state=0),

    GradientBoostingClassifier(random_state=0),

    XGBClassifier(random_state=0),

    SVC(random_state=0)

]



results = list()

for name, clf in zip(clfs_name, clfs):

    cv = cross_val_score(clf, X_train, y_train, cv=kfold, n_jobs=-1)

    results.append([name, cv.mean() - cv.std(), cv.mean() + cv.std()])
pd.DataFrame(results, columns=['clf', '-1 std', '+1 std'])
clf_1_grid = {

    'n_estimators': [50, 100, 300, 500],

    'learning_rate': [0.1, 0.5, 0.9, 1.0]

}

clf_1 = GridSearchCV(AdaBoostClassifier(random_state=0), clf_1_grid, n_jobs=-1, cv=kfold)

clf_1.fit(X_train, y_train)



clf_1_score = clf_1.best_score_

clf_1_estimator = clf_1.best_estimator_
clf_2_grid = {

    'n_estimators': [40, 50, 70, 100],

    'learning_rate': [0.1, 0.2, 0.3],

    'subsample': [0.4, 0.5, 0.6],

    'max_depth': [3, 4, 5]

}

clf_2 = GridSearchCV(GradientBoostingClassifier(random_state=0), clf_2_grid, n_jobs=-1, cv=kfold)

clf_2.fit(X_train, y_train)



clf_2_score = clf_2.best_score_

clf_2_estimator = clf_2.best_estimator_
clf_3_grid = {

    'n_estimators': [50, 100, 300, 500],

    'learning_rate': [0.1, 0.5, 0.9],

    'subsample': [0.3, 0.5, 1.0],

    'max_depth': [3, 4, 5]

}

clf_3 = GridSearchCV(XGBClassifier(random_state=0), clf_3_grid, n_jobs=-1, cv=kfold)

clf_3.fit(X_train, y_train)



clf_3_score = clf_3.best_score_

clf_3_estimator = clf_3.best_estimator_
clf_4_grid = {

    'C': [0.5, 0.6, 0.7, 0.8, 0.9]

}

clf_4 = GridSearchCV(SVC(random_state=0), clf_4_grid, n_jobs=-1, cv=kfold)

clf_4.fit(X_train, y_train)



clf_4_score = clf_4.best_score_

clf_4_estimator = clf_4.best_estimator_
feature_importance = pd.DataFrame([

    clf_1_estimator.feature_importances_,

    clf_2_estimator.feature_importances_,

    clf_3_estimator.feature_importances_

], columns=X_train.columns)



feature_importance['model'] = ['adaboost', 'gradientboost', 'xgb']
feature_importance = feature_importance.set_index('model').unstack().reset_index().rename(columns={'level_0':'feat', 0: 'score'})
fig, ax = plt.subplots(1, 1, figsize=(20, 5))

g = sns.barplot(x='feat', y='score', hue='model', data=feature_importance, ax=ax)
feature_importance.sort_values(['model', 'score'], ascending=False).groupby('model', sort=False).head(4)
# different feature importances between models is good for stacking

# stacked model generalizes better since sub models are computes differently

S_train, S_test = stacking(

    [clf_1_estimator, clf_2_estimator, clf_3_estimator, clf_4_estimator],

    X_train, y_train, X_test, regression=False, n_folds=10,

    stratified=True, shuffle=True, random_state=0, verbose=2

)
sclf = GridSearchCV(XGBClassifier(random_state=0), clf_3_grid, n_jobs=-1, cv=kfold)

sclf.fit(S_train, y_train)
sclf = sclf.best_estimator_.fit(S_train, y_train)
y_pred = sclf.predict(S_test).astype(int)
submit = pd.DataFrame({

    'PassengerId': test_id,

    'Survived': y_pred

})



submit.to_csv('titanic_submission.csv', index=False)
from IPython.display import FileLink

FileLink('titanic_submission.csv')