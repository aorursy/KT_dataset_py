# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
sns.color_palette()

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
df_train.head()
df_test = pd.read_csv("../input/test.csv")
df_test.head()
df_train.shape, df_test.shape
df_train.columns
df_test.columns
df_train.describe()
df_test.describe()
df_train.info()
df_test.info()
df_train['Survived'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(x='Survived', data=df_train)
ax.set_title('Survived Distribution')
ax.set_xlabel('Survived')
ax.set_ylabel('Count')
plt.show()
df_train['Sex'].value_counts()
df_test['Sex'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Sex', data=df_train)
ax.set_title('Distribution by Sex')
ax.set_xlabel('Sex')
ax.set_ylabel('Count')
plt.show()
df_train[df_train['Survived'] == 1]['Sex'].value_counts()
df_train[df_train['Survived'] == 0]['Sex'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Sex', data=df_train, hue='Survived')
ax.set_title('Survival by sex of traveller', fontsize=15)
ax.set_xlabel('Sex', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
df_train['Pclass'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Pclass', data=df_train)
ax.set_title('Passenger Distribution by class', fontsize=15)
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
df_train[df_train['Survived'] == 1]['Pclass'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(x="Pclass", data=df_train, hue='Survived')
ax.set_title('Survival by class')
ax.set_xlabel('Class')
ax.set_ylabel('Count')
plt.show()
df_train['Salutation'] = df_train['Name'].transform(lambda x : x.split(',')[1].split('.')[0])
df_train['Salutation'] = df_train['Salutation'].transform(lambda x: x.str.strip())
df_train['Salutation'].value_counts()
df_test['Salutation'] = df_test['Name'].transform(lambda x : x.split(',')[1].split('.')[0])
df_test['Salutation'] = df_test['Salutation'].transform(lambda x: x.str.strip())
df_test['Salutation'].value_counts()
df_train['Salutation'] = df_train['Salutation'].replace(['Ms', 'Mlle'], 'Miss')
df_test['Salutation'] = df_test['Salutation'].replace(['Ms', 'Mlle'], 'Miss')
df_train['Salutation'] = df_train['Salutation'].replace('Mme', 'Mrs')
df_train['Salutation'] = df_train['Salutation'].replace(['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Don', 
                        'Jonkheer', 'the Countess', 'Capt', 'Lady', 'Dona'], 'Other')
df_test['Salutation'] = df_test['Salutation'].replace(['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Don', 
                        'Jonkheer', 'the Countess', 'Capt', 'Lady', 'Dona'], 'Other')
df_train['Salutation'].value_counts()
fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Salutation', data=df_train)
ax.set_title('Salutation/Title Distribution', fontsize=15)
ax.set_xlabel('Salutation', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Salutation', data=df_train, hue='Survived')
ax.set_title('Salutation/Title Distribution', fontsize=15)
ax.set_xlabel('Salutation', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
df_test['Salutation'].value_counts()
fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Salutation', data=df_test)
ax.set_title('Salutation/Title Distribution', fontsize=15)
ax.set_xlabel('Salutation', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
df_train['Salutation'] = df_train['Salutation'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Other':4})
df_test['Salutation'] = df_test['Salutation'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Other':4})
df_train['Is_Alone'] = 0
for index, item in df_train.iterrows():
    if item['Parch'] ==0 and item['SibSp'] == 0:
        df_train.loc[index, 'Is_Alone'] = 1
df_train['Is_Alone'].value_counts()
fig, ax = plt.subplots(figsize=(10,10))
sns.countplot(x='Is_Alone', data=df_train)
ax.set_title('Lone Travellers Distribution', fontsize=15)
ax.set_xlabel('Is_Alone', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(x='Is_Alone', data=df_train, hue='Survived')
ax.set_title('Survival of Passengers travelled alone', fontsize=15)
ax.set_xlabel('Is_Alone', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
df_test['Is_Alone'] = 0
for index, item in df_test.iterrows():
    if item['Parch'] ==0 and item['SibSp'] == 0:
        df_test.loc[index, 'Is_Alone'] = 1
fig, ax = plt.subplots(figsize=(10,10))
sns.countplot(x='Is_Alone', data=df_test)
ax.set_title('Lone Travellers Distribution - Test', fontsize=15)
ax.set_xlabel('Is_Alone', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
df_train['Cabin'] = df_train['Cabin'].transform(lambda x: 0 if pd.isnull(x) else 1)
df_train['Cabin'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Cabin', data=df_train)
ax.set_title('Cabin Distribution', fontsize=15)
ax.set_xlabel('Has Cabin', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(x='Cabin', data=df_train, hue='Survived')
ax.set_title('Survival of travellers with Cabin', fontsize=15)
ax.set_xlabel('Cabin', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
df_test['Cabin'] = df_test['Cabin'].transform(lambda x: 0 if pd.isnull(x) else 1)
df_test['Cabin'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(x='Cabin', data=df_test)
ax.set_title('Cabin Distribution', fontsize=15)
ax.set_xlabel('Has Cabin', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_train['Embarked'].value_counts()
fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Embarked', data=df_train)
ax.set_title('Embarked Distribution')
ax.set_xlabel('Embarked')
ax.set_ylabel('Count')
plt.show()
fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Embarked', data=df_train, hue='Survived')
ax.set_title('Embarked Distribution')
ax.set_xlabel('Embarked')
ax.set_ylabel('Count')
plt.show()
df_test['Embarked'].value_counts()
fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Embarked', data=df_test)
ax.set_title('Embarked Distribution')
ax.set_xlabel('Embarked')
ax.set_ylabel('Count')
plt.show()
df_train['Embarked'] = df_train['Embarked'].map({'C':0, 'S':1, 'Q':2})
df_test['Embarked'] = df_test['Embarked'].map({'C':0, 'S':1, 'Q':2})
df_train.drop('Ticket', inplace=True, axis=1)
df_test.drop('Ticket', inplace=True, axis=1)
df_test['Fare'].describe()
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
pd.qcut(df_train['Fare'], 4).unique()
df_train.loc[df_train['Fare'] <= 7.91, 'Fare'] = 0
df_train.loc[((df_train['Fare'] > 7.91) & (df_train['Fare'] <= 14.454)), 'Fare'] = 1
df_train.loc[((df_train['Fare'] > 14.454) & (df_train['Fare'] <= 31)), 'Fare'] = 2
df_train.loc[df_train['Fare'] > 31, 'Fare'] = 3
df_test.loc[df_test['Fare'] <= 7.91, 'Fare'] = 0
df_test.loc[((df_test['Fare'] > 7.91) & (df_test['Fare'] <= 14.454)), 'Fare'] = 1
df_test.loc[((df_test['Fare'] > 14.454) & (df_test['Fare'] <= 31)), 'Fare'] = 2
df_test.loc[df_test['Fare'] > 31, 'Fare'] = 3
df_train['Fare'] = df_train['Fare'].astype('int')
df_test['Fare'] = df_test['Fare'].astype('int')
df_train['Fare'].value_counts()
fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Fare', data=df_train)
ax.set_title('Fare Distribution', fontsize=15)
ax.set_xlabel('Fare', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x='Fare', data=df_train, hue='Survived')
ax.set_title('Fare Distribution', fontsize=15)
ax.set_xlabel('Fare', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()
df_train['Age'].isna().sum()
df_test['Age'].isna().sum()
age_avg   = df_train['Age'].mean()
age_std  = df_train['Age'].std()
age_null_count = df_train['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
df_train['Age'][np.isnan(df_train['Age'])] = age_null_random_list
df_train['Age'] = df_train['Age'].astype(int)
    
pd.cut(df_train['Age'], 5).unique()
df_train.loc[df_train['Age']<=16, 'Age'] = 0
df_train.loc[((df_train['Age']>16)&(df_train['Age']<=32)), 'Age'] = 1
df_train.loc[((df_train['Age']>32)&(df_train['Age']<=48)), 'Age'] = 2
df_train.loc[((df_train['Age']>48)&(df_train['Age']<=64)), 'Age'] = 3
df_train.loc[df_train['Age']>64, 'Age'] = 4
age_avg   = df_test['Age'].mean()
age_std  = df_test['Age'].std()
age_null_count = df_test['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
df_test['Age'][np.isnan(df_test['Age'])] = age_null_random_list
df_test['Age'] = df_test['Age'].astype(int)
df_test.loc[df_test['Age']<=16, 'Age'] = 0
df_test.loc[((df_test['Age']>16)&(df_test['Age']<=32)), 'Age'] = 1
df_test.loc[((df_test['Age']>32)&(df_test['Age']<=48)), 'Age'] = 2
df_test.loc[((df_test['Age']>48)&(df_test['Age']<=64)), 'Age'] = 3
df_test.loc[df_test['Age']>64, 'Age'] = 4
df_test.head()
df_train.drop('Name', axis=1, inplace=True)
df_test.drop('Name', axis=1, inplace=True)
df_train.head()
df_test.head()
df_train.dtypes
df_test.dtypes
PassengerId = df_test['PassengerId'].ravel()
y_all = df_train['Survived'].ravel()
df_train.drop('PassengerId', axis=1, inplace=True)
df_test.drop('PassengerId', axis=1, inplace=True)
X_all = df_train.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(X_all.corr(), annot=True)
ax.set_title('Correlation of training set')
plt.show()
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(df_test.corr(), annot=True)
ax.set_title('Correlation of test set')
plt.show()
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
acc_scorer = make_scorer(accuracy_score)
'''
clf = RandomForestClassifier()
rf_params = {
    "n_estimators": [100, 300, 500, 1000],
    "bootstrap": [True, False],
    "criterion": ['gini', 'entropy'],
    "warm_start": [True, False],
    "max_depth": [2, 4, 6],
    "max_features": ['sqrt', 'log2'],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [2, 4, 6]
}

clf = ExtraTreesClassifier()
xt_params = {
    "n_estimators":[100, 300, 500, 1000],
    "bootstrap": [True, False],
    "criterion": ['gini', 'entropy'],
    "warm_start": [True, False],
    "max_depth": [2, 4, 6],
    "max_features": ['sqrt', 'log2'],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [2, 4, 6]
}

clf = AdaBoostClassifier()
ad_params = {
    "n_estimators":[100, 300, 500, 1000],
    "learning_rate": [0.1, 0.3, 0.5, 0.75, 1]
}
clf = GradientBoostingClassifier()
gb_params = {
    "n_estimators":[100, 300, 500, 1000],
    "learning_rate": [0.1, 0.3, 0.5, 0.75, 1],
    "warm_start": [True, False],
    "max_depth": [2, 4, 6],
    "max_features": ['sqrt', 'log2'],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [2, 4, 6]
}
clf = SVC()
sv_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [0.01, 0.1, 1, 10, 100]},
                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]'''
#grid_search = GridSearchCV(clf, param_grid=sv_params, scoring=acc_scorer)
#grid_search.fit(X_train, y_train)
#grid_search.best_estimator_
rf_clf = RandomForestClassifier(bootstrap=False, class_weight=None,
            criterion='entropy', max_depth=6, max_features='log2',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=4,
            min_samples_split=4, min_weight_fraction_leaf=0.0,
            n_estimators=300, n_jobs=None, oob_score=False,
            random_state=42, verbose=0, warm_start=True)

et_clf = ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='entropy',
           max_depth=6, max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
           oob_score=False, random_state=42, verbose=0, warm_start=False)

ad_clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.1, n_estimators=300, random_state=42)

gb_clf = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=2,
              max_features='log2', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=2, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=300,
              n_iter_no_change=None, presort='auto', random_state=42,
              subsample=1.0, tol=0.0001, validation_fraction=0.1,
              verbose=0, warm_start=False)

sv_clf = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.001, verbose=False)
rf_clf.fit(X_train, y_train)
et_clf.fit(X_train, y_train)
ad_clf.fit(X_train, y_train)
gb_clf.fit(X_train, y_train)
sv_clf.fit(X_train, y_train)
rf_rank = rf_clf.feature_importances_
et_rank = et_clf.feature_importances_
ad_rank = ad_clf.feature_importances_
gb_rank = gb_clf.feature_importances_
df_feature_importance = pd.DataFrame({
    'Features': X_all.columns,
    'Random_Forest': rf_rank,
    'Extra_Trees': et_rank,
    'AdaBoost': ad_rank,
    'Gradient_Boost': gb_rank
})
df_feature_importance
fig, ax=plt.subplots(figsize=(10, 8))
sns.barplot(x=df_feature_importance['Features'], y=df_feature_importance['Random_Forest'])
ax.set_title('Random Forest Feature Importance', fontsize=12)
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_xlabel('Column Name', fontsize=12)
plt.show()
fig, ax=plt.subplots(figsize=(10, 8))
sns.barplot(x=df_feature_importance['Features'], y=df_feature_importance['Extra_Trees'])
ax.set_title('Extra Trees Feature Importance', fontsize=12)
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_xlabel('Column Name', fontsize=12)
plt.show()
fig, ax=plt.subplots(figsize=(10, 8))
sns.barplot(x=df_feature_importance['Features'], y=df_feature_importance['AdaBoost'])
ax.set_title('AdaBoost Feature Importance', fontsize=12)
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_xlabel('Column Name', fontsize=12)
plt.show()
fig, ax=plt.subplots(figsize=(10, 8))
sns.barplot(x=df_feature_importance['Features'], y=df_feature_importance['Gradient_Boost'])
ax.set_title('Gradient Boosting Importance', fontsize=12)
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_xlabel('Column Name', fontsize=12)
plt.show()
rf_pred = rf_clf.predict(X_test)
et_pred = et_clf.predict(X_test)
ad_pred = ad_clf.predict(X_test)
gb_pred = gb_clf.predict(X_test)
sv_pred = sv_clf.predict(X_test)
print('Random Forest Accuracy: {0:.2f}'.format(accuracy_score(y_test, rf_pred) * 100))
print('Extra Trees Accuracy: {0:.2f}'.format(accuracy_score(y_test, et_pred) * 100))
print('AdaBoost Accuracy: {0:.2f}'.format(accuracy_score(y_test, ad_pred) * 100))
print('Gradient Boosting Accuracy: {0:.2f}'.format(accuracy_score(y_test, gb_pred) * 100))
print('SVM Accuracy: {0:.2f}'.format(accuracy_score(y_test, sv_pred) * 100))
def KFold_pred(clf, X_all, y_all):
    outcomes = []
    test_scores = []
    kf = KFold(n_splits=5, random_state=42, shuffle=False)
    for i, (train_index, test_index) in enumerate(kf.split(X_all)):
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        outcomes.append(acc)
        print('Fold {0} accuracy {1:.2f}'.format(i, acc))
    mean_accuracy = np.mean(outcomes)
    return mean_accuracy
rf_pred = KFold_pred(rf_clf, X_all, y_all)
print('Random Forest 5 folds mean accuracy: {0:.2f}'.format(rf_pred))
et_pred = KFold_pred(et_clf, X_all, y_all)
print('Extra Trees 5 folds mean accuracy: {0:.2f}'.format(et_pred))
ad_pred = KFold_pred(ad_clf, X_all, y_all)
print('AdaBoost 5 folds mean accuracy: {0:.2f}'.format(ad_pred))
gb_pred = KFold_pred(gb_clf, X_all, y_all)
print('Gradient Boosting 5 folds mean accuracy: {0:.2f}'.format(gb_pred))
sv_pred = KFold_pred(sv_clf, X_all, y_all)
print('SVM 5 folds mean accuracy: {0:.2f}'.format(sv_pred))
def oof_pred(clf, X_all, y_all, df_test):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    oof_train = np.zeros(X_all.shape[0])
    oof_test = np.zeros(df_test.shape[0])
    oof_test_kf = np.empty((kf.get_n_splits(), df_test.shape[0]))
    for i, (train_index, test_index) in enumerate(kf.split(X_all)):
        X_train = X_all.values[train_index]
        y_train = y_all[train_index]
        X_test = X_all.values[test_index]
        y_test = y_all[test_index]
        
        clf.fit(X_train, y_train)
        oof_train[test_index] = clf.predict(X_test)
        oof_test_kf[i, :] = clf.predict(df_test)
    oof_test = oof_test_kf.mean(axis=0)
    return oof_train, oof_test
rf_oof_train, rf_oof_test = oof_pred(rf_clf, X_all, y_all, df_test)
et_oof_train, et_oof_test = oof_pred(et_clf, X_all, y_all, df_test)
ad_oof_train, ad_oof_test = oof_pred(ad_clf, X_all, y_all, df_test)
gb_oof_train, gb_oof_test = oof_pred(gb_clf, X_all, y_all, df_test)
sv_oof_train, sv_oof_test = oof_pred(sv_clf, X_all, y_all, df_test)
base_level_train = pd.DataFrame({
    'Random_Forest':rf_oof_train,
    'Extra_Trees': et_oof_train,
    'AdaBoost': ad_oof_train,
    'Gradient_Boost':gb_oof_train,
    'Support_Vector':sv_oof_train
})
base_level_train.head()
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(base_level_train.corr(), annot=True)
plt.show()
base_level_test = pd.DataFrame({
    'Random_Forest':rf_oof_test,
    'Extra_Trees': et_oof_test,
    'AdaBoost': ad_oof_test,
    'Gradient_Boost':gb_oof_test,
    'Support_Vector':sv_oof_test
})
base_level_test.head()
import xgboost as xgb
xg_clf = xgb.XGBClassifier(
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1)
xg_clf.fit(base_level_train, y_all)
predictions = xg_clf.predict(base_level_test)
output = pd.DataFrame({ 'PassengerId' : PassengerId, 'Survived': predictions.astype(int) })
output.to_csv('titanic-predictions.csv', index = False)
output.tail()
