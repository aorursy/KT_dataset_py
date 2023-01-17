import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.metrics import classification_report

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score, log_loss, accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

import lightgbm as lgbm



import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.shape, test_df.shape
train_df.head()
y = train_df['Survived']

train_df.drop('Survived', axis=1, inplace=True)



combin = pd.concat([train_df, test_df])

combin.shape
combin['Name'] = [i.split(',')[1].split('.')[0].strip() for i in combin['Name']]
combin['Name'] = combin['Name'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

combin['Name'] = combin['Name'].replace('Mlle', 'Miss')

combin['Name'] = combin['Name'].replace('Ms', 'Miss')

combin['Name'] = combin['Name'].replace('Mme', 'Mrs')
combin.isnull().sum()
combin['Has_Cabin'] = combin["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
combin['FamilySize'] = combin['SibSp'] + combin['Parch'] + 1



combin['IsAlone'] = 0

combin.loc[combin['FamilySize'] == 1, 'IsAlone'] = 1
age_avg = combin['Age'].mean()

age_std = combin['Age'].std()

age_null_count = combin['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

combin['Age'][np.isnan(combin['Age'])] = age_null_random_list

combin['Age'] = combin['Age'].astype(int)
combin['Fare'].fillna(combin['Fare'].median(), inplace=True)
combin.loc[ combin['Fare'] <= 7.91, 'Fare']                              = 0

combin.loc[(combin['Fare'] > 7.91) & (combin['Fare'] <= 14.454), 'Fare'] = 1

combin.loc[(combin['Fare'] > 14.454) & (combin['Fare'] <= 31), 'Fare']   = 2

combin.loc[ combin['Fare'] > 31, 'Fare']                                 = 3

combin['Fare'] = combin['Fare'].astype(int)
combin.loc[ combin['Age'] <= 16, 'Age']                         = 0

combin.loc[(combin['Age'] > 16) & (combin['Age'] <= 32), 'Age'] = 1

combin.loc[(combin['Age'] > 32) & (combin['Age'] <= 48), 'Age'] = 2

combin.loc[(combin['Age'] > 48) & (combin['Age'] <= 64), 'Age'] = 3

combin.loc[ combin['Age'] > 64, 'Age']                          = 4 
combin['Embarked'] = combin['Embarked'].fillna('C')
drop_elements = ['PassengerId', 'Ticket', 'Cabin', 'SibSp']

combin = combin.drop(drop_elements, axis = 1)
combin.isnull().sum()
combin_flo = combin.select_dtypes(exclude='object')

combin_cat = combin.select_dtypes(include='object')
combin_cat = pd.get_dummies(combin_cat)
combin = pd.concat([combin_flo, combin_cat], axis=1)

combin.shape
combin.isnull().sum().sum()
X = combin.iloc[:train_df.shape[0], :]

test = combin.iloc[train_df.shape[0]:, :]
scaler = StandardScaler()

X_norm = scaler.fit_transform(X)

test_norm = scaler.transform(test)
X_train, X_val, y_train, y_val = train_test_split(X, y)
y_train.value_counts()
y_val.value_counts()
# param_grid = {'max_depth': [80, 90, 100, 110],

#               'max_features': [2, 3],

#               'min_samples_leaf': [3, 4, 5],

#               'min_samples_split': [8, 10, 12],

#               'n_estimators': [100, 200, 300, 1000]}



# rf = RandomForestClassifier(random_state=21)

# rf_cv = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1, verbose = 0)

# rf_cv.fit(X_train, y_train)

# rf_cv.best_params_
from sklearn.model_selection import RandomizedSearchCV, KFold



kf = KFold(n_splits = 5, random_state = 1)



rfc_parameters = {'max_depth' : [2, 5, 8, 10, 20, 50], 'n_estimators' : [10, 50, 100, 200, 500, 1000, 2000], 'min_samples_split' : [2, 3, 5, 9, 20]}

rfc = RandomForestClassifier(random_state = 1, n_jobs = -1)

clf_rfc = RandomizedSearchCV(rfc, rfc_parameters, n_jobs = -1, cv = kf, scoring = 'roc_auc')



clf_rfc.fit(X_train, y_train)

print(clf_rfc.best_score_)

print(clf_rfc.score(X_val, y_val))

print(clf_rfc.best_params_)
lr_paramaters = {'C' : [0.05, 0.1, 0.2], 'random_state' : [1]}

lr = LogisticRegression()



clf_lr = GridSearchCV(lr, lr_paramaters, n_jobs = -1, cv = kf, scoring = 'roc_auc')



clf_lr.fit(X_train, y_train)

print(clf_lr.best_score_)

print(clf_lr.score(X_val, y_val))

print(clf_lr.best_params_)
svc_paramaters = {'C' : [5.5, 6, 6.5], 'kernel' : ['linear', 'rbf'], 'gamma' : ['auto', 'scale'], 'random_state' : [1]}

svc = SVC(probability=True)



clf_svc = GridSearchCV(svc, svc_paramaters, n_jobs = -1, cv = kf, scoring = 'roc_auc')



clf_svc.fit(X_train, y_train)

print(clf_svc.best_score_)

print(clf_svc.score(X_val, y_val))

print(clf_svc.best_params_)
gbdt_parameters = {'subsample' : [1], 'min_samples_leaf' : [3], 'learning_rate' : [0.1], 'n_estimators' : [50], 'min_samples_split' : [2], 'max_depth' : [3], 'random_state' : [1]}

gbdt = GradientBoostingClassifier()



clf_gbdt = GridSearchCV(gbdt, gbdt_parameters, n_jobs = -1, cv = kf, scoring = 'roc_auc')



clf_gbdt.fit(X_train, y_train)

print(clf_gbdt.best_score_)

print(clf_gbdt.score(X_val, y_val))

print(clf_gbdt.best_params_)
xgb_paramaters = {'subsample' : [0.7], 'min_child_weight' : [1], 'max_depth' : [3], 'learning_rate' : [0.1], 'n_estimators' : [100], 'n_jobs' : [-1], 'random_state' : [1]}

xgb = xgb.XGBClassifier()



clf_xgb = GridSearchCV(xgb, xgb_paramaters, n_jobs = -1, cv = kf, scoring = 'roc_auc')



clf_xgb.fit(X_train, y_train)

print(clf_xgb.best_score_)

print(clf_xgb.score(X_val, y_val))

print(clf_xgb.best_params_)
pred = clf_xgb.predict(test)

sub = pd.read_csv('../input/gender_submission.csv')

sub['Survived'] = pred.astype(int)

sub.to_csv('sub_no_ens.csv', index=False)
classifiers = [

#     ('KNN', KNeighborsClassifier(3)),

    ('SVC', clf_svc),

#     ('DTC', DecisionTreeClassifier()),

    ('RFC', clf_rfc),

#     ('ABC', AdaBoostClassifier()),

    ('GBC', clf_gbdt),

#     ('GNB', GaussianNB()),

#     ('LDA', LinearDiscriminantAnalysis()),

#     ('QDA', QuadraticDiscriminantAnalysis()),

    ('LR', clf_lr),

    ('xgb', clf_xgb),

    ('lgbm', lgbm.LGBMClassifier())]
voting_clf = VotingClassifier(estimators=classifiers, voting='soft')

voting_clf.fit(X_train, y_train)

score = np.mean(cross_val_score(voting_clf,  X_val, y_val, scoring='accuracy'))

score
voting_clf.score(X_val, y_val)
pred = voting_clf.predict(test)
sub = pd.read_csv('../input/gender_submission.csv')

sub['Survived'] = pred.astype(int)

sub.head()
sub.to_csv('sub.csv', index=False)
pred = np.zeros(len(test))



for _, model in classifiers:

    print(model.__class__.__name__)

    model.fit(X, y)

    p = model.predict(test)

    pred = pred + p

pred = pred / len(classifiers)
for idx, i in enumerate(pred):

    if i >= 0.7:

        pred[idx] = 1

    else:

        pred[idx] = 0
sub = pd.read_csv('../input/gender_submission.csv')

sub['Survived'] = pred.astype(int)

sub.head()
sub.to_csv('sub2.csv', index=False)