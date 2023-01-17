# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()
test.head()
train.info()
train.describe().T
train['Pclass'].nunique()
train['Pclass'].value_counts().plot.barh()
fig, ax = plt.subplots(ncols = 3, figsize = (10, 3))

for x in train['Pclass'].unique():

    train[train['Pclass'] == x]['Survived'].value_counts().plot.bar(ax = ax[x-1], title = 'Pclass '+str(x))
comb = pd.concat([train, test], axis = 0).reset_index()
comb.drop('index', axis = 1, inplace = True)
comb['FamSize'] = comb.SibSp + comb.Parch + 1
comb['IsAlone'] =0

comb.loc[comb.FamSize == 1, 'IsAlone'] = 1
(comb['FamSize'] == 1).value_counts()
comb['IsAlone'].value_counts()
comb['Embarked'].fillna('S', inplace = True)
comb['Fare'].fillna(train['Fare'].median(), inplace = True)
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
train['CategoricalFare']
age_avg = comb['Age'].mean()

age_std = comb['Age'].std()

age_null_count = comb['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
comb.at[comb[comb['Age'].isnull()]['Age'].index, 'Age'] = age_null_random_list
comb[comb['Age'].isnull()]
train['CategorcalAge'] = pd.qcut(comb['Age'], 5)
train.head()
def get_title(comb, Name):

    return comb[Name].apply(lambda x: x.split(',')[1].split(' ')[1][:-1])
comb['Title'] = get_title(comb, 'Name')
comb['Title'].replace(['Lady', 'th','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace=True)
comb['Title'].replace(['Mlle', 'Ms'], 'Rare', inplace = True)
comb['Title'].replace(['Mme'], 'Mrs', inplace = True)
comb['Sex'] = comb['Sex'].map({'female':0, 'male':1}).astype(int)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

comb['Title'] = comb['Title'].map(title_mapping).astype(int)
comb['Embarked'] = comb['Embarked'].map({'S': 0, 'C': 1, 'Q' : 2}).astype(int)
comb.loc[comb['Fare'] <= 7.91, 'Fare']= 0

comb.loc[(comb['Fare'] > 7.91) & (comb['Fare'] <= 14.454), 'Fare'] = 1

comb.loc[(comb['Fare'] > 14.454) & (comb['Fare'] <= 31), 'Fare']   = 2

comb.loc[comb['Fare'] > 31, 'Fare']= 3

comb['Fare'] = comb['Fare'].astype(int)



comb.loc[comb['Age'] <= 16, 'Age']= 0

comb.loc[(comb['Age'] > 16) & (comb['Age'] <= 32), 'Age'] = 1

comb.loc[(comb['Age'] > 32) & (comb['Age'] <= 48), 'Age'] = 2

comb.loc[(comb['Age'] > 48) & (comb['Age'] <= 64), 'Age'] = 3

comb.loc[comb['Age'] > 64, 'Age']= 4
ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0

NFOLDS = 5

# kf = KFold(n_folds=NFOLDS, random_state = SEED)
kf = KFold(n_splits= NFOLDS, random_state=SEED)
class SklearnHelper(object):

    def __init__(self, clf, seed = 0, params = None):

        params['random_state'] = seed

        self.clf = clf(**params)

    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)

        

    def predict(self, x):

        return self.clf.predict(x)

    

def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain, ))

    oof_test = np.zeros((ntest, ))

    oof_test_skf = np.empty((NFOLDS, ntest))

        

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train.iloc[train_index]

        y_tr = y_train.iloc[train_index]

        x_te = x_train.iloc[test_index]

        clf.train(x_tr, y_tr)

            

        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)

        

        oof_test[:] = oof_test_skf.mean(axis = 0)

        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
rf_params = {

    'n_jobs': -1,

    'n_estimators': 575,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 3 

}

et_params = {

    'n_jobs': -1,

    'n_estimators':575,

    #'max_features': 0.5,

    'max_depth': 5,

    'min_samples_leaf': 3,

    'verbose': 3

}

ada_params = {

    'n_estimators': 575,

    'learning_rate' : 0.95

}



gb_params = {

    'n_estimators': 575,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 3,

    'verbose': 3

}

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }



rf = SklearnHelper(clf = RandomForestClassifier, seed=SEED, params = rf_params)

et = SklearnHelper(clf = ExtraTreesClassifier, seed = SEED, params = et_params)

gb = SklearnHelper(clf = GradientBoostingClassifier, seed = SEED, params = gb_params)

ada = SklearnHelper(clf = AdaBoostClassifier, seed = SEED, params = ada_params)

svc = SklearnHelper(clf = SVC, seed = SEED, params = svc_params)
comb.drop(['Cabin', 'Name', 'Ticket'], axis = 1, inplace = True)

x_train = comb[comb['Survived'].isna() == False].drop('Survived', axis = 1)

y_train = comb[comb['Survived'].isna() == False]['Survived']

x_test = comb[comb['Survived'].isna()].drop('Survived', axis = 1)
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test)

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test)
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))
gbm = xgb.XGBClassifier(learning_rate = 0.95, 

                       n_estimators = 5000, 

                       max_depth = 4, 

                       min_child_weight = 2,

                       gamma = 1,

                       subsample = 0.8,

                       colsample_bytree = 0.8,

                       scale_pos_weight = 1).fit(x_train, y_train)
gbm_preds = gbm.predict(x_test)
stacking_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived': gbm_preds})
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy_score(sub['Survived'], gbm_preds)
stacking_submission['Survived'] = stacking_submission['Survived'].apply(int)
stacking_submission.to_csv('StackingSubmission.csv', index = False)