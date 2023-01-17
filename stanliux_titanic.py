# Load in our libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

import copy

import seaborn as sns

import random 

color = sns.color_palette()

sns.set_style('darkgrid')

%matplotlib inline



from scipy import stats

from scipy.stats import norm, skew



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import average_precision_score

import xgboost as xgb
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(5)
test.head(5)
ntrain = len(train)

ntest = len(test)
# Obtain our labels

train_label = train['Survived']

# Drop the useless columns and concatenate the tables

PassengerId = test['PassengerId']

train.drop(columns = ['PassengerId'], axis = 1, inplace = True)

test.drop(columns = ['PassengerId'], axis = 1, inplace = True)

all_data = pd.concat([train, test], axis = 0, ignore_index = True)
all_data.drop(columns = ['Survived'], inplace = True)
all_data.dtypes
all_data.isnull().sum()
all_data.Embarked.fillna(value = all_data.Embarked.mode().values[0], inplace = True)
all_data.Fare.fillna(value = all_data.Fare.mean(), inplace = True)
age_avg = all_data['Age'].mean()

age_std = all_data['Age'].std()



age_null_count = all_data['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

all_data['Age'][np.isnan(all_data['Age'])] = age_null_random_list



all_data['Age'] = all_data['Age'].astype(int)
all_data['Cabin'] = all_data['Cabin'].astype(str, copy = True)
all_data['Cabin'] = all_data['Cabin'].apply(lambda x: re.findall(r'([A-Z])\d*',x)[0] if x != 'nan' else 0)
all_data['Cabin'] = all_data['Cabin'].apply(lambda x: 'Z' if x == 0 else x)
all_data.Age.hist(bins = 60)
all_data.Fare.hist(bins = 50)
all_data.Fare.max()
all_data['Fare'] = all_data['Fare'].apply(lambda x: 300 if x >= 500 else x)
all_data['CategoricalAge'] = pd.cut(all_data['Age'], 5)
# Remove all NULLS in the Fare column and create a new feature CategoricalFare

all_data['CategoricalFare'] = pd.qcut(all_data['Fare'], 5)
lambda_1 = 0.70

all_data['Age_adj'] = (np.power(all_data['Age'], lambda_1) - 1)/lambda_1



#Plot the histogram for the adjuested data

sns.distplot(all_data['Age_adj'] , fit=norm);



#Plot the qq plot

fig = plt.figure()

res = stats.probplot(np.array(all_data['Age_adj']), plot=plt)

plt.show()
all_data['Fare_adj'] = np.log(all_data['Fare'] + 1)



#Plot the histogram for the adjuested data

sns.distplot(all_data['Fare_adj'] , fit=norm);



#Plot the qq plot

fig = plt.figure()

res = stats.probplot(np.array(all_data['Fare_adj']), plot=plt)

plt.show()
# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



# Create a new feature Title, containing the titles of passenger names

all_data['Title'] = all_data['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')

all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')

all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
def letter_match(string):

    result1 = re.search(r'^([A-Z]*\S*|[A-Z0-9].*/[A-Z0-9].*)\s',string)

    try:

        result1 = result1.group(0).strip()

        return result1

    except:

        return ''
def num_match(string):

    result2 = re.search(r'\w([0-9]+)$',string)

    try:

        result2 = result2.group(0).strip()

        return result2

    except:

        return ''
def num_len(string):

    if len(string) == 3:

        return 3

    elif len(string) == 4:

        return 4

    elif len(string) == 5:

        return 5

    elif len(string) == 6:

        return 6

    elif len(string) == 7:

        return 7

    else:

        return 0
Ticket= pd.DataFrame()

Ticket['Letter'] = all_data['Ticket'].apply(letter_match)

Ticket['Num'] = all_data['Ticket'].apply(num_match)
Ticket['num_len'] = Ticket['Num'].apply(num_len)
Ticket['num_len'].value_counts()
Ticket['Letter'].value_counts().head(10)
Ticket.drop(columns = ['Letter'], axis = 1, inplace = True)
encoder = LabelEncoder()

# Mapping Sex

all_data['Sex'] = encoder.fit_transform(all_data['Sex']).astype(str)

    

# Mapping titles

all_data['Title'] = encoder.fit_transform(all_data['Title']).astype(str)

    

# Mapping Embarked

all_data['Embarked'] = encoder.fit_transform(all_data['Embarked']).astype(str)

    

# Mapping Fare

all_data['CategoricalFare'] = encoder.fit_transform(all_data['CategoricalFare']).astype(str)

    

# Mapping Age

all_data['CategoricalAge'] = encoder.fit_transform(all_data['CategoricalAge']).astype(str)



# Mapping Cabin

all_data['Cabin'] = encoder.fit_transform(all_data['Cabin']).astype(str)



#Concatenate DataFrame Ticket

data_complete = pd.concat([all_data, Ticket], axis=1, copy = True)
data_complete.drop(columns = ['Age','Fare','Name','Ticket'], axis = 1, inplace = True)
data_complete['Num_1'] = data_complete['Num'].apply(lambda x: x[0] if len(x)>=1 else '0')

data_complete.drop(columns = ['Num'], axis = 1, inplace = True)
data_complete.head(5)
data_complete['Pclass'] = data_complete['Pclass'].astype(str)

data_complete['num_len'] = data_complete['num_len'].astype(str)
data = pd.get_dummies(data_complete)
data.head(5)
X_train_complete = data_complete.loc[:ntrain-1].values

X_test_complete = data_complete.loc[ntrain:].values
X_train = data.loc[:ntrain-1].values

y_train = train_label.values

X_test = data.loc[ntrain:].values
# Some useful parameters which will come in handy later on

r = random.seed(2)

#SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

#kf = KFold(n_splits= NFOLDS, random_state=SEED)

#scores = cross_val_score()

# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)

    

# Class to extend XGboost classifer
def k_fold_spliter(train_len, i_fold, n_splits=5 ):

    train_list = range(train_len)

    left = train_len % n_splits

    

    if left != 0:

        batch = int(np.floor(train_len / n_splits))

        batch_1 = int(np.floor(train_len / n_splits)) + 1

        if i_fold  <= left-1:

            val_set = train_list[batch_1 * (i_fold-1): i_fold * batch_1]

            train_set = list(set(train_list).difference(set(val_set)))

        elif i_fold == left:

            val_set = train_list[batch_1 * (i_fold-1): i_fold * batch]

            train_set = list(set(train_list).difference(set(val_set)))

        else:

            val_set = train_list[batch * (i_fold-1): i_fold * batch]

            train_set = list(set(train_list).difference(set(val_set)))

    else:

        batch = int(train_len/n_splits)

        val_set = train_list[(i_fold-1)*batch : i_fold * batch]

        train_set = list(set(train_list).difference(set(val_set)))

    

    return train_set, val_set
def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i in range(1,NFOLDS+1):

        train_index, test_index = k_fold_spliter(len(x_train), i, n_splits=NFOLDS)

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        

        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i-1, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Put in our parameters for said classifiers



# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 10,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 10,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'rbf',

    'C' : 0.5,

    'gamma': 0.3

    }



# Logistic Regression Classifier parameters

lr_params = {

    'penalty':'l2',

    'C': 0.5,

    'solver':'newton-cg',

    'max_iter':1000,

    'n_jobs' : -1

}
#Perhaps we can solily adjust the parameter for SVC and other few classifiers.
# Create 5 objects that represent our 4 models

SEED = 0



rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

lr = SklearnHelper(clf=LogisticRegression, seed=SEED, params = lr_params)
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, X_train, y_train, X_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf, X_train, y_train, X_test) # Random Forest



ada_oof_train, ada_oof_test = get_oof(ada, X_train, y_train, X_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb, X_train, y_train, X_test) # Gradient Boost



svc_oof_train, svc_oof_test = get_oof(svc, X_train, y_train, X_test) # Support Vector Classifier

lr_oof_train, lr_oof_test = get_oof(lr, X_train, y_train, X_test) # Logistic Regression Classifier



print("Training is complete")
ap_et = average_precision_score(y_train, et_oof_train)

ap_rf = average_precision_score(y_train, rf_oof_train)



ap_ada = average_precision_score(y_train, ada_oof_train)

ap_gb = average_precision_score(y_train, gb_oof_train)



ap_svc = average_precision_score(y_train, svc_oof_train)

ap_lr = average_precision_score(y_train, lr_oof_train)

print(ap_et,ap_rf,ap_ada,ap_gb,ap_svc, ap_lr)
x_train_2 = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train, lr_oof_train), axis=1)

x_test_2 = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test, lr_oof_test), axis=1)
gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

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



scores = cross_val_score(gbm, x_train_2, y_train, cv=5)

print(scores)
gbm.fit(x_train_2, y_train)

predictions = gbm.predict(x_test_2)
# Generate Submission File 

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)