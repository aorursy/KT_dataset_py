# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# essential libraries

import numpy as np 

import pandas as pd

# for data visulization

import matplotlib.pyplot as plt

import seaborn as sns





#for data processing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct

from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split



# for modeling estimators

from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier as gbm

from xgboost.sklearn import XGBClassifier

import lightgbm as lgb



# for measuring performance

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import f1_score

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics

from xgboost import plot_importance

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix



#for tuning parameters

from bayes_opt import BayesianOptimization

from skopt import BayesSearchCV

from eli5.sklearn import PermutationImportance



# Misc.

import os

import time

import gc

import random

from scipy.stats import uniform

import warnings

warnings.filterwarnings('ignore', category = RuntimeWarning)

pd.options.display.max_columns = 150



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', index_col='Id')

test = pd.read_csv('../input/test.csv', index_col='Id')
train.head(3)
test.head(3)
train_null = train.isnull().sum()

train_null_non_zero = train_null[train_null>0] / train.shape[0]
train_null_non_zero
sns.barplot(x=train_null_non_zero, y=train_null_non_zero.index)

_ = plt.title('Fraction of NaN values, %')
train.select_dtypes('object').head()
yes_no_map = {'no':0,'yes':1}

train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)

train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)

train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)   
yes_no_map = {'no':0,'yes':1}

test['dependency'] = test['dependency'].replace(yes_no_map).astype(np.float32)

test['edjefe'] = test['edjefe'].replace(yes_no_map).astype(np.float32)

test['edjefa'] = test['edjefa'].replace(yes_no_map).astype(np.float32)
# Number of missing in each column

missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(train)



missing.sort_values('percent', ascending = False).head(10)
train['v18q1'] = train['v18q1'].fillna(0)

test['v18q1'] = test['v18q1'].fillna(0)
train['v2a1'] = train['v2a1'].fillna(0)

test['v2a1'] = test['v2a1'].fillna(0)
train['rez_esc'] = train['rez_esc'].fillna(0)

test['rez_esc'] = test['rez_esc'].fillna(0)

train['SQBmeaned'] = train['SQBmeaned'].fillna(0)

test['SQBmeaned'] = test['SQBmeaned'].fillna(0)

train['meaneduc'] = train['meaneduc'].fillna(0)

test['meaneduc'] = test['meaneduc'].fillna(0)
#Checking for missing values again to confirm that no missing values present

# Number of missing in each column

missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(train)



missing.sort_values('percent', ascending = False).head(10)
#Checking for missing values again to confirm that no missing values present

# Number of missing in each column

missing = pd.DataFrame(test.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(train)



missing.sort_values('percent', ascending = False).head(10)
train.drop(['idhogar'], inplace = True, axis =1)



test.drop(['idhogar'], inplace = True, axis =1)
train.shape
test.shape
y = train.iloc[:,140]

y.unique()
X = train.iloc[:,1:141]

X.shape
my_imputer = SimpleImputer()

X = my_imputer.fit_transform(X)

scale = ss()

X = scale.fit_transform(X)
#subjecting the same to test data

my_imputer = SimpleImputer()

test = my_imputer.fit_transform(test)

scale = ss()

test = scale.fit_transform(test)
X.shape, y.shape,test.shape
X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y,

                                                    test_size = 0.2)
modelrf = rf()
start = time.time()

modelrf = modelrf.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modelrf.predict(X_test)
(classes == y_test).sum()/y_test.size 
f1 = f1_score(y_test, classes, average='macro')

f1
bayes_cv_tuner = BayesSearchCV(

    #  Place your estimator here with those parameter values

    #      that you DO NOT WANT TO TUNE

    rf(

       n_jobs = 2         # No need to tune this parameter value

      ),



    # 2.12 Specify estimator parameters that you would like to change/tune

    {

        'n_estimators': (100, 500),           # Specify integer-values parameters like this

        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here

        'max_depth': (4, 100),                # integer valued parameter

        'max_features' : (10,64),             # integer-valued parameter

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter

    },



    # 2.13

    n_iter=32,            # How many points to sample

    cv = 3                # Number of cross-validation folds

)
# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
modelrfTuned=rf(criterion="entropy",

               max_depth=77,

               max_features=64,

               min_weight_fraction_leaf=0.0,

               n_estimators=500)
start = time.time()

modelrfTuned = modelrfTuned.fit(X_train, y_train)

end = time.time()

(end-start)/60
yrf=modelrfTuned.predict(X_test)

yrf
yrftest=modelrfTuned.predict(test)

yrftest
#  Get what average accuracy was acheived during cross-validation

bayes_cv_tuner.best_score_
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
modeletf = ExtraTreesClassifier()
start = time.time()

modeletf = modeletf.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modeletf.predict(X_test)



classes
(classes == y_test).sum()/y_test.size
f1 = f1_score(y_test, classes, average='macro')

f1
bayes_cv_tuner = BayesSearchCV(

    #  Place your estimator here with those parameter values

    #      that you DO NOT WANT TO TUNE

    ExtraTreesClassifier( ),



    # 2.12 Specify estimator parameters that you would like to change/tune

    {   'n_estimators': (100, 500),           # Specify integer-values parameters like this

        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here

        'max_depth': (4, 100),                # integer valued parameter

        'max_features' : (10,64),             # integer-valued parameter

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter

    },



    n_iter=32,            # How many points to sample

    cv = 2            # Number of cross-validation folds

)
# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
modeletfTuned=ExtraTreesClassifier(criterion="entropy",

               max_depth=100,

               max_features=64,

               min_weight_fraction_leaf=0.0,

               n_estimators=500)
start = time.time()

modeletfTuned = modeletfTuned.fit(X_train, y_train)

end = time.time()

(end-start)/60
yetf=modeletfTuned.predict(X_test)
yetftest=modeletfTuned.predict(test)
#  Get what average accuracy was acheived during cross-validation

bayes_cv_tuner.best_score_
#  Get what average accuracy was acheived during cross-validation

bayes_cv_tuner.best_score_
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
modelneigh = KNeighborsClassifier(n_neighbors=4)
start = time.time()

modelneigh = modelneigh.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modelneigh.predict(X_test)



classes
(classes == y_test).sum()/y_test.size 
f1 = f1_score(y_test, classes, average='macro')

f1
bayes_cv_tuner = BayesSearchCV(

    #  Place your estimator here with those parameter values

    #      that you DO NOT WANT TO TUNE

    KNeighborsClassifier(

       n_neighbors=4         # No need to tune this parameter value

      ),

    {"metric": ["euclidean", "cityblock"]},

    n_iter=32,            # How many points to sample

    cv = 2            # Number of cross-validation folds

   )
bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
modelneighTuned = KNeighborsClassifier(n_neighbors=4,

               metric="cityblock")
start = time.time()

modelneighTuned = modelneighTuned.fit(X_train, y_train)

end = time.time()

(end-start)/60
yneigh=modelneighTuned.predict(X_test)
yneightest=modelneighTuned.predict(test)
#  Get what average accuracy was acheived during cross-validation

bayes_cv_tuner.best_score_
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
modelgbm=gbm()
start = time.time()

modelgbm = modelgbm.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modelgbm.predict(X_test)



classes
(classes == y_test).sum()/y_test.size 
f1 = f1_score(y_test, classes, average='macro')

f1
bayes_cv_tuner = BayesSearchCV(

    #  Place your estimator here with those parameter values

    #      that you DO NOT WANT TO TUNE

    gbm(

               # No need to tune this parameter value

      ),



    # 2.12 Specify estimator parameters that you would like to change/tune

    {

        'n_estimators': (100, 500),           # Specify integer-values parameters like this

        

        'max_depth': (4, 100),                # integer valued parameter

        'max_features' : (10,64),             # integer-valued parameter

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter

    },



    # 2.13

    n_iter=32,            # How many points to sample

    cv = 2                # Number of cross-validation folds

)
# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
modelgbmTuned=gbm(

               max_depth=84,

               max_features=11,

               min_weight_fraction_leaf=0.04840,

               n_estimators=489)
start = time.time()

modelgbmTuned = modelgbmTuned.fit(X_train, y_train)

end = time.time()

(end-start)/60
ygbm=modelgbmTuned.predict(X_test)
ygbmtest=modelgbmTuned.predict(test)
#  Get what average accuracy was acheived during cross-validation

bayes_cv_tuner.best_score_
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
modelxgb=XGBClassifier()
start = time.time()

modelxgb = modelxgb.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modelxgb.predict(X_test)



classes
(classes == y_test).sum()/y_test.size 
f1 = f1_score(y_test, classes, average='macro')

f1
bayes_cv_tuner = BayesSearchCV(

    #  Place your estimator here with those parameter values

    #      that you DO NOT WANT TO TUNE

    XGBClassifier(

       n_jobs = 2         # No need to tune this parameter value

      ),



    # 2.12 Specify estimator parameters that you would like to change/tune

    {

        'n_estimators': (100, 500),           # Specify integer-values parameters like this

        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here

        'max_depth': (4, 100),                # integer valued parameter

        'max_features' : (10,64),             # integer-valued parameter

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter

    },



    # 2.13

    n_iter=32,            # How many points to sample

    cv = 3                # Number of cross-validation folds

)
# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
modelxgbTuned=XGBClassifier(criterion="gini",

               max_depth=4,

               max_features=15,

               min_weight_fraction_leaf=0.05997,

               n_estimators=499)
start = time.time()

modelxgbTuned = modelxgbTuned.fit(X_train, y_train)

end = time.time()

(end-start)/60
yxgb=modelxgbTuned.predict(X_test)
yxgbtest=modelxgbTuned.predict(test)
#  Get what average accuracy was acheived during cross-validation

bayes_cv_tuner.best_score_
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',

                             random_state=None, silent=True, metric='None', 

                             n_jobs=4, n_estimators=5000, class_weight='balanced',

                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)
start = time.time()

modellgb = modellgb.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modellgb.predict(X_test)



classes
(classes == y_test).sum()/y_test.size 
f1 = f1_score(y_test, classes, average='macro')

f1
bayes_cv_tuner = BayesSearchCV(

    #  Place your estimator here with those parameter values

    #      that you DO NOT WANT TO TUNE

    lgb.LGBMClassifier(

       n_jobs = 2         # No need to tune this parameter value

      ),



    # 2.12 Specify estimator parameters that you would like to change/tune

    {

        'n_estimators': (100, 500),           # Specify integer-values parameters like this

        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here

        'max_depth': (4, 100),                # integer valued parameter

        'max_features' : (10,64),             # integer-valued parameter

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter

    },



    # 2.13

    n_iter=32,            # How many points to sample

    cv = 3                # Number of cross-validation folds

)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
modellgbTuned = lgb.LGBMClassifier(criterion="gini",

               max_depth=5,

               max_features=53,

               min_weight_fraction_leaf=0.01674,

               n_estimators=499)
start = time.time()

modellgbTuned = modellgbTuned.fit(X_train, y_train)

end = time.time()

(end-start)/60
ylgb=modellgbTuned.predict(X_test)
ylgbtest=modellgbTuned.predict(test)
#  Get what average accuracy was acheived during cross-validation

bayes_cv_tuner.best_score_
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
NewTrain = pd.DataFrame()

NewTrain['yrf'] = yrf.tolist()

NewTrain['yetf'] = yetf.tolist()

NewTrain['yneigh'] = yneigh.tolist()

NewTrain['ygbm'] = ygbm.tolist()

NewTrain['yxgb'] = yxgb.tolist()

NewTrain['ylgb'] = ylgb.tolist()



NewTrain.head(5), NewTrain.shape
NewTest = pd.DataFrame()

NewTest['yrf'] = yrftest.tolist()

NewTest['yetf'] = yetftest.tolist()

NewTest['yneigh'] = yneightest.tolist()

NewTest['ygbm'] = ygbmtest.tolist()

NewTest['yxgb'] = yxgbtest.tolist()

NewTest['ylgb'] = ylgbtest.tolist()

NewTest.head(5), NewTest.shape
NewModel=rf(criterion="entropy",

               max_depth=77,

               max_features=6,

               min_weight_fraction_leaf=0.0,

               n_estimators=500)

start = time.time()

NewModel = NewModel.fit(NewTrain, y_test)

end = time.time()

(end-start)/60
ypredict=NewModel.predict(NewTest)
ylgbtest
submit=pd.DataFrame({'Id': ids, 'Target': ylgbtest})

submit.head(5)
submit.to_csv('submit.csv', index=False)