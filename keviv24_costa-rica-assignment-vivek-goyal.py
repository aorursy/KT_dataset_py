# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# essential libraries



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

pd.options.display.max_columns = 150



# Read in data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()

ids=test['Id']

train.shape, test.shape
train.info()   
test.head(10)
sns.countplot("Target", data=train)
sns.countplot(x="r4t3",hue="Target",data=train)
sns.countplot(x="v18q",hue="Target",data=train)
sns.countplot(x="v18q1",hue="Target",data=train)

sns.countplot(x="tamhog",hue="Target",data=train)

sns.countplot(x="hhsize",hue="Target",data=train)
sns.countplot(x="abastaguano",hue="Target",data=train)
sns.countplot(x="noelec",hue="Target",data=train)
train.select_dtypes('object').head()
yes_no_map = {'no':0,'yes':1}

train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)

train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)

train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)

    
yes_no_map = {'no':0,'yes':1}

test['dependency'] = test['dependency'].replace(yes_no_map).astype(np.float32)

test['edjefe'] = test['edjefe'].replace(yes_no_map).astype(np.float32)

test['edjefa'] = test['edjefa'].replace(yes_no_map).astype(np.float32)
train[["dependency","edjefe","edjefa"]].describe()
 # Number of missing in each column

missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(train)



missing.sort_values('percent', ascending = False).head(10)
train['v18q1']     = train['v18q1'].fillna(0)

test['v18q1']      = test['v18q1'].fillna(0)

train['v2a1']      = train['v2a1'].fillna(0)

test['v2a1']       = test['v2a1'].fillna(0)



train['rez_esc']   = train['rez_esc'].fillna(0)

test['rez_esc']    = test['rez_esc'].fillna(0)

train['SQBmeaned'] = train['SQBmeaned'].fillna(0)

test['SQBmeaned']  = test['SQBmeaned'].fillna(0)

train['meaneduc']  = train['meaneduc'].fillna(0)

test['meaneduc']   = test['meaneduc'].fillna(0)
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
train.drop(['Id','idhogar'], inplace = True, axis =1)



test.drop(['Id','idhogar'], inplace = True, axis =1)
train.shape
test.shape
y = train.iloc[:,140]

X = train.iloc[:,1:141]

X.shape, y.shape
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
from sklearn.ensemble import RandomForestClassifier as rf

modelrf = rf()



start = time.time()

modelrf = modelrf.fit(X_train, y_train)

end = time.time()

(end-start)/60
out_class = modelrf.predict(X_test)

out_class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, out_class)

cm





from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, out_class)

accuracy
bayes_cv_tuner = BayesSearchCV(

    rf(

       n_jobs = 2         

      ),

    

    {

        'n_estimators': (100, 500),         

        'criterion': ['gini', 'entropy'],     

        'max_depth': (4, 100),                

        'max_features' : (10,64),            

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   

    },



    n_iter=32,            # How many points to sample

    cv = 3                # Number of cross-validation folds

)
# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
#using the best params 

modelrfTuned=rf(criterion="entropy",

               max_depth=100,

               max_features=64,

               min_weight_fraction_leaf=0.0,

               n_estimators=173)
#fit the data in the model

modelrfTuned = modelrfTuned.fit(X_train, y_train)



#Predict

y_rf=modelrfTuned.predict(X_test)

y_rf
#predict for the test data

y_rf_test=modelrfTuned.predict(test)

y_rf_test
#  Get what average accuracy was acheived during cross-validation

accuracy = bayes_cv_tuner.best_score_

accuracy
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)

#  what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
modelknn = KNeighborsClassifier(n_neighbors=4)
start = time.time()

modelknn = modelknn.fit(X_train, y_train)

end = time.time()

(end-start)/60
out_class = modelrf.predict(X_test)

out_class
(out_class == y_test).sum()/y_test.size 
#from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, out_class)

cm
#from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, out_class)

accuracy
bayes_cv_tuner = BayesSearchCV(

    

    KNeighborsClassifier(

       n_neighbors=4        

      ),

    {"metric": ["euclidean", "cityblock"]},

    n_iter=32,            # How many points to sample

    cv = 2            # Number of cross-validation folds

   )
# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
modelneighTuned = KNeighborsClassifier(n_neighbors=4,metric="cityblock")
#Fit to the model

modelneighTuned = modelneighTuned.fit(X_train, y_train)



# Predict 

y_neigh=modelneighTuned.predict(X_test)

# predict for the test data



y_neigh_test=modelneighTuned.predict(test)
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
modelgbm=gbm()

start = time.time()

modelgbm = modelgbm.fit(X_train, y_train)

end = time.time()

(end-start)/60
out_class = modelgbm.predict(X_test)



out_class
(out_class == y_test).sum()/y_test.size 
#from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, out_class)

cm
#from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, out_class)

accuracy
bayes_cv_tuner = BayesSearchCV(



    gbm(

      ),

    {

        'n_estimators': (100, 500),           

        

        'max_depth': (4, 100),               

        'max_features' : (10,64),            

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   

    },

    n_iter=32,            # How many points to sample

    cv = 2                # Number of cross-validation folds

)
# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
modelgbmTuned=gbm(

               max_depth=100,

               max_features=64,

               min_weight_fraction_leaf=0.0,

               n_estimators=500)
#Re fitting the model with optimized parameters

modelgbmTuned = modelgbmTuned.fit(X_train, y_train)



# Predict

y_gbm=modelgbmTuned.predict(X_test)

# Predicting for the test data

y_gbm_test=modelgbmTuned.predict(test)
#  Get what average accuracy was acheived during cross-validation

accuracy = bayes_cv_tuner.best_score_

accuracy
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
#### accracy is high 100% before and after optimization
model_etf = ExtraTreesClassifier()
start = time.time()

model_etf = model_etf.fit(X_train, y_train)

end = time.time()

(end-start)/60
out_class = model_etf.predict(X_test)



out_class
(out_class == y_test).sum()/y_test.size
#from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, out_class)

cm
#from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, out_class)

accuracy

bayes_cv_tuner = BayesSearchCV(



    ExtraTreesClassifier( ),   

    {   'n_estimators': (100, 500),           

        'criterion': ['gini', 'entropy'],     

        'max_depth': (4, 100),                

        'max_features' : (10,64),             

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   

    },



    n_iter=32,            # How many points to sample

    cv = 2            # Number of cross-validation folds

)

# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
#using the best parameters

modeletfTuned=ExtraTreesClassifier(criterion="entropy",

               max_depth=4,

               max_features=64,

               min_weight_fraction_leaf=0.0,

               n_estimators=100)
#Fit the model

modeletfTuned = modeletfTuned.fit(X_train, y_train)



# predict

y_etf=modeletfTuned.predict(X_test)
#predict with the test data

y_etftest=modeletfTuned.predict(test)
#  Get what average accuracy was acheived during cross-validation

accuracy = bayes_cv_tuner.best_score_

accuracy
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
#### accracy increased from 98.27% to 100%
model_xgb=XGBClassifier()
start = time.time()

model_xgb = model_xgb.fit(X_train, y_train)

end = time.time()

(end-start)/60
out_class = model_xgb.predict(X_test)



out_class
(out_class == y_test).sum()/y_test.size 
#from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, out_class)

cm


#from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, out_class)

accuracy
bayes_cv_tuner = BayesSearchCV(



    XGBClassifier(

       n_jobs = 2         

      ),

    {

        'n_estimators': (100, 500),           

        'criterion': ['gini', 'entropy'],     

        'max_depth': (4, 100),                

        'max_features' : (10,64),             

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   

    },



    n_iter=32,            # How many points to sample

    cv = 3                # Number of cross-validation folds

)
# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
modelxgbTuned=XGBClassifier(criterion="entropy",

               max_depth=51,

               max_features=16,

               min_weight_fraction_leaf=0.2150244429465713,

               n_estimators=355)
# refit the model with optimized data

modelxgbTuned = modelxgbTuned.fit(X_train, y_train)



#predict

y_xgb=modelxgbTuned.predict(X_test)
y_xgbtest=modelxgbTuned.predict(test)
#  Get what average accuracy was acheived during cross-validation

accuracy = bayes_cv_tuner.best_score_

accuracy
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
model_lgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',

                             random_state=None, silent=True, metric='None', 

                             n_jobs=4, n_estimators=5000, class_weight='balanced',

                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)

start = time.time()

model_lgb = model_lgb.fit(X_train, y_train)

end = time.time()

(end-start)/60
out_class = model_lgb.predict(X_test)



out_class
(out_class == y_test).sum()/y_test.size 
#from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, out_class)

cm

#from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, out_class)

accuracy
bayes_cv_tuner = BayesSearchCV(



    lgb.LGBMClassifier(

       n_jobs = 2         

      ),



    {

        'n_estimators': (100, 500),           

        'criterion': ['gini', 'entropy'],     

        'max_depth': (4, 100),                

        'max_features' : (10,64),             

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   

    },

    n_iter=32,            # How many points to sample

    cv = 3                # Number of cross-validation folds

)
# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
#  Get list of best-parameters

bayes_cv_tuner.best_params_
modellgbTuned = lgb.LGBMClassifier(criterion="gini",

               max_depth=23,

               max_features=48,

               min_weight_fraction_leaf=0.4939249242565817,

               n_estimators=437)
# Re fit the model

modellgbTuned = modellgbTuned.fit(X_train, y_train)



# Predict

y_lgb=modellgbTuned.predict(X_test)
# predict for the test data

y_lgb_test=modellgbTuned.predict(test)

#  Get what average accuracy was acheived during cross-validation

accuracy = bayes_cv_tuner.best_score_

accuracy
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
## looking at the accuracy before and after tuning all the models I am choosing Random Forest with following params

## as it was giving the optimum accuracy

modelrfTuned=rf(criterion="entropy",

               max_depth=100,

               max_features=64,

               min_weight_fraction_leaf=0.0,

               n_estimators=173)



#fit the data in the model

modelrfTuned = modelrfTuned.fit(X_train, y_train)



#predict for the test data

y_rf_test=modelrfTuned.predict(test)

y_rf_test
test_final = pd.read_csv('../input/test.csv')



test_final['Pred_Out'] = y_rf_test.tolist()



test_final.head(10)
