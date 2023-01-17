# 1.1 Load pandas, numpy and matplotlib

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from xgboost import plot_importance



# Image manipulation

from skimage.io import imshow, imsave



# Image normalizing and compression

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import TruncatedSVD



# Feature Selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif  # Selection criteria

import eli5

from eli5.sklearn import PermutationImportance



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
pd.options.display.max_columns = 150



# Read in data



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.info()
test.info()
ids=test['Id']
train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 

                                                                             figsize = (8, 6),

                                                                            edgecolor = 'k', linewidth = 2);

plt.xlabel('Number of Unique Values'); plt.ylabel('Count');

plt.title('Count of Unique Values in Integer Columns');
test.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 

                                                                             figsize = (8, 6),

                                                                            edgecolor = 'k', linewidth = 2);

plt.xlabel('Number of Unique Values'); plt.ylabel('Count');

plt.title('Count of Unique Values in Integer Columns');
train.select_dtypes('object').head()
test.select_dtypes('object').head()
mapping = {"yes": 1, "no": 0}



# Apply same operation to both train and test

for df in [train, test]:

    # Fill in the values with the correct mapping

    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)

    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)

    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)



train[['dependency', 'edjefa', 'edjefe']].describe()
test[['dependency', 'edjefa', 'edjefe']].describe()
# Set a few plotting defaults

%matplotlib inline

plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 18

plt.rcParams['patch.edgecolor'] = 'k'



from collections import OrderedDict



# Color mapping

colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})

poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})



plt.figure(figsize = (16, 12))

plt.style.use('fivethirtyeight')



# Iterate through the float columns

for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):

    ax = plt.subplot(3, 1, i + 1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

        # Plot each poverty level as a separate line

        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')



plt.subplots_adjust(top = 2)
test.shape
train.shape
train.shape
# Number of missing in each column

missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(train)



missing.sort_values('percent', ascending = False).head(10)
# Number of missing in each column

missingte = pd.DataFrame(test.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missingte['percent'] = missingte['total'] / len(test)



missingte.sort_values('percent', ascending = False).head(10)
# Variables indicating home ownership

own_variables = [x for x in train if x.startswith('tipo')]





# Plot of the home ownership variables for home missing rent payments

train.loc[train['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),

                                                                        color = 'green',

                                                              edgecolor = 'k', linewidth = 2);

plt.xticks([0, 1, 2, 3, 4],

           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],

          rotation = 60)

plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);
# Variables indicating home ownership

own_variableste = [x for x in test if x.startswith('tipo')]





# Plot of the home ownership variables for home missing rent payments

test.loc[test['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),

                                                                        color = 'green',

                                                              edgecolor = 'k', linewidth = 2);

plt.xticks([0, 1, 2, 3, 4],

           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],

          rotation = 60)

plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);
# Fill in households that own the house with 0 rent payment

train.loc[(train['tipovivi1'] == 1), 'v2a1'] = 0



# Create missing rent payment column

train['v2a1-missing'] = train['v2a1'].isnull()



train['v2a1-missing'].value_counts()
# Fill in households that own the house with 0 rent payment

test.loc[(test['tipovivi1'] == 1), 'v2a1'] = 0



# Create missing rent payment column

test['v2a1-missing'] = test['v2a1'].isnull()



test['v2a1-missing'].value_counts()
train.loc[train['rez_esc'].notnull()]['age'].describe()
test.loc[test['rez_esc'].isnull()]['age'].describe()
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
train.shape
train.drop(['Id','idhogar','v2a1-missing'], inplace = True, axis =1)



test.drop(['Id','idhogar','v2a1-missing'], inplace = True, axis =1)
train.shape
test.shape
y = train.iloc[:,140]

y.unique()
X = train.iloc[:,1:141]

X.shape
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler as ss



sm_imputer = SimpleImputer()

X = sm_imputer.fit_transform(X)

scale = ss()

X = scale.fit_transform(X)

#pca = PCA(0.95)

#X = pca.fit_transform(X)
X.shape, y.shape,test.shape
X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y,

                                                    test_size = 0.2)
from sklearn.ensemble import ExtraTreesClassifier



modeletf = ExtraTreesClassifier()
start = time.time()

modeletf = modeletf.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modeletf.predict(X_test)



classes
(classes == y_test).sum()/y_test.size
from sklearn.metrics import f1_score

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

yetf
yetftest=modeletfTuned.predict(test)

yetftest
#  Get what average accuracy was acheived during cross-validation

bayes_cv_tuner.best_score_
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier as gbm



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

ygbm
ygbmtest=modelgbmTuned.predict(test)

ygbmtest
#  Get what average accuracy was acheived during cross-validation

bayes_cv_tuner.best_score_
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
from lightgbm import LGBMClassifier



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
# Start optimization

bayes_cv_tuner.fit(X_train, y_train)
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

ylgb
ylgbtest=modellgbTuned.predict(test)

ylgbtest
#  Get what average accuracy was acheived during cross-validation

bayes_cv_tuner.best_score_
#  What accuracy is available on test-data

bayes_cv_tuner.score(X_test, y_test)
#  And what all sets of parameters were tried?

bayes_cv_tuner.cv_results_['params']
modelrf = rf()
start = time.time()

modelrf = modelrf.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modelrf.predict(X_test)

classes
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
train1 = pd.DataFrame()



train1['yetf'] = yetf.tolist()

train1['ygbm'] = ygbm.tolist()

train1['ylgb'] = ylgb.tolist()

train1['yrf'] = yrf.tolist()



train1.head(5), train1.shape
test1 = pd.DataFrame()



test1['yetf'] = yetftest.tolist()

test1['ygbm'] = ygbmtest.tolist()

test1['ylgb'] = ylgbtest.tolist()

test1['yrf'] = yrftest.tolist()



test1.head(5), test1.shape
EnsembleModel=rf(criterion="entropy",

               max_depth=77,

               max_features=4,

               min_weight_fraction_leaf=0.0,

               n_estimators=500)
start = time.time()

EnsembleModel = EnsembleModel.fit(train1, y_test)

end = time.time()

(end-start)/60
ypredict=EnsembleModel.predict(test1)
ypredict
ygbmtest
submit=pd.DataFrame({'Id': ids, 'Target': ygbmtest})

submit.head(5)
submit.to_csv('submit.csv', index=False)