# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for data visulization

import matplotlib.pyplot as plt

import seaborn as sns



# for modeling estimators

from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier as gbm

from xgboost.sklearn import XGBClassifier

import lightgbm as lgb

#for data processing

from sklearn.model_selection import train_test_split



#for tuning parameters

from bayes_opt import BayesianOptimization

from skopt import BayesSearchCV

from eli5.sklearn import PermutationImportance





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.



# Misc.

import os

import time

import gc
# Read in data

train = pd.read_csv('../input/train.csv')

train.head()
train.shape
test = pd.read_csv('../input/test.csv')

test.head()
#Perform data visualization

train.plot(figsize = (12,10))
#Count plot

sns.countplot("Target", data=train)
#Count plot

sns.countplot(x="v2a1",hue="Target",data=train)
#Dividing the data into predictors & target

y = train.iloc[:,140]

y.unique()
X = train.iloc[:,1:141]

X.shape


#Label encoding 

train['edjefe'].value_counts()

map2 = {'yes':0,'no':1}

map2

train['edjefe'] = train['edjefe'].replace(map2).astype(np.float32)
test['dependency'] = test['dependency'].map({"yes" : 1, "no" : 0})

test['edjefa'] = test['edjefa'].map({"yes" : 1, "no" : 0})

test['edjefe'] = test['edjefe'].map({"yes" : 1, "no" : 0})
#CLEANING DATA

#     Transform train and test dataframes

#     replacing '0' with NaN

train.replace(0, np.nan)

test.replace(0,np.nan)

#fillna() to replace missing values with the mean value for each column,



train.fillna(train.mean(), inplace=True);

print(train.isnull().sum());



train.shape
test.fillna(test.mean(), inplace=True);

print(test.isnull().sum());



test.shape
# Summary of target feature

train['Target'].value_counts() 


X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y,

                                                    test_size = 0.2)
f, axes = plt.subplots(2, 2, figsize=(6, 6))

sns.boxplot(x="Target", y="hhsize", data=train, ax=axes[0, 0])# size of the house hold 

sns.boxplot(x="Target", y="tamviv", data=train, ax=axes[0, 1])# No of people in the household

sns.boxplot(x="Target", y="v2a1", data=train, ax=axes[1, 0])# monthly rent 

sns.boxplot(x="Target", y="rooms", data=train, ax=axes[1, 1])#no of rooms 

plt.show()
y = train.iloc[:,140]

y.unique()
X = train.iloc[:,1:141]

X.shape
bayes_cv_tuner = BayesSearchCV(

    #  Place your estimator here with those parameter values

    #      that you DO NOT WANT TO TUNE

    XGBClassifier(

       n_jobs = 2         # No need to tune this parameter value

      ),



    # 2.12 Specify estimator parameters that you would like to change/tune

    {

        'n_estimators': (100, 500),           # Specify integer-values parameters like this

       #'criterion': ['gini'],     # Specify categorical parameters as here

        'max_depth': (4, 100),                # integer valued parameter

       # 'max_features' : (10,64),             # integer-valued parameter

        #'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter

    },



    # 2.13

    n_iter=32,            # How many points to sample

    cv = 3                # Number of cross-validation folds

)