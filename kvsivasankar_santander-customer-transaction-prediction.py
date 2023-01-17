# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline

from sklearn.model_selection import train_test_split
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
from tqdm import tqdm

df_test=test.drop(['ID_code'], axis=1)

df_test = df_test.values

unique_samples = []

unique_count = np.zeros_like(df_test)

for feature in tqdm(range(df_test.shape[1])):

    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)

    unique_count[index_[count_ == 1], feature] += 1



# Samples which have unique values are real the others are fake

real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]

synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]



df_test_real = df_test[real_samples_indexes].copy()

df_test_real=pd.DataFrame(df_test_real)

df_test_real=df_test_real.add_prefix('var_')

df_test_real.head()
train_value=train.drop(['ID_code', 'target'], axis=1)

df_combined=pd.concat([train_value, df_test_real])
df_combined.shape
for i in range(25):

    var='var_'+str(i)

    if i%25==0:

        print (i)

    dictionary=df_combined[var].value_counts().to_dict()

    train['count_'+var]=train[var].map(dictionary)

    train['test_'+var]=train[var]*np.log2(train['count_'+var]+1)

    train['test1_'+var]=train[var]/np.log2(train['count_'+var]+1)

    train['test2_'+var]=train[var]*(-np.log2(train['count_'+var]+1))

    train.drop('count_'+var, inplace=True, axis=1)

    dictionary1=df_test_real[var].value_counts().to_dict()

    test['count_'+var]=test[var].map(dictionary)

    test['test_'+var]=np.log2(test['count_'+var]+1)*test[var]

    test['test1_'+var]=test[var]/np.log2(test['count_'+var]+1)

    test['test2_'+var]=test[var]*(-np.log2(test['count_'+var]+1))

    test.drop('count_'+var, inplace=True, axis=1)
ID_code=test['ID_code']

X_test = test.drop(['ID_code'],axis = 1)

#X_test=X_test[X_test.columns[:200].append(X_test.columns[400:])]

X_test.head()
y=train['target']

X = train.drop(['target', 'ID_code'], axis=1)

#X=X[X.columns[:200].append(X.columns[400:])]

X.head()
features = [c for c in X.columns if c not in ['ID_code', 'target']]
features
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from sklearn.metrics import make_scorer, accuracy_score,roc_auc_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

auc_scorer = make_scorer(roc_auc_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=auc_scorer)

grid_obj = grid_obj.fit(X.iloc[:1000,:], y[:1000])



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X.iloc[:1000,:], y[:1000])
predictions = clf.predict(X.iloc[:1000,:])

print(roc_auc_score(y[:1000], predictions))

from sklearn.metrics import make_scorer, accuracy_score,roc_auc_score

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

auc_scorer = make_scorer(roc_auc_score)



# Run the grid search



random_search_obj = RandomizedSearchCV(clf, param_distributions=parameters,

                                   n_iter=100, cv=5)



#grid_obj = GridSearchCV(clf, parameters, scoring=auc_scorer)

random_search_obj = random_search_obj.fit(X.iloc[:1000,:], y[:1000])



# Set the clf to the best combination of parameters

clf = random_search_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X.iloc[:1000,:], y[:1000])
predictions = clf.predict(X.iloc[:1000,:])

print(roc_auc_score(y[:1000], predictions))

from hyperopt import hp, tpe

from hyperopt.fmin import fmin



from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer



### make a scorer fn

auc_scorer = make_scorer(roc_auc_score)



X = X.iloc[:1000,:]

y = y[:1000]



### define obj

def objective(params):

    params = {'n_estimators': int(params['n_estimators']), 'max_depth': int(params['max_depth'])}

    clf = RandomForestClassifier(n_jobs=4, class_weight='balanced', **params)

    score = cross_val_score(clf, X, y, scoring=auc_scorer, cv=StratifiedKFold()).mean()

    print("Gini {:.3f} params {}".format(score, params))

    return -score



### define search space

space = {

    'n_estimators': hp.quniform('n_estimators', 25, 500, 25),  ### quniform defines how values will be sampled

    'max_depth': hp.quniform('max_depth', 1, 10, 1)            ### there are other parameteric distributions also available

}



### put together

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=3)   ### increase for best results
print("Hyperopt estimated optimum {}".format(best))
bestParams = {'n_estimators': best['n_estimators'],

              'max_depth': best['max_depth']

             }

bestModel = RandomForestClassifier(n_estimators = int(best['n_estimators']),max_depth= int(best['max_depth']))



bestModel = bestModel.fit(X,y)



predictions = bestModel.predict(X)

print(roc_auc_score(y, predictions))
from sklearn.model_selection import StratifiedKFold



from sklearn.linear_model import LogisticRegression
folds = 5

kf = StratifiedKFold(n_splits=folds)  

#kf.get_n_splits(X,y)







#scores = cross_val_score(model, X, Y, cv=5, scoring='r2')

#print (scores,np.mean(scores))



lr = LogisticRegression()

#kf = KFold(n=data.shape[0], n_folds=5, shuffle=True, random_state=8)



accuracies = cross_val_score(lr, X,y, scoring='accuracy', cv = kf)

print (accuracies,np.mean(accuracies))
