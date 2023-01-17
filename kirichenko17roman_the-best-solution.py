# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_train.csv")

test = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_test.csv")

submission = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_submission.csv")
train.head()
train.shape
train.info()
y = train["covid_19"].values

y
y.mean()
df = pd.concat([train, test])

df.shape
df.isna().mean(0).sort_values().hist()
df.isna().mean(0).values
df = df.drop((df.isna().mean(0)).index[df.isna().mean(0) > 0.95].values, axis = 'columns')
df.head()
df.groupby('age_quantile').agg({'Platelets': np.median})
df['Platelets_gr'] = df['Platelets'] - df.groupby('age_quantile')['Platelets'].transform(np.median)
df.groupby('age_quantile').agg({'Monocytes': np.median})
df['Monocytes_gr'] = df['Monocytes'] - df.groupby('age_quantile')['Monocytes'].transform(np.median)

df['Leukocytes_gr'] = df['Leukocytes'] - df.groupby('age_quantile')['Leukocytes'].transform(np.median)
df.fillna(-999, inplace = True)

df
df.info()
df = df.drop('covid_19', axis='columns') 
df = df.drop('id', axis='columns') 
train_X = df.iloc[:4000, :].select_dtypes(exclude=['object']).values

test_X = df.iloc[4000:, :].select_dtypes(exclude=['object']).values
#from imblearn.under_sampling import RandomUnderSampler



#ros = RandomUnderSampler(random_state=0)

#train_X, train_y = ros.fit_sample(train_X, y)
#from imblearn.over_sampling import RandomOverSampler



#ros = RandomOverSampler(random_state=0)

#train_X, train_y = ros.fit_sample(train_X, y)
#from imblearn.over_sampling import SMOTE



#ros = SMOTE(k_neighbors=50, random_state=0)

#train_X, train_y = ros.fit_sample(train_X, y)
train_X.shape
from sklearn.tree import DecisionTreeClassifier



tree_clf = DecisionTreeClassifier(

    max_depth=10, 

    min_samples_leaf=28, 

    max_features=0.94, 

    criterion="gini",                                    

    random_state=1)



dt = tree_clf.fit(train_X, y)

y_pred = dt.predict_proba(test_X)
imp = pd.DataFrame({

    'colnames': df.columns,

    'importances': tree_clf.feature_importances_

})



good_columns = imp.loc[imp['importances'] != 0, :].colnames.values
imp
train_X = df.iloc[:4000, :].loc[:, good_columns].values

test_X = df.iloc[4000:, :].loc[:, good_columns].values
#from imblearn.over_sampling import SMOTE



#ros = SMOTE(k_neighbors=50, random_state=0)

#train_X, train_y = ros.fit_sample(train_X, y)
#from sklearn.tree import DecisionTreeClassifier

#

#tree_clf = DecisionTreeClassifier(

#    max_depth=10, 

#    min_samples_leaf=21, 

#    max_features=0.9, 

#    criterion="gini",                                    

#    random_state=1)

#

#dt = tree_clf.fit(train_X, y)

#y_pred = dt.predict_proba(test_X)
#from sklearn.model_selection import GridSearchCV

#

#param_grid = {

#    'max_depth': [10], 

#    'min_samples_leaf': [28], 

#    'max_features': [0.94, 0.93, 0.95] 

#}

#

#grid = GridSearchCV(tree_clf, cv=5, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')

#grid.fit(train_X, y)
#grid.best_score_
#grid.best_params_
#y_pred = grid.best_estimator_.predict_proba(test_X)
#from sklearn.tree import DecisionTreeClassifier

#

#tree_clf = DecisionTreeClassifier(

#    max_depth=10, 

#    min_samples_leaf=28, 

#    max_features=0.94, 

#    criterion="gini",                                    

#    random_state=1)

#

#dt = tree_clf.fit(train_X, y)

#y_pred = dt.predict_proba(test_X)
#pd.Series(y_pred[:, 1]).hist()
import lightgbm as lgb



#bt = lgb.LGBMClassifier(n_jobs=-1, objective='binary', boosting_type='gbdt', learning_rate=0.01, random_state=1)

#

#

#grid_params = {

#    'n_estimators': [200, 250, 300],

#    'colsample_bytree': [0.9, 0.85, 0.8],

#    'num_leaves': [5, 10, 20],

#    'subsample': [1],

#    'min_split_gain': [0],

#    'reg_alpha': [0.0, 0.01], 

#    'reg_lambda': [0.0, 0.1],

#    'min_child_samples': [10, 20,30,40]

#}

#

#grid = GridSearchCV(bt, cv=5, param_grid=grid_params, n_jobs=-1, scoring='roc_auc')

#grid.fit(train_X, y)
#grid.best_score_
#grid.best_params_
#grid.cv_results_['split0_test_score'].mean(), grid.cv_results_['split1_test_score'].mean(), grid.cv_results_['split2_test_score'].mean(), grid.cv_results_['split3_test_score'].mean(), grid.cv_results_['split0_test_score'].mean()
bt = lgb.LGBMClassifier(colsample_bytree=0.85, min_child_samples=20, min_split_gain=0, n_estimators=250,

                        num_leaves=10, reg_alpha=0.0, reg_lambda=0.1, subsample=1, n_jobs=-1, objective='binary', 

                        boosting_type='gbdt', learning_rate=0.01, random_state=1)
bt.fit(train_X, y)
bt.predict_proba(test_X)[:,1]
submission['covid_19'] = bt.predict_proba(test_X)[:,1]

submission
submission.to_csv('submission.csv', index=False)