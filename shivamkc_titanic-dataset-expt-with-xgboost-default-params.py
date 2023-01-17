# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import relevant libraries

import sklearn 

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import make_scorer

import xgboost as xgb
def train_data_clean(data):

    data_nullcount = data.isnull().sum()

    col_null_vals = list(data_nullcount[data_nullcount!=0].index)

    if len(col_null_vals)==0:

        return data

    data_dtypes = data.dtypes

    for colname in col_null_vals:

        if data_dtypes[colname]==object:

            data[colname]=data[colname].fillna(data[colname].mode()[0])

        else:

            data[colname]=data[colname].fillna(data[colname].mean())

    return data
def test_impute_using_train_data(test_data,train_data):

    data_test_isnull = test_data.isnull().sum()

    null_collist = list(data_test_isnull[data_test_isnull!=0].index)

    if len(null_collist)==0:

        return test_data

    data_dtypes = test_data.dtypes

    for colname in null_collist:

        if data_dtypes[colname]==object:

            test_data[colname]=test_data[colname].fillna(train_data[colname].mode()[0])

        else:

            test_data[colname]=test_data[colname].fillna(train_data[colname].mean())

    return test_data
def test_data_prep_titanic(data,columns_to_drop,colnames):

    data = data.drop(columns_to_drop,axis=1)

    data_dum = pd.get_dummies(data,drop_first=True)

    data_dum_2 = data_dum[colnames]

    xdata = data_dum_2.values

    return xdata
def data_prep_titanic(data,columns_to_drop):

    data = data.drop(columns_to_drop,axis=1)

    data_train = data.drop(['Survived'],axis=1)

    data_dum = pd.get_dummies(data_train,drop_first=True)

    ydata = data['Survived'].values

    xdata = data_dum.values

    colnames = data_dum.columns

    return xdata,ydata,colnames
def accuracy_metric(ytrue,ypred):

    score = sum(ypred==ytrue)/len(ypred)

    return score
# Perform grid search on xgboost

def xgboost_grid_search(model,params_grid,xtrain,ytrain):

    score_obj = make_scorer(accuracy_metric, greater_is_better=True)

    clf_search = GridSearchCV(model, params_grid,scoring=score_obj)

    clf_search.fit(xtrain,ytrain)

    return clf_search.best_params_
train_data_part = pd.read_csv('../input/titanic/train.csv')

train_data_new = train_data_clean(train_data_part.copy())

cols_todrop = ['PassengerId','Name','Ticket','Cabin']

xdata,ydata,colnames = data_prep_titanic(train_data_new.copy(),cols_todrop)

xtrain,xtest,ytrain,ytest = train_test_split(xdata,ydata,test_size=0.2,random_state=100,stratify=ydata)
params_grid = {

    'learning_rate':[0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3],

    'gamma':[0,1,2,3,5,7,9],

    'max_depth':[4,5,6,7,8,9],

    'n_estimators':[10,30,50,70,90,100]

}
# Performing grid search for xgbclassifier

estimator = xgb.XGBClassifier()

best_params = xgboost_grid_search(estimator,params_grid,xtrain,ytrain)



learning_rate = best_params['learning_rate']

gamma = best_params['gamma']

max_depth = best_params['max_depth']

n_estimators = best_params['n_estimators']



clf = xgb.XGBClassifier(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,gamma=gamma)

clf.fit(xtrain,ytrain)
# Accuracy of the best model on train set

ypred_train = clf.predict(xtrain)

print('Train set accuracy for XGBoost using GridSearchCV:',accuracy_metric(ytrain,ypred_train))
# Accuracy on the test set

ypred = clf.predict(xtest)

print('Test set accuracy for XGBoost using GridSearchCV:',accuracy_metric(ytest,ypred))
# Default Model Comparison and looking for accuracy on test set

clf_def = xgb.XGBClassifier()

clf_def.fit(xtrain,ytrain)

ypred_def = clf_def.predict(xtest)

print('Test set accuracy for Default XGBoost:',accuracy_metric(ytest,ypred_def))
# Default Model Accuracy on train set

ypred_def_train = clf_def.predict(xtrain)

print('Train set accuracy for Default XGBoost:',accuracy_metric(ytrain,ypred_def_train))
test_data = pd.read_csv('../input/titanic/test.csv')

test_data_clean = test_impute_using_train_data(test_data.copy(),train_data_new.copy())

dtest = test_data_prep_titanic(test_data_clean,cols_todrop,colnames)

dpred = clf.predict(dtest)



test_data_submit = test_data_clean.copy()

test_data_submit['Survived'] = dpred

test_data_submit_part = test_data_submit[['PassengerId','Survived']]

test_data_submit_part.to_csv('submissions.csv',index=False)