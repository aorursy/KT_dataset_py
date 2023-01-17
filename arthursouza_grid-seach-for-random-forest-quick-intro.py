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
train = pd.read_csv('/kaggle/input/learn-together/train.csv')

test = pd.read_csv('/kaggle/input/learn-together/test.csv')
def data_info(data):        

    info = pd.DataFrame()

    info['var'] = data.columns

    info['# missing'] = list(data.isnull().sum())

    info['% missing'] = info['# missing'] / data.shape[0]

    info['types'] = list(data.dtypes)

    info['unique values'] = list(len(data[var].unique()) for var in data.columns)

    

    return info
data_info(train)
x_train = train.drop(['Id', 'Cover_Type'], axis=1)

y_train = train['Cover_Type']
from sklearn.model_selection import KFold

cv_kfold = KFold(5, shuffle = False, random_state=12) 
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score



def grid_search(clf, X_train, y_train, params, score, cv):    

    grid = GridSearchCV(clf, params, scoring = score, cv = cv, return_train_score=True)

    grid_fitted = grid.fit(X_train, np.ravel(y_train))

    print ("Best score: %.4f" % grid_fitted.best_score_)

    print ("Best parameters: %s" % grid_fitted.best_params_)

    return grid_fitted, grid_fitted.best_estimator_, grid_fitted.cv_results_
%%time

from sklearn.ensemble import RandomForestClassifier



params = {

    'n_estimators':[128, 200, 256],

    'criterion' : ['gini'],

    'max_depth': [3, 5, 7, None],

    'max_features': ['sqrt']

}

clf = RandomForestClassifier()

grid, model, results = grid_search(clf, x_train, y_train, params, 'accuracy', cv_kfold)

model
%%time

model.fit(x_train, y_train)
%%time

x_test = test.drop('Id', axis=1)

y_pred = model.predict(x_test)
test['Cover_Type'] = y_pred

test[['Id', 'Cover_Type']].to_csv('submission.csv', index=False)
test.Cover_Type.value_counts()