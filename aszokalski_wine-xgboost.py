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
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier, XGBRegressor





import missingno as msno

from eli5.sklearn import PermutationImportance
#load data

data = pd.read_csv("/kaggle/input/staszic1/train.csv")

data.head()
#examine data

msno.matrix(data)

#nothing is missing
#process data

columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

X = data[columns]

y = data.quality

X.head()
#examine labels

y.describe() 

#looks like we're dealing with multi-class classification problem
#split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.head()
#Grid search

"""

XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1, objective='binary:logistic',

booster='gbtree', tree_method='auto', n_jobs=1, gpu_id=-1, gamma=0, min_child_weight=1, max_delta_step=0, 

subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, 

scale_pos_weight=1, base_score=0.5, random_state=0, missing=None, **kwargs)

"""



test_model = XGBRegressor(learning_rate=0.1, n_estimatoors=70, max_depth=8, min_child_weight=4, gamma=0, colsample_bytree=0.75, subsample=0.65, reg_alpha=0.1, objective='reg:squarederror')

test_parameters_a1 = {

    'n_estimators' : [100, 200, 300, 400],

}



test_parameters_a2 = {

    'n_estimators' : [50, 70, 100, 120],

}



test_parameters_b1 = {

    'max_depth' : [4, 5, 6],

    'min_child_weight' : [4, 5, 6]

}



test_parameters_b2 = {

    'max_depth' : [8, 9],

    'min_child_weight' : [3, 4, 5]

}



test_parameters_c = {

   'gamma':[i/10.0 for i in range(0,5)],

}



test_parameters_d1 = {

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

}



test_parameters_d2 = {

 'subsample':[i/100.0 for i in range(65,75,5)],

 'colsample_bytree':[i/100.0 for i in range(75,85,5)]

}



test_parameters_e1 = {

 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

}

# test_parameters_e2 = {

#  'reg_alpha':[0.00002, 0.00003, 0.00004 ]

# }





gs_model = GridSearchCV(estimator = test_model,param_grid = test_parameters_e1, cv=5)

gs_model.fit(X,y)

print(gs_model.best_params_)



"""







perm = PermutationImportance(model, random_state=1).fit(train_X, train_y)

eli5.show_weights(perm, feature_names = train_X.columns.tolist())

"""
model = XGBRegressor(learning_rate=0.025, n_estimatoors=70, max_depth=8, min_child_weight=4, gamma=0, colsample_bytree=0.75, subsample=0.65, reg_alpha=0.1, objective='reg:squarederror')

model.fit(X, y)

model.score(X_test, y_test)
test = pd.read_csv("/kaggle/input/staszic1/test.csv")

wyniki = model.predict(test[columns])

wyniki = pd.DataFrame(wyniki, columns=['answer'])

wyniki.head()
wyniki.to_csv('wynik.csv')