# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
X = train.copy()

X.dropna(axis=0, subset=['Cover_Type'], inplace=True)

y = X.Cover_Type           

X.drop(['Cover_Type'], axis=1, inplace=True)
class SetIndex(BaseEstimator, TransformerMixin):

    def __init__(self, column = 'id'):

        #print(subset)

        self.column = column



    def transform(self, X, *_):

        #print(X)

        return X.set_index(self.column)



    def fit(self, *_):

        return self



set_index = SetIndex(column = 'Id')
preprocessor = Pipeline( steps = [

    ('set_index', set_index),

])
model1 = RandomForestClassifier(n_estimators=100)

model2 = XGBRegressor(objective = 'reg:squarederror')
my_pipeline = Pipeline( steps = [

    ('preprocessor', preprocessor),

    ('model', model1),

])
param_grid = {

    'model__n_estimators': [x for x in range(50, 1550, 150)]

}

grid = GridSearchCV(my_pipeline, cv=2, param_grid=param_grid, verbose=8, n_jobs=10)
grid.fit(X,y)
print("Best: %f using %s" % (grid.best_score_, 

    grid.best_params_))
means = grid.cv_results_['mean_test_score']

stds = grid.cv_results_['std_test_score']

params = grid.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))

    pass
grid.refit
preds_test = grid.predict(test)
# Save test predictions to file

output = pd.DataFrame({'Id': test.Id,

                       'Cover_Type': preds_test})

output.to_csv('submission.csv',   index=False)