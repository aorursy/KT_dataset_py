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
train_file_path = '../input/train.csv' 
test_file_path = '../input/test.csv' 
submission_file_path = 'submission.csv' 

train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

train.fillna(0)
test.fillna(0)

train.dtypes
data_predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']

train_y = train['Survived']
train_X = train[data_predictors]
test_X = test[data_predictors]

print(test_X.head())
train_x = pd.get_dummies(train_X)
test_x = pd.get_dummies(test_X)
train_x = train_x.fillna(0)
train_x
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor

plot_model = GradientBoostingRegressor()
plot_model.fit(train_x, train_y)
dep_plots = plot_partial_dependence(plot_model,       
                                   features=[1, 3], 
                                   X=train_x,            # raw predictors data.
                                   feature_names=['Pclass', 'Age', 'SibSp', 'Fare'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis
from xgboost import XGBRegressor
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

data_model = make_pipeline(Imputer(), XGBRegressor())
data_model.fit(train_x, train_y)

scores = cross_val_score(data_model, train_x, train_y, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error %2f' %(-1 * scores.mean()))
prediction = data_model.predict(test_x)
print('Estimated survivors: ' + str(prediction.astype(int).sum()) + ' passengers')
result = test.assign(Survived=prediction.astype(int))
result.to_csv(submission_file_path,sep=',',columns=['PassengerId', 'Survived'], index=False)