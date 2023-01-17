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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer

targetted_predictor_cols=['Age','Fare','Sex']

def get_some_data():
    X=pd.read_csv('../input/titanic/train.csv')
    y=X.Survived
    x=X[targetted_predictor_cols]
    x=pd.get_dummies(x)
    my_imputer=Imputer()
    imputed_x=my_imputer.fit_transform(x)
    return imputed_x,y

X,y=get_some_data()
my_model=GradientBoostingRegressor()
my_model.fit(X,y)
my_plots = plot_partial_dependence(my_model, 
                                   features=[0,1,2],
                                   X=X, 
                                   feature_names=targetted_predictor_cols, 
                                   grid_resolution=10) 

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer

targetted_predictor_cols=['YearBuilt', 'LotArea', 'LotFrontage']

def get_some_data():
    X=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
    y=X.SalePrice
    x=X[targetted_predictor_cols]
    x=pd.get_dummies(x)
    my_imputer=Imputer()
    imputed_x=my_imputer.fit_transform(x)
    return imputed_x,y

X,y=get_some_data()
my_model=GradientBoostingRegressor()
my_model.fit(X,y)
my_plots = plot_partial_dependence(my_model, 
                                   features=[0,1,2],
                                   X=X, 
                                   feature_names=targetted_predictor_cols, 
                                   grid_resolution=10)

