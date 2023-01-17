!pip install pycaret
from pycaret.regression import *
import numpy as np 
import pandas as pd
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
train.head()
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
test.head()
features_drop = ['casual','registered']
train.drop(features_drop,axis = 1,inplace = True)
print(train.isnull().sum(axis=0))
print(test.isnull().sum(axis=0))
stp = setup(train, target="count", normalize = True)
models = compare_models(n_select=3)    # n_select - selecting top 3 best performed models