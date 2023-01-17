#!pip install --upgrade pip

!pip install fastai==0.7.0 ## Based on Fast.ai ML course



%load_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np 

import pandas as pd

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from fastai.imports import *

from fastai.structured import *

import os

from matplotlib import pyplot as plt

import math



import lime

import lime.lime_tabular



import shap

import eli5

from eli5.sklearn import PermutationImportance



print(os.listdir("../input/"))
train_df = pd.read_csv("../input/train.csv", index_col = 'Id')

test_df = pd.read_csv("../input/test.csv", index_col = 'Id')

train_df.head()
train_df.drop(['MSSubClass', 'MSZoning'],axis =1, inplace = True)

train_df.dropna(axis = 1, how ='any',inplace = True)

train_df = train_df.select_dtypes(exclude=['object'])
train_X, val_X, train_y, val_y = train_test_split(train_df.drop(['SalePrice'],axis=1), train_df['SalePrice'], test_size=0.30, random_state=42)
m = RandomForestRegressor(n_estimators=1, min_samples_leaf=3, n_jobs=-1, max_depth = 3, oob_score=True) ## Use all CPUs available

m.fit(train_X, train_y)
draw_tree(m.estimators_[0], train_X, precision=3)
model = RandomForestRegressor(n_estimators=20, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True) ## Use all CPUs available

model.fit(train_X, train_y)
perm = PermutationImportance(m, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
feature = np.array(['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',

       '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',

       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold', 'YrSold'])
categorical_features = np.argwhere(np.array([len(set(train_X.values[:,x])) for x in range(train_X.values.shape[1])]) <= 20).flatten()
explainer = lime.lime_tabular.LimeTabularExplainer(train_X.values, feature_names=feature, class_names=['SalePrice'], categorical_features=categorical_features, verbose=True, mode='regression')
exp = explainer.explain_instance(val_X.values[25], model.predict, num_features=5)

exp.show_in_notebook(show_table=True)
exp = explainer.explain_instance(val_X.values[73], model.predict, num_features=5)

exp.show_in_notebook(show_table=True)
exp = explainer.explain_instance(val_X.values[173], model.predict, num_features=5)

exp.show_in_notebook(show_table=True)