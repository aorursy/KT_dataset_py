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
# do all import here

import pandas as pd

import numpy as np
df_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_train
df_train.columns
cols_use=['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',

       'Fireplaces', 'GarageArea', 'MSZoning', 'PavedDrive', 'Neighborhood', 'BldgType', 'HouseStyle', 'SalePrice']
df=df_train[cols_use]

df.info()
# Import DictVectorizer

from sklearn.feature_extraction import DictVectorizer



# Convert df into a dictionary: df_dict

df_dict = df.to_dict("records")



# Create the DictVectorizer object: dv

dv = DictVectorizer(sparse=False)



# Apply dv on df: df_encoded

df_encoded = dv.fit_transform(df_dict)



# Print the resulting first five rows

print(df_encoded[:5,:])



# Print the vocabulary

print(dv.vocabulary_)
# Import necessary modules

from sklearn.feature_extraction import DictVectorizer

from sklearn.pipeline import Pipeline

import xgboost as xgb



X=df.iloc[:,:-1]

y=df.iloc[:,-1]

# Fill LotFrontage missing values with 0

X.LotFrontage = X.LotFrontage.fillna(0)



# Setup the pipeline steps: steps

steps = [("ohe_onestep", DictVectorizer(sparse=False)),

         ("xgb_model", xgb.XGBRegressor())]



# Create the pipeline: xgb_pipeline

xgb_pipeline = Pipeline(steps)



# Fit the pipeline

xgb_pipeline.fit(X.to_dict("records"), y)
# Import necessary modules

from sklearn.feature_extraction import DictVectorizer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

# Cross-validate the model

cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict("records"), y, cv=10, scoring="neg_mean_squared_error")



# Print the 10-fold RMSE

print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))