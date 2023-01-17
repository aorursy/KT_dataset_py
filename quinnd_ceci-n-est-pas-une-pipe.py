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



import pandas as pd

import numpy as np

from datetime import date

from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler, PolynomialFeatures

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.feature_selection import SelectKBest

from category_encoders.binary import BinaryEncoder

from functools import reduce

from time import time

from scipy.special import comb



numerical_transformer = Pipeline(steps=[ 

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('encode', BinaryEncoder())

])



preprocessor = ColumnTransformer(

     transformers=[

         ('num', numerical_transformer, nums),

         ('cat', categorical_transformer, cats)])



d=2

# # Append classifier to preprocessing pipeline.

# # Now we have a full prediction pipeline.

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('poly_gen', PolynomialFeatures(degree=d)), 

                      ('select', SelectKBest()),

                      ('classifier', SGDClassifier(loss='log', tol=np.exp(-bb), max_iter=int(np.exp(bb))))])



bb = 10

j = 3

encoded_full = preprocessor.fit_transform(df_train).shape[1]

poly = comb(encoded_full+d, d, exact=True)

kmin = best_k-j*bb

kmax = best_k+j**j*bb



best_k = 60 # from a previous selectkbest

grid_params = {

    'classifier__alpha': [np.exp(-5), np.exp(-6.5), np.exp(-4)],#[np.exp(k) for k in range(-8,-1)],

    'select__k': range(kmin, kmax, j**j**j)

}



cv_ = 12

search = GridSearchCV(clf, param_grid=grid_params, iid=False, 

                      cv=cv_, return_train_score=True, verbose=7, 

                      n_jobs=-1)



NUMBER_OF_JOBS = cv_ * reduce(lambda x,y: x*y, [len(x) for x in grid_params.values()])#* search.get_params['cv']

print(NUMBER_OF_JOBS)



encoded_full, poly, (kmin, kmax)