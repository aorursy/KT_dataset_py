!pip install impyute

!pip install datawig
from sklearn.datasets import fetch_california_housing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_error

from math import sqrt

import random

import pandas as pd

import numpy as np



# Imputaion libray 

from sklearn.impute import SimpleImputer

import sys

from impyute.imputation.cs import fast_knn, mice

import datawig
# Random seed

random.seed(0)
# Dataset

dataset = fetch_california_housing()

x = pd.DataFrame(dataset.data) 

y = pd.DataFrame(dataset.target)

x.columns = ['0','1','2','3','4','5','6','7']

x.insert(loc=len(x.columns), column='target', value=y)
#Randomly replace 40% of the first column with NaN values

column = x['0']

print(column.size)

missing_pct = int(column.size * 0.4)

i = [random.choice(range(column.shape[0])) for _ in range(missing_pct)]

column[i] = np.NaN

print(column.shape[0])
# Simple imputation 

imp_mean = SimpleImputer(strategy='mean')

imp_mean.fit(x)

imputed_x = imp_mean.transform(x)
imp_mean = SimpleImputer( strategy='most_frequent')

imp_mean.fit(x)

imputed_train_df = imp_mean.transform(x)
# KNN imputation using Impyute libray

sys.setrecursionlimit(100000) #Increase the recursion limit of the OS

imputed_training=fast_knn(x.values, k=30)
# MICE Multivariate Imputation by Chained Equation

imputed_training = mice(x.values)
# Deep learning imputation

df_train, df_test = datawig.utils.random_split(x)

#Initialize a SimpleImputer model

imputer = datawig.SimpleImputer(

    input_columns=['1','2','3','4','5','6','7', 'target'], # column(s) containing information about the column we want to impute

    output_column= '0', # the column we'd like to impute values for

    output_path = 'imputer_model' # stores model data and metrics

    )

#Fit an imputer model on the train data

imputer.fit(train_df=df_train, num_epochs=10)



#Impute missing values and return original dataframe with predictions

imputed = imputer.predict(df_test)