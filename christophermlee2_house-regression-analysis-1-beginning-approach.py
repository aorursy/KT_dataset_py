import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

pd.options.display.max_rows = 999

pd.options.display.max_columns = 999
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

print(df.shape)

print("ROWS: 1460 COLUMNS: 81")
df.head(10)
df.tail()
df.isnull().sum()
df.info()
## This code will seperate the features by their dtype "object" however; you can adjust this seperation into whatever dtype you want or need depending on your context or data.

categorical_features = df.select_dtypes(include="object")

numerical_features = df.select_dtypes(exclude="object")



print("This is the number of Categorical Features:", len(categorical_features.columns))

print("This is the number of Numerical Features: ", len(numerical_features.columns))
##This is the code to convert the missing values for the numerical features to the mean value.

## You would need to replace the features with their relevant names

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())



## This is the code to conver the missing values for the categorical features to the mode value.

df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode())