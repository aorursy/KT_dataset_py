# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
print("Size of the DataFrame ", df.shape)

df.isnull().sum()[df.isnull().sum() > 0] 
df.head()
df.head().PoolQC
#B == Boolean Variable

#I did this cos there is a possibility that values may be missing cos the houses don't have them

dropCols = ["PoolQC","Fence", "MiscFeature", "Alley"]

df["PoolQCB"] = (df["PoolQC"].isnull()).astype(int) 

df["FenceB"] = (df["Fence"].isnull()).astype(int) 

df["MiscFeatureB"] = (df["MiscFeature"].isnull()).astype(int) 

df["AlleyB"] = (df["Alley"].isnull()).astype(int) 

df.head().PoolQCB
testdf = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
testdf["PoolQCB"] = (testdf["PoolQC"].isnull()).astype(int) 

testdf["FenceB"] = (testdf["Fence"].isnull()).astype(int) 

testdf["MiscFeatureB"] = (testdf["MiscFeature"].isnull()).astype(int) 

testdf["AlleyB"] = (testdf["Alley"].isnull()).astype(int) 
catCols = [c for c in df.columns if

                    df[c].nunique() < 10 and 

                    df[c].dtype == "object"]



numCols = [c for c in df.columns if 

                df[c].dtype in ['int64', 'float64']]
df.drop(dropCols, axis = 1, inplace = True)

testdf.drop(dropCols, axis = 1,inplace = True)
print(numCols)
df['OverallQual'].nunique()
plt.figure(figsize = (15,15))



#Baths

plt.subplot(331)

dfCorr = df[['BsmtHalfBath']+ ['FullBath']+['HalfBath']+['SalePrice']]

sns.heatmap(dfCorr.corr(), annot = True)



#Lot, Overall Conditions and Qualities

dfCorr = df[['LotFrontage']+ ['LotArea']+['OverallQual']+['SalePrice']]

plt.subplot(332)

sns.heatmap(dfCorr.corr(), annot = True)



#Basements

dfCorr = df[['BsmtFinSF1'] +['BsmtUnfSF']+['TotalBsmtSF']+['SalePrice']] #['BsmtUnfSF'] ['BsmtFinSF1'] + 

plt.subplot(333)

sns.heatmap(dfCorr.corr(), annot = True)



#BoolVariables

dfCorr = df[['PoolQCB'] +['FenceB']+['MiscFeatureB']+['AlleyB']+['SalePrice']]

plt.subplot(334)

sns.heatmap(dfCorr.corr(), annot = True)



#Years

plt.subplot(335)

dfCorr = df[['YearBuilt']+ ['YearRemodAdd']+['SalePrice']]

sns.heatmap(dfCorr.corr(), annot = True)
#Numerical Columns to be dropped

#['BsmtUnfSF'] ['BsmtFinSF1'] 'BsmtHalfBath','OverallCond', 'BsmtFinSF2', 

dropCols2 = ['PoolQCB', 'FenceB', 'MiscFeatureB', 'AlleyB'] 
sns.set()

plt.figure(figsize = (20,20))

plt.subplot(331)

sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice', data = df, hue = 'OverallQual', alpha = 0.85)



plt.subplot(332)

sns.scatterplot(x = 'PoolQCB', y = 'SalePrice', data = df, hue = 'OverallQual', alpha = 0.85)
target = df["SalePrice"]
df.drop(["SalePrice"], axis = 1, inplace = True)

df.drop(dropCols2, axis = 1, inplace = True)

testdf.drop(dropCols2, axis = 1, inplace = True)
numCols = [c for c in df.columns if 

                df[c].dtype in ['int64', 'float64']]



catCols = [c for c in df.columns if

                    df[c].nunique() < 10 and 

                    df[c].dtype == "object"]

print(numCols)
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
train, valid, yTrain, yValid = train_test_split(df, target, train_size=0.85, random_state=0)
cols = catCols + numCols  

train = train[cols].copy()

valid = valid[cols].copy()

test = testdf[cols].copy()
valid.head()
train.shape
# Preprocessing for numerical data

numTrans = SimpleImputer(strategy='mean')



# Preprocessing for categorical data

catTrans = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])
# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numTrans, numCols),    

        ('cat', catTrans, catCols)

    ])

#,        ('cat', catTrans, catCols)





# Define model

model = RandomForestRegressor(n_estimators=200, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])
from xgboost import XGBRegressor
"""

# Preprocessing of training data, fit model 

clf.fit(train, yTrain)



# Preprocessing of validation data, get predictions

preds = clf.predict(valid)



print('MAE:', mean_absolute_error(yValid, preds))

"""
model1 = XGBRegressor(n_estimators=700, learning_rate=0.0255, early_stopping_rounds = 5, random_state = 0)



# Bundle preprocessing and modeling code in a pipeline

clf1 = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model1)

                     ])
clf1.fit(train, yTrain)



# Preprocessing of validation data, get predictions

preds1 = clf1.predict(valid)



print('MAE:', mean_absolute_error(yValid, preds1))

predsTest = clf1.predict(test)
output = pd.DataFrame({'Id': test.Id,

                       'SalePrice': predsTest})

output.to_csv('submission.csv', index=False)
