# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Data = pd.read_table('../input/orange_small_train.data').replace('\\', '/')
Data.head(10)
Data.tail(10)
Data.shape
Data.info()
Data.describe()
Data.apply(lambda x: x.count(), axis=1)
Data.isnull().sum(axis=0)
threshold = 0.2

Data = Data.drop(Data.std()[Data.std() < threshold].index.values, axis=1)

Data = Data.loc[:, pd.notnull(Data).sum()>len(Data)*.8]
DataVars = Data.columns
data_types = {Var: Data[Var].dtype for Var in DataVars}

for Var in DataVars:
    if data_types[Var] == int:
        x = Data[Var].astype(float)
        Data.loc[:, Var] = x
        data_types[Var] = x.dtype
    elif data_types[Var] != float:
        x = Data[Var].astype('category')
        Data.loc[:, Var] = x
        data_types[Var] = x.dtype

data_types
float_DataVars = [Var for Var in DataVars
                     if data_types[Var] == float]
float_DataVars
# mark zero values as missing or NaN
#Data[float_DataVars].isnull().replace(0, np.NaN)
# fill missing values with mean column values
#Data[float_DataVars].fillna(nData[float_DataVars].mean(), inplace=True)
# count the number of NaN values in each column
#Data[float_DataVars]  

float_x_means = Data.mean()

for Var in float_DataVars:
    x = Data[Var]
    isThereMissing = x.isnull()
    if isThereMissing.sum() > 0:
        Data.loc[isThereMissing.tolist(), Var] = float_x_means[Var]        
Data[float_DataVars].isnull().sum()
np.allclose(Data.mean(), float_x_means)
#Let's see the number of categories of each categorical feature:

DataVars = Data.columns

categorical_DataVars = [Var for Var in DataVars
                           if data_types[Var] != float]

categorical_levels = Data[categorical_DataVars].apply(lambda col: len(col.cat.categories))

categorical_levels
#Those variables having over 500 categories are likely to be just text / character data. 
#Let's get rid of them:

categorical_DataVars = categorical_levels[categorical_levels <= 500].index

categorical_DataVars
collapsed_categories = {}
removed_categorical_DataVars = set()

for Vars in categorical_DataVars:
    
    isTheremissing_value = Data[Vars].isnull()
    if isTheremissing_value.sum() > 0:
        Data[Vars].cat.add_categories('unknown', inplace=True)
        Data.loc[isTheremissing_value.tolist(), Vars] = 'unknown'
Data[categorical_DataVars].isnull().sum()
Data.info()



label = pd.read_table('../input/orange_small_train_churn.txt').replace('\\', '/')
label.head(10)
