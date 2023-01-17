# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import packages

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

import sklearn.metrics as metrics

import math
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")



test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



#Creating a copy of the train and test datasets

c_test  = test.copy()

c_train  = train.copy()
#Verifying if everything is okay for train set et test set

c_train.head()
c_test.head()
#Assemble train set and test set

c_train['train']  = 1

c_test['train']  = 0

df = pd.concat([c_train, c_test], axis=0, sort=False)
df.head()
#Seeing rapidly NAN values

sns.heatmap(df.isna())
#Creating a df of %'s nan values

NAN = [(col, df[col].isna().mean()*100) for col in df]

NAN = pd.DataFrame(NAN, columns=["name", "perc"])



NAN
#Count the number of features that have more than 40% of NAN

NAN = NAN[NAN.perc > 40]

NAN.sort_values("perc", ascending=False)
#4 features with more than 40% of NAN values, we can drop them

df = df.drop(["PoolQC", "MiscFeature", "Alley", "Fence"], axis=1)

df.head()
#Select numerical and categorical features

object_features = df.select_dtypes(include=['object'])

numerical_features = df.select_dtypes(exclude=['object'])
#Null values for columns

null_counts = object_features.isnull().sum() 

print("Null values in features :\n", null_counts)
#Fill all features that have more than 50 null values

for c in object_features:

    if df[c].isnull().sum() > 50:

        df[c] = df[c].fillna('None')

        

df["BsmtQual"].value_counts()
#Fill all features that have less than 50 null values

for c in object_features:

    if df[c].isnull().sum() < 50:

        df[c] = df[c].fillna('None')

        

df["BsmtExposure"].value_counts()
#Null values for columns

null_counts = numerical_features.isnull().sum()

print("Null values in features :\n", null_counts)
#Fill features with logical values for GarageYrBlt and LotFrontage

print(numerical_features["LotFrontage"].median())

print((numerical_features["YrSold"]-numerical_features["YearBuilt"]).median())
#For GarageYrBlt we'll fill with 1979 because, 2014 - 35 = 1979

numerical_features['GarageYrBlt'] = numerical_features['GarageYrBlt'].fillna(numerical_features['YrSold']-35)

numerical_features['LotFrontage'] = numerical_features['LotFrontage'].fillna(68)
#Fill the rest with 0

numerical_features = numerical_features.fillna(0)
plt.rcParams.update({'figure.max_open_warning': 0})



#Visualising columns

for col in object_features:

    plt.figure(figsize=[10,3])

    object_features[col].value_counts().plot(kind='bar', title=col)
#Some features have bad variance, so we'll delete them

object_features = object_features.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)
def encoding(object_features):

    

    code = {'TA':2,'Gd':3, 'Fa':1,'Ex':4,'Po':1,

            'None':0,'Y':1,'N':0,'Reg':3,'IR1':2,

            'IR2':1,'IR3':0,"None" : 0,

            "No" : 2, "Mn" : 2, "Av": 3,"Gd" : 4,

            "Unf" : 1, "LwQ": 2, "Rec" : 3,

            "BLQ" : 4, "ALQ" : 5, "GLQ" : 6

                }

    

    object_features['ExterQual'] = object_features['ExterQual'].map(code)

    object_features['ExterCond'] = object_features['ExterCond'].map(code)

    object_features['BsmtCond'] = object_features['BsmtCond'].map(code)

    object_features['BsmtQual'] = object_features['BsmtQual'].map(code)

    object_features['HeatingQC'] = object_features['HeatingQC'].map(code)

    object_features['KitchenQual'] = object_features['KitchenQual'].map(code)

    object_features['FireplaceQu'] = object_features['FireplaceQu'].map(code)

    object_features['GarageQual'] = object_features['GarageQual'].map(code)

    object_features['GarageCond'] = object_features['GarageCond'].map(code)

    object_features['CentralAir'] = object_features['CentralAir'].map(code)

    object_features['LotShape'] = object_features['LotShape'].map(code)

    object_features['BsmtExposure'] = object_features['BsmtExposure'].map(code)

    object_features['BsmtFinType1'] = object_features['BsmtFinType1'].map(code)

    object_features['BsmtFinType2'] = object_features['BsmtFinType2'].map(code)



    PavedDrive =   {"N" : 0, "P" : 1, "Y" : 2}

    object_features['PavedDrive'] = object_features['PavedDrive'].map(PavedDrive)

        

    rest_object_features = object_features.select_dtypes(include=['object'])

    object_features = pd.get_dummies(object_features, columns=rest_object_features.columns) 

   

    return object_features
final_df = pd.concat([encoding(object_features), numerical_features], axis=1, sort=False)

final_df.head()