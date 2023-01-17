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
df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.ticker as mtick

from sklearn.model_selection import train_test_split

from sklearn import metrics as sm

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso

from sklearn.tree import DecisionTreeClassifier

from scikitplot.metrics import plot_roc_curve

from sklearn.metrics import mean_squared_error,r2_score
df.head()
df.shape
df.isna().sum()
df['LotFrontage'].fillna(value=df['LotFrontage'].median(),inplace=True)
df.isnull().sum()
df.dtypes
cat=df.select_dtypes(include="object")

num=df.select_dtypes(include="number")
cat=pd.get_dummies(cat,dtype='int')

df=pd.concat([cat,num],axis=1)
import seaborn as sns

def heatMap(df):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(20, 14))

    sns.heatmap(corr, annot=True, fmt=".2f")

    plt.xticks(range(len(corr.columns)), corr.columns);

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.show()
heatMap(df)
print(df['SalePrice'].min())

print(df['SalePrice'].max())

print(df['SalePrice'].median())
df.columns
df['SalePrice_range']=pd.cut(df['SalePrice'],[34000,50000,100000,250000,500000,755000],right=True)
#VISUALISATION#
LA_SPR=df.groupby(['SalePrice_range'])['LotArea'].size().plot(kind='bar',stacked=True)

LA_SPR.set_ylabel('LotArea')

LA_SPR.set_xlabel('SalePrice_range')



for i in LA_SPR.patches:

    width, height = i.get_width(), i.get_height()

    x, y =i.get_xy() 

    LA_SPR.annotate('{:.0f}%'.format(height), (i.get_x()+.40*width, i.get_y()+.3*height),

                color = 'RED')
df['SaleCondition'].value_counts().plot(kind='bar',grid=True,figsize=(10,7))
df['SaleType'].value_counts().plot(kind='bar',grid=True,figsize=(10,7))
df.groupby(['SaleCondition','SaleType']).size().unstack().plot(kind='bar',grid=True)
df.pivot_table(index='SaleType',aggfunc='mean',dropna=True)
df['LotShape'].unique()
df['LandContour'].unique()
df['Utilities'].unique()
df.columns
train_y=df['SalePrice']

train_x=df[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition']]