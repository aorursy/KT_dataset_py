# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import seaborn as sns

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
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
test.head()
train.shape
test.shape
train["SalePrice"].describe()
#it should be normally distributed 

from scipy.stats import norm

sns.distplot(train.SalePrice, fit =norm);

#we have our target variable skewed to the left 
print(train['SalePrice'].skew())

print(train["SalePrice"].kurt())
# reduce the skewness to zero by doing log transformation 

train["SalePrice"] =np.log(train["SalePrice"])
sns.distplot(train.SalePrice, fit =norm)
print(train["SalePrice"].skew())

print(train["SalePrice"].kurt())
# train["SalePrice"]
import matplotlib.pyplot as plt
corr_mat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_mat,vmax=0.9, square=True)

plt.show();
corrmat = train.corr()

cols = corrmat.nlargest(21, 'SalePrice')['SalePrice'].index #specify number of columns to display i.e 21

f, ax = plt.subplots(figsize=(18, 10)) #size of matrix

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':12}, yticklabels=cols.values,

                 xticklabels=cols.values)



plt.yticks(rotation=0, size=15)

plt.xticks(rotation=90, size=15)

plt.title("Correlation Matrix",style='oblique', size= 20)

plt.show()
sns.scatterplot(x=train['LotArea'], y=train['SalePrice']);
train.columns
sns.scatterplot(x=train['YearBuilt'], y=train['SalePrice']);
sns.scatterplot(x=train['GarageArea'], y=train['SalePrice']);
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice']);
train.head()
#droppping the outliers 

train= train.drop(train[(train["GrLivArea"] > 4000)].index)
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice']);
sns.scatterplot(x=train['MSZoning'], y=train['SalePrice']);
sns.scatterplot(x=train['Street'], y=train['SalePrice']);
sns.boxplot(x=train['Street'], y=train['SalePrice']);
plt.figure(figsize=(15,8))

sns.boxplot(x=train['OverallQual'], y=train['SalePrice']);
# put together the training ad test set for cleaning 

#using concat, loc





#a data frame to show us the missing values 

train_null = train.isnull().sum()/len(train)*100

train_null[train_null >0].sort_values(ascending =False)
# filled out this columns with none

for col in ("PoolQC","MiscFeature","GarageType","FireplaceQu","Alley","GarageFinish", 

            "GarageQual","GarageCond","BsmtExposure","BsmtFinType2","BsmtFinType1",

            "BsmtCond","BsmtQual","MasVnrArea","MasVnrType","Fence"):

        train[col] = train[col].fillna("None")

train["LotFrontage"].isnull().sum()


train["LotFrontage"]
#fillLotfootage with median or mean or mode 



train["LotFrontage"]= train["LotFrontage"].fillna(train["LotFrontage"].median())
train["LotFrontage"].isnull().sum()
# fill up electrical with mode

train["Electrical"] = train["Electrical"].fillna(train["Electrical"].mode()[0])

train["Electrical"].isnull().sum()
train["Electrical"].head()
# Relational exprolation of categorical variables 

#box plots 
train.MSZoning.value_counts()
train_null2 = train.isnull().sum()/len(train)*100
train_null2[train_null2 >0].sort_values(ascending =False)
train["GarageYrBlt"] = train["GarageYrBlt"].fillna(0)
train.head()
train.info(verbose=True)
train.describe(include=['O']).columns
train.describe(include=['O']).columns
train.YrSold.value_counts()
# isNumeric = is_numeric(train)
numeric_feature = train.dtypes[train.dtypes != "O"].index
numeric_feature
# 	MSSubClass,YrSolda  --->let's add this as we go by'

train["YrSold"] = train["YrSold"].astype(str)

train["MSSubClass"] = train["MSSubClass"].astype(str)

train["OverallCond"] =train["OverallCond"].astype(str)

train["MoSold"] =train["MoSold"].astype(str)
cols = ("YrSold", "MSSubClass","OverallCond",

        "MoSold","MSZoning","Street","Alley","LotShape",

        "LandContour","LotConfig","SaleType","SaleCondition","MiscFeature","Fence",

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',

       'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive')
from sklearn.preprocessing import LabelEncoder

for c in cols:

    le = LabelEncoder()

    le.fit(list(train[c].values))

    train[c] = le.transform(list(train[c].values))

train["MoSold"].value_counts()
train.head()
#drop utillities "Utilities",PoolQC"

# train = train.drop(["RoofMatl"], axis =1)
train["LotConfig"].value_counts()
train.shape
train.head()
train["Fence"].value_counts()
# train.describe(include=['O']).columns
train["Exterior1st"].value_counts()
train.info(verbose=True)
train.head()
type(train[numeric_feature])
train[numeric_feature]
#looking for continous variables skweness example LotArea,



numeric_features_skew = train[numeric_feature].skew().sort_values(ascending =False)
numeric_features_skew
# transfrom them using boxcox 



skewed_features = numeric_features_skew[(numeric_features_skew) > 0.5]

skewed_features
from scipy import stats
#transform the columns using Boxcox

skewed_features,fitted_lambda = stats.boxcox(skewed_features)
fitted_lambda 
skewed_features
# find skewness of continous variables transfrom them using boxcox 

# normal distribution ___> identify those columns and boxcox 



# Lasso Regression and gradient boosting regression 

X_train = np.array([[ 1000., -1.,  2.],

                    [ 2.,  0.004,  0.],

                    [ 0.,  1., -1.]])
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax
# mean 0

# Standanard deviation 1