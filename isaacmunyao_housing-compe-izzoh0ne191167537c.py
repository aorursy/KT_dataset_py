# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Bring in tha data
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#Confirming import of data
train.head()
test.head()
#Df shape
train.shape
test.shape
#df describe
train.describe()
train.columns
test.columns
#describe saleprice
train["SalePrice"].describe()
#Check for distribution of data
#using seaborn
from scipy.stats import norm
sns.distplot(train.SalePrice, fit =norm);
#check skewness and kurtosis
print(train['SalePrice'].skew())
print(train['SalePrice'].kurt())
#Reduce skewness using log transformation
SingleLog_y = np.log1p(train['SalePrice']) 
sns.distplot(SingleLog_y, color ="r")
#print skewness and kurtosis after 1st reduction
print("Skew after 1st Log Transformation: %f" % SingleLog_y.skew())
print("Kurt after 1st Log Transformation: %f" % SingleLog_y.kurt())
#Further Reduction of Skewness by taking Logrithm of Logrithm
DoubleLog_y = np.log1p(SingleLog_y)
sns.distplot(DoubleLog_y, color ="r")
print("Skew after 2nd Log Transformation: %f" % DoubleLog_y.skew())
print("Kurt after 2nd Log Transformation: %f" % DoubleLog_y.kurt())
#Correlation
import matplotlib.pyplot as plt

plt.matshow(train.corr())
plt.show()
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
#scatter plots 
sns.scatterplot(x=train['YearBuilt'], y=train['SalePrice'])
sns.scatterplot(x=train['GarageArea'], y=train['SalePrice']);
#boxplot
sns.boxplot(x=train['Street'], y=train['SalePrice'])
#dropping the outliers
train= train.drop(train[(train["GrLivArea"] > 4000)].index)
train
test
#Concat train and test
combo = pd.concat([train, test], axis ='index')
combo
#total % of missing values 
combo_null = combo.isna()
combo_null.sum()/len(combo)*100
#missing values sol 2
combo_null = combo.isna().sum()/len(combo)*100
combo_null[combo_null >0].sort_values(ascending =False)
combo.info()
# filled out this columns with none
for col in ("PoolQC","MiscFeature","GarageType","FireplaceQu","Alley","GarageFinish", 
            "GarageQual","GarageCond","BsmtExposure","BsmtFinType2","BsmtFinType1",
            "BsmtCond","BsmtQual","MasVnrArea","MasVnrType","Fence"):
        combo[col] = combo[col].fillna("None")
#check is columns have been filled
combo["PoolQC"]
#Column has missing values and is of numeric type
combo["LotFrontage"]
#sum of none values in column
combo["LotFrontage"].isnull().sum()
#fillLotfootage with median 
combo["LotFrontage"]= combo["LotFrontage"].fillna(combo["LotFrontage"].median())
#check if column has been filled with median
combo["LotFrontage"].isnull().sum()
#there are no null values
#fill up electrical with mode
combo["Electrical"] = combo["Electrical"].fillna(combo["Electrical"].mode()[0])
combo["Electrical"].isnull().sum()
#looking at the remaining missing values
combo_null2 = combo.isnull().sum()/len(combo)*100
combo_null2[combo_null2 >0].sort_values(ascending =False)
#fill in GarageYrBlt with 0
combo["GarageYrBlt"] = combo["GarageYrBlt"].fillna(0)
combo["GarageYrBlt"].isnull().sum()
#fill up SalePrice with mode
combo["SalePrice"] = combo["SalePrice"].fillna(combo["SalePrice"].mode()[0])
#looking at the remaining missing values
combo_null2 = combo.isnull().sum()/len(combo)*100
combo_null2[combo_null2 >0].sort_values(ascending =False)
#check for numeric types
numeric_feature = combo.dtypes[combo.dtypes != "O"].index
numeric_feature
#check for categorical values
combo.describe(include=['O']).columns
#get all columns and dtypes
train.info(verbose=True)
#get all categorical columns
combo.describe(include=['O']).columns
#confirming all are categorical by value counts on multiple columns
combo.Fence.value_counts()
combo.MSZoning.value_counts()
combo.Street.value_counts()
#transform all numeric features to strings
#first call all numeric columns
numeric_feature
#change numeric to strings
all_features = numeric_feature
combo[all_features] = combo [all_features].astype(str)
combo.info()
cols = ("YrSold", "MSSubClass","OverallCond",
        "MoSold","MSZoning","Street","Alley","LotShape",
        "LandContour","LotConfig","SaleType","SaleCondition","MiscFeature","Fence",
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
       'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive','Id','LotFrontage', 'LotArea', 'OverallQual',
       'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',"SalePrice" 
     )
#Change all columns to numeric
from sklearn.preprocessing import LabelEncoder
for c in cols:
    le = LabelEncoder()
    le.fit(list(combo[c].values))
    combo[c] = le.transform(list(combo[c].values))

#confirmng cataegorical features have been changed to numeric 
combo["MoSold"].value_counts()
combo["MSZoning"].value_counts()
combo.info()
#drop utilities , RoofMatl and PoolQC
combo = combo.drop(["RoofMatl", "Utilities", "PoolQC"], axis = 1)
combo.info()
type(combo[numeric_feature])
combo[numeric_feature]
#looking for continous variables skweness example LotArea,

numeric_features_skew = combo[numeric_feature].skew().sort_values(ascending =False)
numeric_features_skew
# transfrom them using boxcox 

skewed_features = numeric_features_skew[(numeric_features_skew) > 0.5]
skewed_features
from scipy import stats
#transform the columns using Boxcox
skewed_features,fitted_lambda = stats.boxcox(skewed_features)
fitted_lambda
#Defining X
X = combo.loc[:,combo.columns != 'SalePrice']
X
#Define y
y = combo['SalePrice']
#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=42)
X_train.values
X_train.shape
#scaling X_test 
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_test_minmax = min_max_scaler.fit_transform(X_test.values)

X_test_minmax
X_test_minmax = min_max_scaler.fit_transform(X_test.values)
X_test_minmax
X_test_minmax.shape
X_train.shape
X_test.shape
y_train.shape
y_test.shape
print(y_train.shape)
print(y_test.shape)
print(X_train.shape)
print(X_test_minmax.shape)
#Fit model
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)
