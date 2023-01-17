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
import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from scipy import stats

from scipy.stats import norm, skew 

import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.float_format', lambda x: '%.3f' % x)

%matplotlib inline

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
whole_df = pd.concat([train, test])
whole_df
# Saving ID columns of train and test for the RMSE test

train_salePrice = train.loc[:, 'SalePrice']

train_ID = train.loc[:, 'Id']

test_ID = test.loc[:, 'Id']
train['SalePrice']
whole_df.set_index('Id', drop = True, inplace = True)
sns.boxplot(x = train_salePrice, y = whole_df.loc[train_ID, 'PoolQC'])
sns.distplot(train['SalePrice'])
plt.subplots(figsize = (13,9))

sns.heatmap(whole_df.corr())
plt.subplots(figsize = (10,7))

corrmatrix = whole_df.corr()

cols = corrmatrix.nlargest(15,'SalePrice').index

#cm = np.corrcoef(train[cols].values.T)

top_cols_df = whole_df.loc[:,cols]

sns.heatmap(top_cols_df.corr(), annot = True)
whole_df.drop(['GarageArea', '1stFlrSF', 'TotRmsAbvGrd', 'GarageYrBlt', 'Utilities','MiscVal'], axis = 1, inplace = True)
corrmatrix = train.corr()

columns = corrmatrix.nlargest(15,'SalePrice').index

top_cols_df = train.loc[:,columns]

sns.heatmap(top_cols_df.corr(), annot = True)
total = whole_df.isnull().sum().sort_values(ascending = False)

percent = (whole_df.isnull().sum()/whole_df.isnull().count()).sort_values()

missing = pd.concat([total, percent], axis =1, keys = ['Total', 'Percent'])

print('Total Missing Values')

miss_cols = missing.iloc[:20].index

missing.iloc[:20]

plt.subplots(figsize = (12,9))

sns.heatmap(whole_df[miss_cols].isnull())
whole_df['BsmtQual'].unique()
# Handle missing values for features where median/mean or most common value doesn't make sense



# Alley : data description says NA means "no alley access"

whole_df.loc[:, "Alley"] = whole_df.loc[:, "Alley"].fillna("None")



# BedroomAbvGr : NA most likely means 0

whole_df.loc[:, "BedroomAbvGr"] = whole_df.loc[:, "BedroomAbvGr"].fillna(0)



# BsmtQual etc : data description says NA for basement features is "no basement"

whole_df.loc[:, "BsmtQual"] = whole_df.loc[:, "BsmtQual"].fillna("None")

whole_df.loc[:, "BsmtCond"] = whole_df.loc[:, "BsmtCond"].fillna("None")

whole_df.loc[:, "BsmtExposure"] = whole_df.loc[:, "BsmtExposure"].fillna("None")

whole_df.loc[:, "BsmtFinType1"] = whole_df.loc[:, "BsmtFinType1"].fillna("None")

whole_df.loc[:, "BsmtFinType2"] = whole_df.loc[:, "BsmtFinType2"].fillna("None")

whole_df.loc[:, "BsmtFullBath"] = whole_df.loc[:, "BsmtFullBath"].fillna(0)

whole_df.loc[:, "BsmtHalfBath"] = whole_df.loc[:, "BsmtHalfBath"].fillna(0)

whole_df.loc[:, "BsmtUnfSF"] = whole_df.loc[:, "BsmtUnfSF"].fillna(0)



# CentralAir : NA most likely means No

whole_df.loc[:, "CentralAir"] = whole_df.loc[:, "CentralAir"].fillna("N")



# Condition : NA most likely means Normal

whole_df.loc[:, "Condition1"] = whole_df.loc[:, "Condition1"].fillna("Norm")

whole_df.loc[:, "Condition2"] = whole_df.loc[:, "Condition2"].fillna("Norm")



# EnclosedPorch : NA most likely means no enclosed porch

whole_df.loc[:, "EnclosedPorch"] = whole_df.loc[:, "EnclosedPorch"].fillna(0)



# External stuff : NA most likely means average

whole_df.loc[:, "ExterCond"] = whole_df.loc[:, "ExterCond"].fillna("TA")

whole_df.loc[:, "ExterQual"] = whole_df.loc[:, "ExterQual"].fillna("TA")



# Fence : data description says NA means "no fence"

whole_df.loc[:, "Fence"] = whole_df.loc[:, "Fence"].fillna("None")



# FireplaceQu : data description says NA means "no fireplace"

whole_df.loc[:, "FireplaceQu"] = whole_df.loc[:, "FireplaceQu"].fillna("None")

whole_df.loc[:, "Fireplaces"] = whole_df.loc[:, "Fireplaces"].fillna(0)



# Functional : data description says NA means typical

whole_df.loc[:, "Functional"] = whole_df.loc[:, "Functional"].fillna("Typ")



# GarageType etc : data description says NA for garage features is "no garage"

whole_df.loc[:, "GarageType"] = whole_df.loc[:, "GarageType"].fillna("None")

whole_df.loc[:, "GarageFinish"] = whole_df.loc[:, "GarageFinish"].fillna("None")

whole_df.loc[:, "GarageQual"] = whole_df.loc[:, "GarageQual"].fillna("None")

whole_df.loc[:, "GarageCond"] = whole_df.loc[:, "GarageCond"].fillna("None")

whole_df.loc[:, "GarageCars"] = whole_df.loc[:, "GarageCars"].fillna(0)



# HalfBath : NA most likely means no half baths above grade

whole_df.loc[:, "HalfBath"] = whole_df.loc[:, "HalfBath"].fillna(0)



# HeatingQC : NA most likely means typical

whole_df.loc[:, "HeatingQC"] = whole_df.loc[:, "HeatingQC"].fillna("TA")



# KitchenAbvGr : NA most likely means 0

whole_df.loc[:, "KitchenAbvGr"] = whole_df.loc[:, "KitchenAbvGr"].fillna(0)



# KitchenQual : NA most likely means typical

whole_df.loc[:, "KitchenQual"] = whole_df.loc[:, "KitchenQual"].fillna("TA")



# LotFrontage : NA most likely means no lot frontage

whole_df.loc[:, "LotFrontage"] = whole_df.loc[:, "LotFrontage"].fillna(0)



# LotShape : NA most likely means regular

whole_df.loc[:, "LotShape"] = whole_df.loc[:, "LotShape"].fillna("Reg")



# MasVnrType : NA most likely means no veneer

whole_df.loc[:, "MasVnrType"] = whole_df.loc[:, "MasVnrType"].fillna("None")

whole_df.loc[:, "MasVnrArea"] = whole_df.loc[:, "MasVnrArea"].fillna(0)



# MiscFeature : data description says NA means "no misc feature"

whole_df.loc[:, "MiscFeature"] = whole_df.loc[:, "MiscFeature"].fillna("None")



# MSZoning : NA replaced with the most frequently occuring value 'RL'

whole_df.loc[:, "MSZoning"] = whole_df.loc[:, "MSZoning"].fillna(whole_df.loc[:, "MSZoning"].mode()[0])



# OpenPorchSF : NA most likely means no open porch

whole_df.loc[:, "OpenPorchSF"] = whole_df.loc[:, "OpenPorchSF"].fillna(0)



# PavedDrive : NA most likely means not paved

whole_df.loc[:, "PavedDrive"] = whole_df.loc[:, "PavedDrive"].fillna("N")



# PoolQC : data description says NA means "no pool"

whole_df.loc[:, "PoolQC"] = whole_df.loc[:, "PoolQC"].fillna("None")

whole_df.loc[:, "PoolArea"] = whole_df.loc[:, "PoolArea"].fillna(0)



# SaleCondition : NA most likely means normal sale

whole_df.loc[:, "SaleCondition"] = whole_df.loc[:, "SaleCondition"].fillna("Normal")



# ScreenPorch : NA most likely means no screen porch

whole_df.loc[:, "ScreenPorch"] = whole_df.loc[:, "ScreenPorch"].fillna(0)



# WoodDeckSF : NA most likely means no wood deck

whole_df.loc[:, "WoodDeckSF"] = whole_df.loc[:, "WoodDeckSF"].fillna(0)
whole_df.drop('SalePrice', axis = 1, inplace = True)
total = whole_df.isnull().sum().sort_values(ascending = False)

percent = (whole_df.isnull().sum()/whole_df.isnull().count()).sort_values()

missing = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

print('Total Missing Values')

miss_cols = missing.iloc[:20].index

missing.iloc[:20]
for i in miss_cols[:7]:

    whole_df.loc[:, i] = whole_df.loc[:, i].fillna(whole_df.loc[:, i].mode()[0]) 
total = whole_df.isnull().sum().sort_values(ascending = False)

percent = (whole_df.isnull().sum()/whole_df.isnull().count()).sort_values()

missing = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

print('Total Missing Values')

missing.iloc[:20]

# Checking if there is any null values left in the data set

whole_df.isnull().values.any()
# Taking the Ordinal Features in one list

ord_fields = ['MSSubClass','ExterQual','LotShape','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','HeatingQC','Functional','FireplaceQu','KitchenQual', 'GarageFinish',

            'GarageQual','GarageCond','PoolQC','Fence']



# Ordering values of each column in 'ord_fields'

orders = [ 

    #MSSubClass 

    ['20','30','40','45','50','60','70','75','80','85', '90','120','150','160','180','190'],

    #ExterQual 

    ['Fa','TA','Gd','Ex'],

    #LotShape 

    ['Reg','IR1' ,'IR2','IR3'], 

    #BsmtQual 

    ['None','Fa','TA','Gd','Ex'],

    #BsmtCond

    ['None','Po','Fa','TA','Gd','Ex'],

    #BsmtExposure 

    ['None','No','Mn','Av','Gd'],

    #BsmtFinType1 

    ['None','Unf','LwQ', 'Rec','BLQ','ALQ' , 'GLQ' ],

    #BsmtFinType2 

    ['None','Unf','LwQ', 'Rec','BLQ','ALQ' , 'GLQ' ],

    #HeatingQC 

    ['Po','Fa','TA','Gd','Ex'], 

    #Functional 

    ['Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'], 

    #FireplaceQu 

    ['None','Po','Fa','TA','Gd','Ex'], 

    #KitchenQual 

    ['Fa','TA','Gd','Ex'],

    #GarageFinish 

    ['None','Unf','RFn','Fin'],

    #GarageQual 

    ['None','Po','Fa','TA','Gd','Ex'],

    #GarageCond

    ['None','Po','Fa','TA','Gd','Ex'],

    #PoolQC

    ['None','Fa','Gd','Ex'], 

    #Fence

    ['None','MnWw','GdWo','MnPrv','GdPrv']]



# Using 'OrdinalEncoder' from the sklearn preprocessing package to convert from Categorical to



from sklearn.preprocessing import OrdinalEncoder

for i in range(len(orders)): 

    ord_en = OrdinalEncoder(categories = {0 : orders[i]}) 

    whole_df.loc[:, ord_fields[i]] = ord_en.fit_transform(whole_df.loc[:, ord_fields[i]].values.reshape(-1,1))
categorical = whole_df.select_dtypes(include = ["object"]).columns

numerical = whole_df.select_dtypes(exclude = ["object"]).columns



print("Numerical features : " + str(len(numerical)))

print("Categorical features : " + str(len(categorical)))

num_df = whole_df[numerical]

cat_df = whole_df[categorical]
# Number of unique values in the Categorical features DataFrame 'cat_df' per column

cat_df.nunique()
cat_cols = cat_df.columns

from sklearn.preprocessing import LabelEncoder

for one in cat_cols:

    lab_enc = LabelEncoder() 

    lab_enc.fit(list(whole_df[one].values)) 

    whole_df[one] = lab_enc.transform(list(whole_df[one].values))
# Checking to see if there are any categorical columns left in 'whole_df'

categorical = whole_df.select_dtypes(include = ["object"]).columns

numerical = whole_df.select_dtypes(exclude = ["object"]).columns



print("Numerical features : " + str(len(numerical)))

print("Categorical features : " + str(len(categorical)))

num_df = whole_df[numerical]

cat_df = whole_df[categorical]
# Creat a histogram and normal probability plot for 'Sale Price'

sns.distplot(train_salePrice, fit = norm)

fig = plt.figure()

res = stats.probplot(train_salePrice, plot = plt)
# Applying the Box-Cox log transformation to the target feature with lambda = 0.15



lam = 0.15

from scipy.special import boxcox1p

train_log_SP = boxcox1p(train_salePrice, lam)
# Plotting the transformed histogram and normal probability plot of 'Sale Price'

sns.distplot(train_log_SP, fit = norm)

fig = plt.figure()

res = stats.probplot(train_log_SP, plot = plt)
# Creating a table of variables and their relative skewness

skewed = whole_df[numerical].apply(lambda x: skew(x)).sort_values(ascending = False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed})

skewness.head(20)
# We transform features with skewness larger than 0.75 to ensure the linearity of features.

skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index



for feat in skewed_features:

    whole_df[feat] = boxcox1p(whole_df[feat], lam)
ntrain = len(train.index)

ntrain
train = whole_df.iloc[ : ntrain]

test = whole_df.iloc[ntrain : ]



y = train_log_SP

X = train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
sns.set_style(style = 'whitegrid')

sns.jointplot(x = y_test, y = predictions)

#plt.scatter(y_test, predictions)

plt.xlabel('Y label')

plt.ylabel('Predicted Y')
from sklearn import metrics

print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
metrics.explained_variance_score(y_test,predictions)
lm_test = LinearRegression()
lm_test.fit(X,y)
test_predictions = lm_test.predict(test)
len(test_predictions)
from scipy.special import boxcox1p, inv_boxcox1p

test_predictions = inv_boxcox1p(test_predictions, lam)
sns.jointplot(x = train_ID, y = train_salePrice)

sns.jointplot(x = test_ID, y = test_predictions)
solution = pd.DataFrame({"id": test_ID, "SalePrice": test_predictions})

solution.to_csv("House_Solutions.csv", index = False)