# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, probplot, linregress
import math

from sklearn.preprocessing import StandardScaler,RobustScaler,LabelEncoder,PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import xgboost

import warnings
warnings.filterwarnings("ignore")
# INITIALIZE DATA
df_TEST  = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df_TRAIN = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

# SHAPE OF DATA
print("The shape of training data is {} collums and {} rows." .format(df_TRAIN.shape[1], df_TRAIN.shape[0]))
print("The shape of testing data is {} collums and {} rows." .format(df_TEST.shape[1], df_TEST.shape[0]))

df_TRAIN.sample(3)
# To clean all the data we must first join TRAIN and TEST for the next steps must be done in the whole set
df_ALLDATA = pd.concat([df_TRAIN, df_TEST],axis=0,sort=False)

# Percentage of missing values per feature
null_per = lambda x: ((x.isnull()).sum()/len(x) * 100).sort_values(ascending = False)
df_nullDATA = null_per(df_ALLDATA).head(20)
print(df_nullDATA)

print('\nFeatures that have more than 20% of missing values')
print("="*50)
print(df_nullDATA[df_nullDATA.values > 20].index.tolist())

# Let's remove features with >20% of missing values, because they will not be valuable for analysis
df_ALLDATA = df_ALLDATA.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1)
# We have to fill the missing values. First, let's see how much (numerically) missing values there are per columns:
print("Missing values per collumn:")
print("="*50)

print((df_ALLDATA.isnull()).sum().sort_values(ascending = False).iloc[:31])
# Obviously all the 'SalePrice' missing values comes from the TEST dataset, so just ignore it.
# We only have to fill the missing values from 'LotFrontage' and beyond

# Strategy for filling missing values:
# GarageYrBlt --> Same year the house was built
# GarageFinish --> Change NA for "NoGarage"
# GarageQual --> Change NA for "NoGarage"
# GarageCond --> Change NA for "NoGarage"
# GarageType --> Change NA for "NoGarage"
# BsmtExposure --> Change NA for "NoBasement"
# BsmtFinType1 --> Change NA for "NoBasement"
# BsmtFinType2 --> Change NA for "NoBasement"
# BsmtCond --> Change NA for "NoBasement"
# All the others --> Mode/Mean

# Another possibilty is to fill the missing values using KNN-Imputer. Later I will see if it makes any visible difference
# Besides that, I'm going to create 2 new columns to adress the fact that not having a garage or basement affects the price
df_ALLDATA.loc[df_ALLDATA['GarageYrBlt'].isnull(), 'HasGarage'] = 0
df_ALLDATA.loc[df_ALLDATA['GarageYrBlt'].notnull(), 'HasGarage'] = 1
df_ALLDATA.loc[df_ALLDATA['BsmtExposure'].isnull(), 'HasBasement'] = 0
df_ALLDATA.loc[df_ALLDATA['BsmtExposure'].notnull(), 'HasBasement'] = 1

df_ALLDATA['GarageYrBlt'] = df_ALLDATA['GarageYrBlt'].fillna(df_ALLDATA['YearBuilt'])
df_ALLDATA['GarageFinish'] = df_ALLDATA['GarageFinish'].fillna('NoGarage')
df_ALLDATA['GarageQual'] = df_ALLDATA['GarageQual'].fillna('NoGarage')
df_ALLDATA['GarageCond'] = df_ALLDATA['GarageCond'].fillna('NoGarage')
df_ALLDATA['GarageType'] = df_ALLDATA['GarageType'].fillna('NoGarage')
df_ALLDATA['BsmtExposure'] = df_ALLDATA['BsmtExposure'].fillna('NoBasement')
df_ALLDATA['BsmtFinType1'] = df_ALLDATA['BsmtFinType1'].fillna('NoBasement')
df_ALLDATA['BsmtFinType2'] = df_ALLDATA['BsmtFinType2'].fillna('NoBasement')
df_ALLDATA['BsmtQual'] = df_ALLDATA['BsmtQual'].fillna('NoBasement')
df_ALLDATA['BsmtCond'] = df_ALLDATA['BsmtCond'].fillna('NoBasement')
df_ALLDATA['MasVnrType'] = df_ALLDATA['MasVnrType'].fillna(df_ALLDATA['MasVnrType'].mode()[0])
df_ALLDATA['MasVnrArea'] = df_ALLDATA['MasVnrArea'].fillna(df_ALLDATA['MasVnrArea'].mean())
df_ALLDATA['MSZoning'] = df_ALLDATA['MSZoning'].fillna(df_ALLDATA['MSZoning'].mode()[0])
df_ALLDATA['BsmtHalfBath'] = df_ALLDATA['BsmtHalfBath'].fillna(df_ALLDATA['BsmtHalfBath'].mode()[0])
df_ALLDATA['Utilities'] = df_ALLDATA['Utilities'].fillna(df_ALLDATA['Utilities'].mode()[0])
df_ALLDATA['Functional'] = df_ALLDATA['Functional'].fillna(df_ALLDATA['Functional'].mode()[0])
df_ALLDATA['BsmtFullBath'] = df_ALLDATA['BsmtFullBath'].fillna(df_ALLDATA['BsmtFullBath'].mode()[0])
df_ALLDATA['BsmtFinSF1'] = df_ALLDATA['BsmtFinSF1'].fillna(df_ALLDATA['BsmtFinSF1'].mean())
df_ALLDATA['Exterior2nd'] = df_ALLDATA['Exterior2nd'].fillna(df_ALLDATA['Exterior2nd'].mode()[0])
df_ALLDATA['Exterior1st'] = df_ALLDATA['Exterior1st'].fillna(df_ALLDATA['Exterior1st'].mode()[0])
df_ALLDATA['BsmtFinSF2'] = df_ALLDATA['BsmtFinSF2'].fillna(df_ALLDATA['BsmtFinSF2'].mode()[0])
df_ALLDATA['BsmtUnfSF'] = df_ALLDATA['BsmtUnfSF'].fillna(df_ALLDATA['BsmtUnfSF'].mean())
df_ALLDATA['Electrical'] = df_ALLDATA['Electrical'].fillna(df_ALLDATA['Electrical'].mode()[0])
df_ALLDATA['TotalBsmtSF'] = df_ALLDATA['TotalBsmtSF'].fillna(df_ALLDATA['TotalBsmtSF'].mean())
df_ALLDATA['KitchenQual'] = df_ALLDATA['KitchenQual'].fillna(df_ALLDATA['KitchenQual'].mode()[0])
df_ALLDATA['GarageArea'] = df_ALLDATA['GarageArea'].fillna(df_ALLDATA['GarageArea'].mean())
df_ALLDATA['GarageCars'] = df_ALLDATA['GarageCars'].fillna(df_ALLDATA['GarageCars'].mode()[0])
df_ALLDATA['HalfBath'] = df_ALLDATA['HalfBath'].fillna(df_ALLDATA['HalfBath'].mode()[0])
df_ALLDATA['LotFrontage'] = df_ALLDATA['LotFrontage'].fillna(df_ALLDATA['LotFrontage'].mean())
df_ALLDATA['SaleType'] = df_ALLDATA['SaleType'].fillna(df_ALLDATA['SaleType'].mode()[0])

# Let's see if still there are missing values
print("Current missing values per collumn (Except 'SalePrice'):")
print("="*60)

print((df_ALLDATA.isnull()).sum().sort_values(ascending = False).iloc[:31])
# There are some features that are 'object'-class and we need to make than numerical. Let's encode them:

# Collumns that are not int64 after encoding
print("Columns that have 'object' values")
print("="*50)
print(df_ALLDATA.dtypes[df_ALLDATA.dtypes.values == 'O'].index.tolist())

lb_make = LabelEncoder()
df_ALLDATA['MSZoning'] = lb_make.fit_transform(df_ALLDATA["MSZoning"])
df_ALLDATA['Street'] = lb_make.fit_transform(df_ALLDATA["Street"])
df_ALLDATA['LotShape'] = lb_make.fit_transform(df_ALLDATA["LotShape"])
df_ALLDATA['LandContour'] = lb_make.fit_transform(df_ALLDATA["LandContour"])
df_ALLDATA['Utilities'] = lb_make.fit_transform(df_ALLDATA["Utilities"])
df_ALLDATA['LotConfig'] = lb_make.fit_transform(df_ALLDATA["LotConfig"])
df_ALLDATA['LandSlope'] = lb_make.fit_transform(df_ALLDATA["LandSlope"])
df_ALLDATA['Neighborhood'] = lb_make.fit_transform(df_ALLDATA["Neighborhood"])
df_ALLDATA['Condition1'] = lb_make.fit_transform(df_ALLDATA["Condition1"])
df_ALLDATA['Condition2'] = lb_make.fit_transform(df_ALLDATA["Condition2"])
df_ALLDATA['BldgType'] = lb_make.fit_transform(df_ALLDATA["BldgType"])
df_ALLDATA['HouseStyle'] = lb_make.fit_transform(df_ALLDATA["HouseStyle"])
df_ALLDATA['RoofStyle'] = lb_make.fit_transform(df_ALLDATA["RoofStyle"])
df_ALLDATA['RoofMatl'] = lb_make.fit_transform(df_ALLDATA["RoofMatl"])
df_ALLDATA['Exterior1st'] = lb_make.fit_transform(df_ALLDATA["Exterior1st"])
df_ALLDATA['Exterior2nd'] = lb_make.fit_transform(df_ALLDATA["Exterior2nd"])
df_ALLDATA['MasVnrType'] = lb_make.fit_transform(df_ALLDATA["MasVnrType"])
df_ALLDATA['ExterQual'] = lb_make.fit_transform(df_ALLDATA["ExterQual"])
df_ALLDATA['ExterCond'] = lb_make.fit_transform(df_ALLDATA["ExterCond"])
df_ALLDATA['Foundation'] = lb_make.fit_transform(df_ALLDATA["Foundation"])
df_ALLDATA['BsmtQual'] = lb_make.fit_transform(df_ALLDATA["BsmtQual"])
df_ALLDATA['BsmtCond'] = lb_make.fit_transform(df_ALLDATA["BsmtCond"])
df_ALLDATA['BsmtExposure'] = lb_make.fit_transform(df_ALLDATA["BsmtExposure"])
df_ALLDATA['BsmtFinType1'] = lb_make.fit_transform(df_ALLDATA["BsmtFinType1"])
df_ALLDATA['BsmtFinType2'] = lb_make.fit_transform(df_ALLDATA["BsmtFinType2"])
df_ALLDATA['Heating'] = lb_make.fit_transform(df_ALLDATA["Heating"])
df_ALLDATA['HeatingQC'] = lb_make.fit_transform(df_ALLDATA["HeatingQC"])
df_ALLDATA['CentralAir'] = lb_make.fit_transform(df_ALLDATA["CentralAir"])
df_ALLDATA['Electrical'] = lb_make.fit_transform(df_ALLDATA["Electrical"])
df_ALLDATA['KitchenQual'] = lb_make.fit_transform(df_ALLDATA["KitchenQual"])
df_ALLDATA['Functional'] = lb_make.fit_transform(df_ALLDATA["Functional"])
df_ALLDATA['GarageType'] = lb_make.fit_transform(df_ALLDATA["GarageType"])
df_ALLDATA['GarageFinish'] = lb_make.fit_transform(df_ALLDATA["GarageFinish"])
df_ALLDATA['GarageQual'] = lb_make.fit_transform(df_ALLDATA["GarageQual"])
df_ALLDATA['GarageCond'] = lb_make.fit_transform(df_ALLDATA["GarageCond"])
df_ALLDATA['PavedDrive'] = lb_make.fit_transform(df_ALLDATA["PavedDrive"])
df_ALLDATA['SaleType'] = lb_make.fit_transform(df_ALLDATA["SaleType"])
df_ALLDATA['SaleCondition'] = lb_make.fit_transform(df_ALLDATA["SaleCondition"])


# Collumns that are not int64 after encoding
print("\nColumns that have 'object' values after encoding")
print("="*50)
print(df_ALLDATA.dtypes[df_ALLDATA.dtypes.values == 'O'].index.tolist())
# Now that we have cleaned the data it can be broken back into TRAIN and TEST
df_TRAIN = df_ALLDATA[:1460]

df_TEST_1 = df_ALLDATA[1460:]
df_TEST = df_TEST_1.drop('SalePrice', axis = 1)

# SHAPE OF DATA
print("The shape of training data is {} collums and {} rows." .format(df_TRAIN.shape[1], df_TRAIN.shape[0]))
print("The shape of testing data is {} collums and {} rows." .format(df_TEST.shape[1], df_TEST.shape[0]))
# Features correlation with 'SalePrice'
TRAIN_corr = (df_TRAIN.corr()['SalePrice']).sort_values(ascending = False)
print(TRAIN_corr)

# Features that have correlation greater than 0.56 with 'SalePrice'
print("\nFeatures that have correlation greater than 0.6 with 'SalePrice'")
print("="*80)
HIGH_features = TRAIN_corr[abs(TRAIN_corr.values) > 0.6].index.tolist()
print(HIGH_features) # We can fine-tune this value to achieve better results...i guess
# We can improve our data if we remove outliners from the most prominent features, so let's search for them.

##########################################################
### SALE PRICE ###
X = df_TRAIN['SalePrice']
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('Sale Price')
ax[0].grid()
probplot(X, fit=norm, plot=ax[1])
ax[1].title.set_text('Sale Price')
ax[1].grid()

X = np.log(df_TRAIN['SalePrice'])
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm, color='green')
ax[0].title.set_text('Probability Mass Function')
ax[0].grid()
probplot(X, fit=norm, plot=ax[1])
ax[1].get_lines()[0].set_markerfacecolor('g')
ax[1].get_lines()[0].set_markeredgecolor('g')
ax[1].title.set_text('Q-Q Plot')
ax[1].grid()
##########################################################
### OVERALL QUALITY ###
X = df_TRAIN['OverallQual']
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('OverallQual')
ax[0].grid()
probplot(X, fit=norm, plot=ax[1])
ax[1].title.set_text('OverallQual')
ax[1].grid()

X = np.log(df_TRAIN['OverallQual'])
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('Probability Mass Function')
ax[0].grid()
probplot(X, fit=norm, plot=ax[1])
ax[1].get_lines()[0].set_markerfacecolor('g')
ax[1].get_lines()[0].set_markeredgecolor('g')
ax[1].title.set_text('Q-Q Plot')
ax[1].grid()
##########################################################
### GROUND LIVING AREA ###
X = df_TRAIN['GrLivArea']
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('GrLivArea')
ax[0].grid()
probplot(X, fit=norm, plot=ax[1])
ax[1].title.set_text('GrLivArea')
ax[1].grid()

X = np.log(df_TRAIN['GrLivArea'])
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('Probability Mass Function')
ax[0].grid()

probplot(X, fit=norm, plot=ax[1])
ax[1].get_lines()[0].set_markerfacecolor('g')
ax[1].get_lines()[0].set_markeredgecolor('g')
ax[1].title.set_text('Q-Q Plot')
ax[1].grid()
##########################################################
### GARAGE CARS ###
X = df_TRAIN['GarageCars']
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('GarageCars')
ax[0].grid()
probplot(X, fit=norm, plot=ax[1])
ax[1].title.set_text('GarageCars')
ax[1].grid()

X = np.log1p(df_TRAIN['GarageCars'])
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('Probability Mass Function')
ax[0].grid()

probplot(X, fit=norm, plot=ax[1])
ax[1].get_lines()[0].set_markerfacecolor('g')
ax[1].get_lines()[0].set_markeredgecolor('g')
ax[1].title.set_text('Q-Q Plot')
ax[1].grid()
##########################################################
### GARAGE AREA ###
X = df_TRAIN['GarageArea']
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('GarageArea')
ax[0].grid()
probplot(X, fit=norm, plot=ax[1])
ax[1].title.set_text('GarageArea')
ax[1].grid()

X = np.log1p(df_TRAIN['GarageArea'])
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('Probability Mass Function')
ax[0].grid()

probplot(X, fit=norm, plot=ax[1])
ax[1].get_lines()[0].set_markerfacecolor('g')
ax[1].get_lines()[0].set_markeredgecolor('g')
ax[1].title.set_text('Q-Q Plot')
ax[1].grid()
##########################################################
### TOTAL BASEMENT SURFACE ###
X = df_TRAIN['TotalBsmtSF']
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('TotalBsmtSF')
ax[0].grid()
probplot(X, fit=norm, plot=ax[1])
ax[1].title.set_text('TotalBsmtSF')
ax[1].grid()

X = np.log1p(df_TRAIN['TotalBsmtSF'])
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('Probability Mass Function')
ax[0].grid()

probplot(X, fit=norm, plot=ax[1])
ax[1].get_lines()[0].set_markerfacecolor('g')
ax[1].get_lines()[0].set_markeredgecolor('g')
ax[1].title.set_text('Q-Q Plot')
ax[1].grid()
# 1. Since the created feature "HasBasement" hasn't shown a high correlation with 'SalePrice' 
# I think it's safe to remove rows in which 'TotalBsmtSF' is zero and also greater than 3000.
# Number of rows
print("Number of rows before removing outliner")
print('='*50)
print(df_TRAIN.shape[0])

# Let's drop the outliners
df_TRAIN = df_TRAIN[df_TRAIN['TotalBsmtSF'] != 0]
df_TRAIN = df_TRAIN[df_TRAIN['TotalBsmtSF'] < 3000]

# Number of rows now
print("\nNumber of rows after removing outliner")
print('='*50)
print(df_TRAIN.shape[0])
##########################################################
### 1st FLOOR SURFACE ###
X = df_TRAIN['1stFlrSF']
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('1stFlrSF')
ax[0].grid()
probplot(X, fit=norm, plot=ax[1])
ax[1].title.set_text('1stFlrSF')
ax[1].grid()

X = np.log(df_TRAIN['1stFlrSF'])
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('Probability Mass Function')
ax[0].grid()

probplot(X, fit=norm, plot=ax[1])
ax[1].get_lines()[0].set_markerfacecolor('g')
ax[1].get_lines()[0].set_markeredgecolor('g')
ax[1].title.set_text('Q-Q Plot')
ax[1].grid()
# We can see outliner where values of 1stFlrSF is greater than 3500-ish
# Number of rows
print("Number of rows before removing outliner")
print('='*50)
print(df_TRAIN.shape[0])

# Let's drop the outliners
df_TRAIN = df_TRAIN[df_TRAIN['1stFlrSF'] < 3500]

# Number of rows now
print("\nNumber of rows after removing outliner")
print('='*50)
print(df_TRAIN.shape[0])
##########################################################
### EXTERNAL QUALITY ###
X = df_TRAIN['ExterQual']
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('ExterQual')
ax[0].grid()
probplot(X, fit=norm, plot=ax[1])
ax[1].title.set_text('ExterQual')
ax[1].grid()

X = np.log1p(df_TRAIN['ExterQual'])
fig, ax = plt.subplots(ncols=2,figsize=(16,6))
sns.distplot(X,ax=ax[0], fit=norm)
ax[0].title.set_text('Probability Mass Function')
ax[0].grid()

probplot(X, fit=norm, plot=ax[1])
ax[1].get_lines()[0].set_markerfacecolor('g')
ax[1].get_lines()[0].set_markeredgecolor('g')
ax[1].title.set_text('Q-Q Plot')
ax[1].grid()
# Let's see how the correlation matrix changed after that

TRAIN_corr = (df_TRAIN.corr()['SalePrice']).sort_values(ascending = False)
print(TRAIN_corr)

# Features that have correlation greater than 0.56 with 'SalePrice'
print("\nFeatures that have correlation greater than 0.6 with 'SalePrice'")
print("="*80)
HIGH_features = TRAIN_corr[abs(TRAIN_corr.values) > 0.6].index.tolist()
print(HIGH_features) # We can fine-tune this value to achieve better results...i guess
df_TRAIN.shape
Y = df_TRAIN["SalePrice"]
test_id = df_TEST["Id"]

del df_TRAIN["Id"]
del df_TEST["Id"]
del df_TRAIN["SalePrice"]

final = pd.concat([df_TRAIN,df_TEST],axis=0)

# Create the Scaler object
scaler = StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(final)
final = pd.DataFrame(scaled_df)

df_train=final.iloc[:1418,:]
df_test=final.iloc[1418:,:]
df_test.shape

X = df_TRAIN

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)
#############################
# LINEAR REGRESSION #

linear_reg=LinearRegression()
linear_reg.fit(X_train,Y_train)

print("R-Squared Value for Training Set: {:.3f}".format(linear_reg.score(X_train,Y_train)))
print("R-Squared Value for Test Set: {:.3f}".format(linear_reg.score(X_test,Y_test)))
#############################
# RANDOM FOREST #
R_forest=RandomForestRegressor()
R_forest.fit(X_train,Y_train)

print("R-Squared Value for Training Set: {:.3f}".format(R_forest.score(X_train,Y_train)))
print("R-Squared Value for Test Set: {:.3f}".format(R_forest.score(X_test,Y_test)))
y_pred_rforest = R_forest.predict(df_TEST)
##############################
# XG BOOST #

regressor=xgboost.XGBRegressor()

regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
       importance_type='gain', interaction_constraints='',
       learning_rate=0.1, max_delta_step=0, max_depth=2,
       min_child_weight=1, monotone_constraints='()',
       n_estimators=900, n_jobs=0, num_parallel_tree=1,
       objective='reg:squarederror', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
       validate_parameters=1, verbosity=None)

regressor.fit(X_train,Y_train)

print("R-Squared Value for Training Set: {:.3f}".format(regressor.score(X_train,Y_train)))
print("R-Squared Value for Test Set: {:.3f}".format(regressor.score(X_test,Y_test)))
y_pred_xgb = regressor.predict(df_TEST)
########################################
# REGULARIZATION #

y_pred_01 = np.floor((y_pred_rforest + y_pred_xgb)/2)
pred_df_01 = pd.DataFrame(y_pred_01, columns=['SalePrice'])
test_id_df = pd.DataFrame(test_id, columns=['Id'])

submission_01 = pd.concat([test_id_df, pred_df_01], axis=1)
submission_01.to_csv(r'submission.csv', index=False)
submission_01.head()