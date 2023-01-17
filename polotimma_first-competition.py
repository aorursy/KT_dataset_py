import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from scipy import stats
from scipy.stats import norm, skew #for some statistics

%matplotlib inline
# Import dataset
train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_data.shape, test_data.shape
# Preview train dataset
train_data.head()

# Preview test dataset
test_data.head()
# Remove IDs from train and test set, not useful for model
train_ID = train_data['Id']
test_ID = test_data['Id']
train_data.drop(['Id'], axis=1, inplace=True)
test_data.drop(['Id'], axis=1, inplace=True)
train_data.shape, test_data.shape
#descriptive statistics summary
train_data['SalePrice'].describe()
# Skewness and Kurtosis
# Skewness - measure of the lack of symmetry in the data
# Kurtosis - shows whether there is many outliers in the data
print("Skewness: %f" % train_data['SalePrice'].skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())
# Distribution plot
f, ax = plt.subplots(figsize=(10,5))
sns.distplot(train_data['SalePrice'])
ax.set(xlabel="SalePrice")
ax.set(ylabel="Frequency")
ax.set(title="SalePrice Distribution")

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)

plt.show()
# Look at data correlation using heatmap
corr = train_data.corr()
plt.subplots(figsize=(15,15))
sns.heatmap(corr, fmt='.1f', cmap="Blues", square=True)
plt.show()
# Checking missing values in train dataset

train_data.isnull().sum().sort_values(ascending=False)
# Get features with missing values
features_with_na = [feature for feature in train_data.columns if train_data[feature].isnull().sum() > 0]

for feature in features_with_na:
    data = train_data.copy()
    #Create a variable that indicates 1 if the values is missing and 0 otherwise.
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    # Plot bar graph of median SalesPrice for values missing or present in train dataset
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.title(feature)
    plt.show()
#below these are about having the feature or not, so we add None to missing values
train_data["PoolQC"] = train_data["PoolQC"].fillna("None")
train_data["MiscFeature"] = train_data["MiscFeature"].fillna("None")
train_data["Alley"] = train_data["Alley"].fillna("None")
train_data["Fence"] = train_data["Fence"].fillna("None")
train_data["FireplaceQu"] = train_data["FireplaceQu"].fillna("None")


test_data["PoolQC"] = test_data["PoolQC"].fillna("None")
test_data["MiscFeature"] = test_data["MiscFeature"].fillna("None")
test_data["Alley"] = test_data["Alley"].fillna("None")
test_data["Fence"] = test_data["Fence"].fillna("None")
test_data["FireplaceQu"] = test_data["FireplaceQu"].fillna("None")


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
train_data["LotFrontage"] = train_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

test_data["LotFrontage"] = test_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

#below these are about having Garage or not
train_data["GarageType"] = train_data["GarageType"].fillna("None")
train_data["GarageFinish"] = train_data["GarageFinish"].fillna("None")
train_data["GarageQual"] = train_data["GarageQual"].fillna("None")
train_data["GarageCond"] = train_data["GarageCond"].fillna("None")
train_data["GarageYrBlt"] = train_data["GarageYrBlt"].fillna(0)
train_data["GarageArea"] = train_data["GarageArea"].fillna(0)
train_data["GarageCars"] = train_data["GarageCars"].fillna(0)

test_data["GarageType"] = test_data["GarageType"].fillna("None")
test_data["GarageFinish"] = test_data["GarageFinish"].fillna("None")
test_data["GarageQual"] = test_data["GarageQual"].fillna("None")
test_data["GarageCond"] = test_data["GarageCond"].fillna("None")
test_data["GarageYrBlt"] = test_data["GarageYrBlt"].fillna(0)
test_data["GarageArea"] = test_data["GarageArea"].fillna(0)
test_data["GarageCars"] = test_data["GarageCars"].fillna(0)

#below these are about having Basement or not
train_data["BsmtQual"] = train_data["BsmtQual"].fillna("None")
train_data["BsmtCond"] = train_data["BsmtCond"].fillna("None")
train_data["BsmtExposure"] = train_data["BsmtExposure"].fillna("None")
train_data["BsmtFinType1"] = train_data["BsmtFinType1"].fillna("None")
train_data["BsmtFinType2"] = train_data["BsmtFinType2"].fillna("None")

test_data["BsmtQual"] = test_data["BsmtQual"].fillna("None")
test_data["BsmtCond"] = test_data["BsmtCond"].fillna("None")
test_data["BsmtExposure"] = test_data["BsmtExposure"].fillna("None")
test_data["BsmtFinType1"] = test_data["BsmtFinType1"].fillna("None")
test_data["BsmtFinType2"] = test_data["BsmtFinType2"].fillna("None")

#below these are about having MasVnr or not
train_data["MasVnrType"] = train_data["MasVnrType"].fillna("None")
train_data["MasVnrArea"] = train_data["MasVnrArea"].fillna(0)

test_data["MasVnrType"] = test_data["MasVnrType"].fillna("None")
test_data["MasVnrArea"] = test_data["MasVnrArea"].fillna(0)

#below these are about having MSSubClass or not
train_data['MSSubClass'] = train_data['MSSubClass'].fillna("None")
test_data['MSSubClass'] = test_data['MSSubClass'].fillna("None")

#MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
train_data['MSZoning'] = train_data['MSZoning'].fillna("RL")
test_data['MSZoning'] = test_data['MSZoning'].fillna("RL")


#Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, 
#Therefore, this feature won't help in predictive modelling. We can then safely remove it.
train_data = train_data.drop(['Utilities'], axis=1)
test_data = test_data.drop(['Utilities'], axis=1)

#Functional : data description says NA means NonFunctional
train_data["Functional"] = train_data["Functional"].fillna("NonFunctional")
test_data["Functional"] = test_data["Functional"].fillna("NonFunctional")

#Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
train_data['Electrical'] = train_data['Electrical'].fillna("SBrkr")
test_data['Electrical'] = test_data['Electrical'].fillna("SBrkr")

#KitchenQual: It has one NA value,  Since this feature has mostly 'TA', we can set that for the missing value.
train_data['KitchenQual'] = train_data['KitchenQual'].fillna("TA")
test_data['KitchenQual'] = test_data['KitchenQual'].fillna("TA")

#Exterior1st and Exterior2nd :Both Exterior 1 & 2 have only one missing value.  Since this feature has mostly 'VinylSd', we can set that for the missing value.
train_data['Exterior1st'] = train_data['Exterior1st'].fillna('VinylSd')
train_data['Exterior2nd'] = train_data['Exterior2nd'].fillna('VinylSd')

test_data['Exterior1st'] = test_data['Exterior1st'].fillna('VinylSd')
test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna('VinylSd')

#SaleType : Since this feature has mostly 'WD', we can set that for the missing value.
train_data['SaleType'] = train_data['SaleType'].fillna("WD")
test_data['SaleType'] = test_data['SaleType'].fillna("WD")


train_data.isnull().sum().sort_values(ascending=False)
test_data.isnull().sum().sort_values(ascending=False)
# Get features with missing values in test data
features_with_na = [feature for feature in test_data.columns if test_data[feature].isnull().sum() > 0]
features_with_na
test_data['BsmtFinSF1'] = test_data['BsmtFinSF1'].fillna(0)
test_data['BsmtFinSF2'] = test_data['BsmtFinSF2'].fillna(0)
test_data['BsmtUnfSF'] = test_data['BsmtUnfSF'].fillna(0)
test_data['BsmtFullBath'] = test_data['BsmtFullBath'].fillna(0)
test_data['BsmtHalfBath'] = test_data['BsmtHalfBath'].fillna(0)
test_data['TotalBsmtSF'] = test_data['TotalBsmtSF'].fillna(0)
test_data.isnull().sum().sort_values(ascending=False)
numerical_features = [feature for feature in train_data.columns if train_data[feature].dtype != 'O']
print("Number of numerical features: ", len(numerical_features))
train_data[numerical_features].head()
date_features = [feature for feature in numerical_features if 'Year' in feature or 'Yr' in feature]
print("Number of temporal features: ", len(date_features))
train_data[date_features].head()
features_with_unique = [feature for feature in numerical_features if len(train_data[feature].unique()) > 0 and feature not in date_features ]
print("Length of discrete features: ", len(features_with_unique))
for feature in features_with_unique:
    print("Feature {} has {} unique values".format(feature, len(train_data[feature].unique())))

features_with_unique = [feature for feature in features_with_unique if len(train_data[feature].unique()) <= 30]
print("Length of discrete features: ", len(features_with_unique))
for feature in features_with_unique:
    print("Feature {} has {} unique values".format(feature, len(train_data[feature].unique())))
for feature in features_with_unique:
    data = train_data.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.title(feature)
    plt.show()
features_with_continuous = [feature for feature in numerical_features if feature not in features_with_unique + date_features ]
print("Length of continuous features: ", len(features_with_continuous))
train_data[features_with_continuous].head()
for feature in features_with_continuous:
    data = train_data.copy()
    
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()
for feature in features_with_continuous:
    data = train_data.copy()
    plt.scatter(data[feature], data['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.title(feature)
    plt.show()
train_data = train_data.drop(train_data[train_data['LotFrontage'] > 200].index)
train_data = train_data.drop(train_data[train_data['LotArea'] > 100000].index)
train_data = train_data.drop(train_data[train_data['MasVnrArea'] > 1200].index)
train_data = train_data.drop(train_data[train_data['BsmtFinSF1'] > 3000].index)
train_data = train_data.drop(train_data[train_data['BsmtFinSF2'] > 1200].index)
train_data = train_data.drop(train_data[train_data['TotalBsmtSF'] > 4000].index)
train_data = train_data.drop(train_data[train_data['1stFlrSF'] > 3500].index)
train_data = train_data.drop(train_data[train_data['GrLivArea'] > 4000].index)
train_data = train_data.drop(train_data[train_data['WoodDeckSF'] > 800].index)
train_data = train_data.drop(train_data[train_data['OpenPorchSF'] > 450].index)
train_data = train_data.drop(train_data[train_data['EnclosedPorch'] > 400].index)

for feature in features_with_continuous:
    data = train_data.copy()
    plt.scatter(data[feature], data['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.title(feature)
    plt.show()
for feature in features_with_continuous:
    data = train_data.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('Sale Price')
        plt.title(feature)
        plt.show()
for feature in features_with_continuous:
    data = train_data.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.title(feature)
        plt.show()
features_with_categorial = [feature for feature in train_data.columns if train_data[feature].dtypes == 'O']
train_data[features_with_categorial].head()
for feature in features_with_categorial:
    print("Feature {} has {} unique values".format(feature, len(train_data[feature].unique())))
for feature in features_with_categorial:
    data = train_data.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.title(feature)
    plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train_data["SalePrice"] = np.log1p(train_data["SalePrice"])

#Check the new distribution 
sns.distplot(train_data['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_data['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
plt.show()
num_non_zero_skewed_features_train_set = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

for feature in num_non_zero_skewed_features_train_set:
    train_data[feature] = np.log(train_data[feature])
    test_data[feature] = np.log(test_data[feature])
    
#Check the new distribution 
sns.distplot(train_data['GrLivArea'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_data['GrLivArea'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_data['GrLivArea'], plot=plt)
plt.show()
from scipy.stats import norm, skew
numeric_feats = train_data.dtypes[train_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]
skewness = skewness.dropna();

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index

lam = 0.15
for feat in skewed_features :
    if(feat != 'SalePrice'):
        train_data[feat] = boxcox1p(train_data[feat], lam)
        test_data[feat] = boxcox1p(test_data[feat], lam)
train_data['TotalPorchSF'] = train_data['OpenPorchSF'] + train_data['EnclosedPorch'] + train_data['3SsnPorch'] + train_data['ScreenPorch']
test_data['TotalPorchSF'] = test_data['OpenPorchSF'] + test_data['EnclosedPorch'] + test_data['3SsnPorch'] + test_data['ScreenPorch']
train_data['TotalBaths'] = train_data['BsmtFullBath'] + train_data['FullBath'] + 0.5*(train_data['BsmtHalfBath'] + train_data['HalfBath'])
test_data['TotalBaths'] = test_data['BsmtFullBath'] + test_data['FullBath'] + 0.5*(test_data['BsmtHalfBath'] + test_data['HalfBath'])
train_data['Age'] = train_data['YrSold'].astype('int64') - train_data['YearBuilt']
test_data['Age'] = test_data['YrSold'].astype('int64') - test_data['YearBuilt']
train_data['TotalAreaSF'] = train_data['TotalBsmtSF'] + train_data['GrLivArea']
test_data['TotalAreaSF'] = test_data['TotalBsmtSF'] + test_data['GrLivArea']
numerical_features.append('TotalAreaSF')
numerical_features.append('Age')
numerical_features.append('TotalBaths')
numerical_features.append('TotalPorchSF')
numerical_features
# simplified features
train_data['haspool'] = train_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train_data['has2ndfloor'] = train_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train_data['hasgarage'] = train_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train_data['hasbsmt'] = train_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train_data['hasfireplace'] = train_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

test_data['haspool'] = test_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
test_data['has2ndfloor'] = test_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test_data['hasgarage'] = test_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
test_data['hasbsmt'] = test_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test_data['hasfireplace'] = test_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
print("Find most important features relative to target")
corr = train_data.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
#print(corr.SalePrice)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(corr.SalePrice)
# OverallQual      0.820322
# GrLivArea        0.728439
# GarageCars       0.679303
# TotalBaths       0.668820
# GarageArea       0.652430
# TotalAreaSF      0.639406
# TotalBsmtSF      0.639033
# 1stFlrSF         0.605857
# YearBuilt        0.601263
# FullBath         0.592964

important_features = ['OverallQual', 'GrLivArea', 'GarageCars','TotalBaths','GarageArea','TotalBsmtSF','TotalAreaSF','1stFlrSF','FullBath','YearBuilt']

for feature in important_features:
    train_data[feature+"-s2"] = train_data[feature] ** 2
    train_data[feature+"-s3"] = train_data[feature] ** 3

    test_data[feature+"-s2"] = test_data[feature] ** 2
    test_data[feature+"-s3"] = test_data[feature] ** 3
#check the new shape of train_data and test_data
train_data.shape, test_data.shape
print("Find most important features relative to target")
corr = train_data.corr()
corr.sort_values(["SalePrice"], ascending = True, inplace = True)
#print(corr.SalePrice)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(corr.SalePrice)
# Age              -0.601521 
# EnclosedPorch    -0.152172
# KitchenAbvGr     -0.148353
# MSSubClass       -0.077021
# LowQualFinSF     -0.043052
# OverallCond      -0.040999
# YrSold           -0.025930
# MiscVal          -0.022768
# BsmtHalfBath     -0.019457
# BsmtFinSF2       -0.009059


train_data.drop(['Age'], axis=1, inplace=True)
train_data.drop(['EnclosedPorch'], axis=1, inplace=True)
train_data.drop(['KitchenAbvGr'], axis=1, inplace=True)
train_data.drop(['MSSubClass'], axis=1, inplace=True)
#train_data.drop(['LowQualFinSF'], axis=1, inplace=True)
# train_data.drop(['OverallCond'], axis=1, inplace=True)
# train_data.drop(['YrSold'], axis=1, inplace=True)
# train_data.drop(['MiscVal'], axis=1, inplace=True)
# train_data.drop(['BsmtHalfBath'], axis=1, inplace=True)
# train_data.drop(['BsmtFinSF2'], axis=1, inplace=True)


test_data.drop(['Age'], axis=1, inplace=True)
test_data.drop(['EnclosedPorch'], axis=1, inplace=True)
test_data.drop(['KitchenAbvGr'], axis=1, inplace=True)
test_data.drop(['MSSubClass'], axis=1, inplace=True)
#test_data.drop(['LowQualFinSF'], axis=1, inplace=True)
# test_data.drop(['OverallCond'], axis=1, inplace=True)
# test_data.drop(['YrSold'], axis=1, inplace=True)
# test_data.drop(['MiscVal'], axis=1, inplace=True)
# test_data.drop(['BsmtHalfBath'], axis=1, inplace=True)
# test_data.drop(['BsmtFinSF2'], axis=1, inplace=True)
#check the new shape of train_data and test_data
train_data.shape, test_data.shape
# Look at data correlation using heatmap
corr = train_data.corr()
plt.subplots(figsize=(15,15))
sns.heatmap(corr, fmt='.1f', cmap="Blues", square=True)
plt.show()
train1 = train_data.copy()
test1 = test_data.copy()

data = pd.concat([train1,test1], axis=0)
train_rows = train1.shape[0]

for feature in features_with_categorial:
    dummy = pd.get_dummies(data[feature])
    for col_name in dummy.columns:
        if (type(col_name) == str):
            dummy.rename(columns={col_name: feature+"_"+col_name}, inplace=True)
        else:
            dummy.drop([col_name], axis = 1, inplace=True)
    data = pd.concat([data, dummy], axis = 1)
    data.drop([feature], axis = 1, inplace=True)

train1 = data.iloc[:train_rows, :]
test1 = data.iloc[train_rows:, :] 

print("Train",train1.shape)
print("Test",test1.shape)
overfit = []
for i in train1.columns:
    counts = train1[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(train1) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)
overfit.append('MSZoning_C (all)')

train1 = train1.drop(overfit, axis=1).copy()
test1 = test1.drop(overfit, axis=1).copy()
#check the new shape of train_data and test_data
train1.shape, test1.shape
from sklearn.preprocessing import MinMaxScaler, RobustScaler

scaling_features = [feature for feature in train1.columns if feature not in ['SalePrice']]
scaling_features
scaler = RobustScaler()
scaler.fit(train1[scaling_features])
X_train = scaler.transform(train1[scaling_features])
X_test = scaler.transform(test1[scaling_features])
print("Train", X_train.shape)
print("Test", X_test.shape)
scaler = MinMaxScaler()
scaler.fit(train1[scaling_features])
X_train = scaler.transform(train1[scaling_features])
X_test = scaler.transform(test1[scaling_features])
print("Train", X_train.shape)
print("Test", X_test.shape)
y_train = train1['SalePrice']
X = pd.concat([train1[['SalePrice']].reset_index(drop=True), pd.DataFrame(X_train, columns = scaling_features)], axis =1)
print(X.shape)
X.head()
train1.drop(['SalePrice'], axis=1, inplace=True)
test1.drop(['SalePrice'], axis=1, inplace=True)
#import libraries
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR

n_folds = 12

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmsle(y_train, y_pred):
    return np.sqrt(mean_squared_error(y_train, y_pred))
lasso = Lasso(alpha =0.0005, random_state=0)
elasticNet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=0)
kernelRidge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
svr = SVR(C= 20, epsilon= 0.008, gamma=0.0003)
gradientBoosting = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =0)
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =0, nthread = -1)
lgbm = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11, random_state=0)
randomForest = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=0)
scores ={}
score = rmsle_cv(lasso)
print("Lasso:: Mean:",score.mean(), " Std:", score.std())
scores['lasso'] = (score.mean(), score.std())
lasso_model = lasso.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_train)
rmsle(y_train,y_pred_lasso)
score = rmsle_cv(elasticNet)
print("ElasticNet:: Mean:",score.mean(), " Std:", score.std())
scores['elasticNet'] = (score.mean(), score.std())
elasticNet_model = elasticNet.fit(X_train, y_train)
y_pred_elasticNet = elasticNet_model.predict(X_train)
rmsle(y_train,y_pred_elasticNet)
score = rmsle_cv(kernelRidge)
print("KernelRidge:: Mean:",score.mean(), " Std:", score.std())
scores['kernelRidge'] = (score.mean(), score.std())
kernelRidge_model = kernelRidge.fit(X_train, y_train)
y_pred_kernelRidge = kernelRidge_model.predict(X_train)
rmsle(y_train,y_pred_kernelRidge)
score = rmsle_cv(svr)
print("SVR:: Mean:",score.mean(), " Std:", score.std())
scores['svr'] = (score.mean(), score.std())
svr_model = svr.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_train)
rmsle(y_train,y_pred_svr)
score = rmsle_cv(xgb)
print("XGBRegressor:: Mean:",score.mean(), " Std:", score.std())
scores['xgb'] = (score.mean(), score.std())
xgb_model = xgb.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_train)
rmsle(y_train,y_pred_xgb)
score = rmsle_cv(gradientBoosting)
print("GradientBoostingRegressor:: Mean:",score.mean(), " Std:", score.std())
scores['gradientBoosting'] = (score.mean(), score.std())
gradientBoosting_model = gradientBoosting.fit(X_train, y_train)
y_pred_gradientBoosting = gradientBoosting_model.predict(X_train)
rmsle(y_train,y_pred_gradientBoosting)
score = rmsle_cv(lgbm)
print("LGBMRegressor:: Mean:",score.mean(), " Std:", score.std())
scores['lgbm'] = (score.mean(), score.std())
lgbm_model = lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_train)
rmsle(y_train,y_pred_lgbm)
score = rmsle_cv(randomForest)
print("RandomForestRegressor:: Mean:",score.mean(), " Std:", score.std())
scores['randomForest'] = (score.mean(), score.std())
randomForest_model = randomForest.fit(X_train, y_train)
y_pred_randomForest = randomForest_model.predict(X_train)
rmsle(y_train,y_pred_randomForest)
def ensemble_models(X):
    return ((0.125 * lasso_model.predict(X)) +
            (0.125 * elasticNet_model.predict(X)) +
            (0.125 * kernelRidge_model.predict(X)) +
            (0.125 * svr_model.predict(X)) +
            (0.125 * gradientBoosting_model.predict(X)) + 
            (0.125 * xgb_model.predict(X)) +
            (0.125 * lgbm_model.predict(X)) +
            (0.125 * randomForest_model.predict(X)))
averaged_score = rmsle(y_train, ensemble_models(X_train))
scores['averaged'] = (averaged_score, 0)
print('RMSLE averaged score on train data:', averaged_score)
def stack_models(X):
  return (
            (0.1 * lasso_model.predict(X)) +
            (0.1 * elasticNet_model.predict(X)) +
            (0.2 * gradientBoosting_model.predict(X)) + 
            (0.2 * svr_model.predict(X)) +
            (0.2 * xgb_model.predict(X)) +
            (0.2 * lgbm_model.predict(X)))

stacked_score = rmsle(y_train, stack_models(X_train))
scores['stacked'] = (stacked_score, 0)
print('RMSLE stacked score on train data:', stacked_score)
sns.set_style("white")
fig = plt.figure(figsize=(20, 10))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.4f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score', size=20, labelpad=12.5)
plt.xlabel('Regression Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)
plt.title('Regression Model Scores', size=20)
plt.show()
test_predict = np.exp(stack_models(X_test))
print(test_predict[:5])
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = test_predict
sub.to_csv('submission.csv',index=False)
sub1 = pd.read_csv('submission.csv')
sub1.head()