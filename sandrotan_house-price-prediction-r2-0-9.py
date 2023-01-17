# Import libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
# Load data sets

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_ID = df_test['Id']
# Print the first 10 rows

df_train.head(10)
# Get descriptive stats

df_train.describe()
# Now drop the  'Id' column since it's unnecessary for  the prediction process.

df_train.drop("Id", axis=1, inplace=True)

df_test.drop("Id", axis=1, inplace=True)

print(df_train.columns)

print(df_train['SalePrice'].describe())
# histogram

sns.distplot(df_train['SalePrice'])
# scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# scatter plot LotFrontage/saleprice

var = 'LotFrontage'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# scatter plot 1stFlrSF/saleprice

var = '1stFlrSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# boxplot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data, palette=sns.color_palette("hls", 8))

fig.axis(ymin=0, ymax=800000)
# boxplot YearBuilt/saleprice

var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data, palette=sns.color_palette("hls", 8))

fig.axis(ymin=0, ymax=800000)

plt.xticks(rotation=90)
# correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)



# saleprice correlation matrix

k = 10  # number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

print(cols)

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values,

                 xticklabels=cols.values)
# Combine train and test set for preprocessing

X_all = pd.concat((df_train, df_test)).reset_index(drop=True)

X_all.drop(['SalePrice'], axis=1, inplace=True)

print("X_all size is : {}".format(X_all.shape))
# Handle missing values (NA, NaN)

# Check if any missing values

X_all_na = (X_all.isnull().sum() / len(X_all)) * 100

X_all_na = X_all_na.drop(X_all_na[X_all_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio': X_all_na})

print(missing_data.head(10))
# Drop features

X_all = X_all.drop(['PoolQC', "MiscFeature", "Alley", 'Fence', 'FireplaceQu'], axis=1)
# Fill features

X_all["MasVnrType"] = X_all["MasVnrType"].fillna("None")

X_all["MasVnrArea"] = X_all["MasVnrArea"].fillna(0)

X_all['MSZoning'] = X_all['MSZoning'].fillna(X_all['MSZoning'].mode()[0])

X_all["Functional"] = X_all["Functional"].fillna("Typ")

X_all['Electrical'] = X_all['Electrical'].fillna(X_all['Electrical'].mode()[0])

X_all['KitchenQual'] = X_all['KitchenQual'].fillna(X_all['KitchenQual'].mode()[0])

X_all['Exterior1st'] = X_all['Exterior1st'].fillna(X_all['Exterior1st'].mode()[0])

X_all['Exterior2nd'] = X_all['Exterior2nd'].fillna(X_all['Exterior2nd'].mode()[0])

X_all['SaleType'] = X_all['SaleType'].fillna(X_all['SaleType'].mode()[0])

X_all['MSSubClass'] = X_all['MSSubClass'].fillna("None")

X_all['Utilities'] = X_all['Utilities'].fillna('AllPub')

X_all["LotFrontage"] = X_all.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    X_all[col] = X_all[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    X_all[col] = X_all[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    X_all[col] = X_all[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    X_all[col] = X_all[col].fillna('None')
# Tranform numeric feature to categorical 

X_all['MSSubClass'] = X_all['MSSubClass'].apply(str)
# Check remaining missing values if any

X_all_na = (X_all.isnull().sum() / len(X_all)) * 100

X_all_na = X_all_na.drop(X_all_na[X_all_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio': X_all_na})

print("remaining missing value:", missing_data)
# Deal with highly skewed features

from scipy.stats import skew



numeric_feats = X_all.dtypes[X_all.dtypes != "object"].index

skewed_feats = X_all[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew': skewed_feats})

print(skewness)
skewness = skewness[abs(skewness) > 1]

print("There are {} skewed numerical features to transform".format(skewness.shape[0]))



skewed_features = skewness.index

for feat in skewed_features:

    X_all[feat] = np.log1p((X_all[feat]))
# Make categorical features dummies

X_all = pd.get_dummies(X_all)
# Create train and test set

X_train = X_all[:df_train.shape[0]].values

X_test = X_all[df_train.shape[0]:].values

y_train = np.log1p(df_train['SalePrice'])

print(X_train.shape, y_train.shape)

from sklearn.model_selection import KFold, cross_val_score

kfolds = KFold(n_splits=10, shuffle=True, random_state=1)

# Define model

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=20000, learning_rate=0.02, max_depth=3, max_features='sqrt',

                                min_samples_leaf=5, min_samples_split=5, loss='ls')
# Define model hyper parameters for cross validation

gbr_param_grid = {

    "n_estimators": [40000, 60000, 80000],

    "learning_rate": [0.01, 0.02],

    "max_depth": [3],

    "max_features": ["sqrt"],

    "min_samples_leaf": [5],

    "min_samples_split": [5],

    "loss": ["ls", 'huber']

}
from sklearn.model_selection import  GridSearchCV

def grid_search(model, param_grid):

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=kfolds, scoring='r2', verbose=0, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print("Best: {0:.4f} using {1}".format(grid_search.best_score_, grid_search.best_params_))



    return grid_search
grid_search(gbr, gbr_param_grid)
# Prediction



# Use the below parameters grid:

#              param_grid={'learning_rate': [0.02, 0.01], 'loss': ['ls', 'huber'],

#                          'max_depth': [3], 'max_features': ['sqrt'],

#                          'min_samples_leaf': [5], 'min_samples_split': [5],

#                          'n_estimators': [40000, 60000, 80000]},

    

# model_final = grid_search(knr, knr_param_grid)

# y_pred = np.expm1(np.expm1(model_final.predict(X_test)))

# y_test = pd.DataFrame()

# y_test['Id'] = test_id.values

# y_test['SalePrice'] = y_pred

# y_test.to_csv('submission.csv',index=False)