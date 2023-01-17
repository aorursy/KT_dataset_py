import pandas as pd
# save filepath to variable for easier access

file_path = '../input/train.csv'

# read the data and store data in DataFrame titled data

data = pd.read_csv(file_path) 
# print a data summary here

data.describe()
# The mean sale price is:

unrounded = data.SalePrice.mean()

rounded = round(unrounded)

roundedInt = int(rounded)

print("unrounded =", unrounded)

print("rounded =", roundedInt)
# The year the oldest house was built is:

oldestHouseYear = data.YearBuilt.min()

print(oldestHouseYear)
import matplotlib.pyplot as plt
plt.hist(data.SalePrice, bins=50);
# plot a histogram of the YearBuilt column here

plt.hist(data.YearBuilt, bins=50);
y = data.SalePrice
# create a list of numeric variables called predictors

print(data.columns)

x = ['LotArea','OverallQual','OverallCond','YearBuilt','YrSold']
# create a DataFrame called X containing the predictors here

X = data[x]
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
# fit your decision tree model here

model.fit(X,y)
# make predictions with your model here

predY = model.predict(X)
# compare the model's predictions with the true sale prices of the first few houses here

X.assign(Prediction = predY).assign(Y = y).head()
from sklearn.metrics import mean_absolute_error
# compute the mean absolute error of your predictions here

mean_absolute_error(y,predY)
# compute the mean absolute error on the validation data here

from sklearn.model_selection import train_test_split

val_model = DecisionTreeRegressor()

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, test_size = 0.2)

val_model.fit(train_X, train_y)

val_predictions = val_model.predict(val_X)

mean_absolute_error(val_y, val_predictions)
# make predictions for the test data here



# load libraries

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_selection import SelectKBest, RFE, RFECV, SelectFromModel, f_regression

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.metrics import mean_absolute_error

rs = 4319
# load the reference data to use for modelling

modelSet = pd.read_csv('../input/train.csv')

submitSet = pd.read_csv('../input/test.csv')



# preprocess the data 

# remove redundant columns

modelSet = modelSet.drop(labels = ['Id'], axis=1)

submitSet = submitSet.drop(labels = ['Id'], axis=1)



# Categorical/Numerical feature separation

# Categorical

# MSSubClass, MSZoning, Street(B), Alley, LotShape(N), LandContour, Utilities(N), LotConfig, LandSlope(N), Neighborhood, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, 

# MasVnrType, ExterQual(N), ExterCond(N), Foundation, BsmtQual(N), BsmtCond(N), BsmtExposure(N), BsmtFinType1(N), BsmtFinType2(N), Heating, HeatingQC(N), CentralAir(B), Electrical, KitchenQual(N), Functional(N), 

# FireplaceQu(N), GarageType, GarageFinish(N), GarageQual(N), GarageCond(N), PavedDrive(N), PoolQC(N), Fence(N), MiscFeature, SaleType, SaleCondition

# Numerical

# LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, 

# FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageYrBlt, GarageCars, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal, MoSold, YrSold

cat_feat = modelSet[['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 

                     'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 

                     'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 

                     'MiscFeature', 'SaleType', 'SaleCondition']].copy()

num_feat = modelSet[['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 

                     'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 

                     'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']].copy()

cat_subfeat = submitSet[['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 

                     'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 

                     'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 

                     'MiscFeature', 'SaleType', 'SaleCondition']].copy()

num_subfeat = submitSet[['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 

                     'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 

                     'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']].copy()



# convert categorical features to numeric features

# within categorical features, conver all instances of NaN to 'x'

cat_feat = cat_feat.fillna('x')

cat_subfeat = cat_subfeat.fillna('x')

# concatenate categorical features (numeric assignment of values in one dataframe is the same in other)

cat_allfeat = pd.concat([cat_feat, cat_subfeat])

# one-hot encoding for no inherent order (nominal), numeric if there is an order (ordinal). ...Binary exists too

enc = OneHotEncoder()

temp = enc.fit_transform(cat_allfeat)

temp = pd.DataFrame(temp.todense())

# separate features back to modelling features and (sub)mission features

conv_cat_feat = temp.iloc[:cat_feat.shape[0]]

conv_cat_subfeat = temp.iloc[cat_feat.shape[0]:cat_allfeat.shape[0]]

conv_cat_subfeat = conv_cat_subfeat.reset_index(drop=True)



# data conversion

# within numerical features, convert all instances of non-numeric values (should all be NA) to 0

num_feat = num_feat.fillna(0)

num_subfeat = num_subfeat.fillna(0)



# merge converted categorical features with numerical features

X_all = pd.concat([conv_cat_feat, num_feat], axis=1)

X_suball = pd.concat([conv_cat_subfeat, num_subfeat], axis=1)



# data transformation

# data normalization



#num_feat.describe()

#num_subfeat.describe()
# Split train set and test set for modelling

X_train, X_test, y_train, y_test = train_test_split(X_all, modelSet['SalePrice'], test_size=0.1, random_state=rs)



# Select a regression model

# LinearRegression

model = LinearRegression(fit_intercept=False)



# Feature selection via SelectKBest



# Feature selection via RFE (Recursive Feature Elimination)



# Feature selection via RFECV (Recursive Feature Elmination with Cross Validation)

selector = RFECV(estimator=model, step=1, min_features_to_select=1, cv=StratifiedKFold(n_splits=5, random_state=rs),scoring='neg_mean_absolute_error')

feat_res = selector.fit(X_train, y_train)

print("Optimal number of features : %d" % selector.n_features_)



# Feature selection via SelectFromModel



# The selected features

sel_X_train = feat_res.transform(X_train)

sel_X_test = feat_res.transform(X_test)



y_true, y_pred = y_test, feat_res.estimator_.predict(sel_X_test)

mean_absolute_error(y_true, y_pred)

# prepare your submission file here

test = pd.read_csv('../input/test.csv')

submission_features = feat_res.transform(X_suball)

test_predictions = feat_res.estimator_.predict(submission_features)

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': test_predictions})

submission.to_csv('submission.csv', index=False)