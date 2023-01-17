import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from scipy.stats import norm, skew
#Import data

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

#Inspect train data

train.head()
#detailed info about column types and missing values

train.info()
#Isolate y, drop categorical variables, drop Na rows. Separate y and X

data = train

data = data.select_dtypes(exclude=['object'])

data = data.dropna(axis=0)

y = data["SalePrice"]

X = data.drop(["SalePrice"], axis=1)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
#Always check amount of na before starting

data.isnull().sum().sum()
X_train, X_test, y_train, y_test = train_test_split(X, y)
#RF_1

model_rf1 = RandomForestRegressor(n_estimators = 500, random_state=17)

model_rf1.fit(X_train, y_train)
rf1_pred = model_rf1.predict(X_test)
mae_rf1 = mean_absolute_error(rf1_pred, y_test)

print(mae_rf1)
#XGB_1

model_xgb1 = XGBRegressor(n_estimators=500, learning_rate=0.1 ,random_state = 17)

model_xgb1.fit(X_train, y_train)
xgb1_pred = model_xgb1.predict(X_test)
mae_xgb1 = mean_absolute_error(xgb1_pred, y_test)

print(mae_xgb1)
#make a submission for the competition

predictor_cols = X_train.columns

test = test[predictor_cols]

xgb1_finalpred = model_xgb1.predict(test)

print(xgb1_finalpred)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': xgb1_finalpred})

my_submission.to_csv('xgb1_pred.csv', index=False)
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)

ntrain = train.shape[0]

ntest = test.shape[0]

y = train["SalePrice"]

data = pd.concat([train, test], sort = True).reset_index(drop=True)
#EDA

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))

sns.distplot(y_train, bins=50,  rug = True, ax = ax1)

sns.regplot(data["SalePrice"], data["GrLivArea"], ax = ax2)

plt.show()
#inspect correlation: select only numeric cols

numeric_cols = data.select_dtypes(exclude=['object'])

num_corr = numeric_cols.corr()

fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(num_corr, linewidths =.5)

plt.show()
#Finally drop SalePrice column, which we have previously stored in variable y

data.drop(['SalePrice'], axis=1, inplace=True)
nullcols = data.isnull().sum().sort_values(ascending=False)

nullcols = pd.DataFrame(nullcols, columns=["Total"])

nullcols[nullcols["Total"] > 0]
#considering the great amount of NA I drop some variables

#also drop utilities

data.drop("PoolQC", axis = 1, inplace = True)

data.drop("MiscFeature", axis = 1, inplace = True)

data.drop("Alley", axis = 1, inplace = True)

data.drop("Fence", axis = 1, inplace = True)

data = data.drop(['Utilities'], axis=1)

#dealing with other NA

data['FireplaceQu'] = data['FireplaceQu'].fillna("No")

data['LotFrontage'] = data['LotFrontage'].fillna(0)

data['GarageCond'] = data['GarageCond'].fillna("No")

data['GarageType'] = data['GarageType'].fillna("No")

data['GarageYrBlt'].fillna((data['GarageYrBlt'].mean()), inplace=True)

data['GarageFinish'] = data['GarageFinish'].fillna("No")

data['GarageQual'] = data['GarageQual'].fillna("No")

data['BsmtExposure'] = data['BsmtExposure'].fillna("No")

data['BsmtFinType2'] = data['BsmtFinType2'].fillna("No")

data['BsmtFinType1'] = data['BsmtFinType1'].fillna("No")

data['BsmtCond'] = data['BsmtCond'].fillna("No")

data['BsmtQual'] = data['BsmtQual'].fillna("No")

data['MasVnrArea'] = data['MasVnrArea'].fillna("No")

data['MasVnrType'] = data['MasVnrType'].fillna("No")

data['Electrical'] = data['Electrical'].fillna("SBrkr")

data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    data[col] = data[col].fillna(0)

data["Functional"] = data["Functional"].fillna("Typ")

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

for col in ('GarageArea', 'GarageCars'):

    data[col] = data[col].fillna(0)

data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data.isnull().sum().sum()
#transforming numerical variables that are actually categorical in the correct format

data['MSSubClass'] = data['MSSubClass'].apply(str)

data['OverallCond'] = data['OverallCond'].astype(str)

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)
#Check which categorical variables are left and process them

categorical_variables = data.select_dtypes(include=['object'])

cat_cols = categorical_variables.columns

cat_cols
#select meaningful columns to be labeled

from sklearn.preprocessing import LabelEncoder

cols = ('BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC',  'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for col in cols:

    encoder = LabelEncoder() 

    encoder.fit(list(data[col].values)) 

    data[col] = encoder.transform(list(data[col].values))

#add total house area variable

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

data.info()

numeric_feats = data.dtypes[data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness)>0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    data[feat] = boxcox1p(data[feat], lam)
#add dummies

data = pd.get_dummies(data)    

data.info()
#Models

train = data[:ntrain]

test = data[ntrain:]

X = train

X_train, X_test, y_train, y_test = train_test_split(X, y)
model_xgb2 = XGBRegressor(n_estimators=500, learning_rate=0.1 ,random_state = 17)
model_xgb2.fit(X_train, y_train)
xgb2_pred = model_xgb2.predict(X_test)

mae_xgb2 = mean_absolute_error(xgb2_pred, y_test)

print(mae_xgb2)
predictor_cols = X_train.columns

test = test[predictor_cols]

xgb2_finalpred = model_xgb2.predict(test)

print(xgb2_finalpred)

my_submission2 = pd.DataFrame({'Id': test_ID, 'SalePrice': xgb2_finalpred})

my_submission2.to_csv('xgb2_pred.csv', index=False)
#improving the model with hyperparameters



from sklearn.pipeline import Pipeline

my_pipeline = Pipeline(steps=[('model', XGBRegressor(colsample_bytree = 0.46, gamma = 0.047, 

                             learning_rate = 0.05, max_depth = 3, 

                             min_child_weight= 1.8, n_estimators = 2200,

                             reg_alpha = 0.47, reg_lambda = 0.86,

                             subsample = 0.52, random_state =17))

                             ])



from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline, X_train, y_train,

                              cv=5,

                              scoring='neg_mean_absolute_error')

print("Average MAE score (across experiments):")

print(scores.mean())
model_xgb3 = XGBRegressor(colsample_bytree = 0.46, gamma = 0.047, 

                             learning_rate = 0.05, max_depth = 3, 

                             min_child_weight= 1.8, n_estimators = 2200,

                             reg_alpha = 0.47, reg_lambda = 0.86,

                             subsample = 0.52, random_state =17)

model_xgb3.fit(X_train, y_train)
predictor_cols = X_train.columns

test = test[predictor_cols]

xgb3_finalpred = model_xgb3.predict(test)

print(xgb3_finalpred)

my_submission3 = pd.DataFrame({'Id': test_ID, 'SalePrice': xgb3_finalpred})

my_submission3.to_csv('xgb3_pred.csv', index=False)