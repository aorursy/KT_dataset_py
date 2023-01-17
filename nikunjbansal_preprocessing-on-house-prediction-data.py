import pandas as pd

import numpy as np

import pylab as pl

import seaborn as sns

import sys, os

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

from matplotlib import pyplot as plt

from sklearn.metrics import r2_score,mean_squared_error,explained_variance_score

import missingno as nullnum
input_data_dir = "../input/house-prices-advanced-regression-techniques"

train_df = pd.read_csv(os.path.join(input_data_dir, "train.csv"))

test_df = pd.read_csv(os.path.join(input_data_dir, "test.csv"))

full_df = pd.concat([train_df, test_df], sort=False)
print("Size of training dataset       : {}".format(train_df.shape))

print("Size of test dataset           : {}".format(test_df.shape))
full_df.info()
full_df.describe().T
pyplot.figure(figsize=(20, 7))

sns.heatmap(train_df.isnull(), cbar=False)

pyplot.show()
nullnum.heatmap(train_df)
x = train_df.SalePrice.sort_values().reset_index().index

y = train_df.SalePrice.sort_values().reset_index()["SalePrice"]

plt.scatter(x, y, color = "darkblue")

plt.xlabel("Index", size=36)

plt.ylabel("Sales Price", size=36)

plt.title("Distribution of target variable", size=30, pad=26)
corr_data = train_df.corr()

pyplot.figure(figsize=(32, 22))

sns.set_style('ticks')

sns.heatmap(corr_data, annot=True)

pyplot.show()
corr_data.SalePrice.apply(lambda x: abs(x)).sort_values(ascending=False).iloc[1:11][::-1].plot(kind='barh',color='red')

plt.title("Top 10 highly correlated features", size=20, pad=26)

plt.xlabel("Correlation coefficient")

plt.ylabel("Features")
train_df = train_df.drop(['Id'], axis=1)

test_df = test_df.drop(['Id'], axis=1)

full_df = full_df.drop(['Id'], axis=1)
categorical_features = full_df.select_dtypes(include = ["object"]).columns

numerical_features = full_df.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

full_df_num = full_df[numerical_features]

full_df_cat = full_df[categorical_features]
print("Total missing values for categorical features in full_df: " + str(full_df_cat.isnull().values.sum()))
missing_val_col = ["Alley", 

                   "PoolQC", 

                   "MiscFeature",

                   "Fence",

                   "FireplaceQu",

                   "GarageType",

                   "GarageFinish",

                   "GarageQual",

                   "GarageCond",

                   'BsmtQual',

                   'BsmtCond',

                   'BsmtExposure',

                   'BsmtFinType1',

                   'BsmtFinType2',

                   'MasVnrType']



for i in missing_val_col:

    full_df_cat[i] = full_df_cat[i].fillna('None')
full_df_cat['Functional'] = full_df_cat['Functional'].fillna('Typ') 

full_df_cat['Utilities'] = full_df_cat['Utilities'].fillna('AllPub') 

full_df_cat['Exterior1st'] = full_df_cat['Exterior1st'].fillna(full_df_cat['Exterior1st'].mode()[0]) 

full_df_cat['Exterior2nd'] = full_df_cat['Exterior2nd'].fillna(full_df_cat['Exterior2nd'].mode()[0])

full_df_cat['KitchenQual'] = full_df_cat['KitchenQual'].fillna("TA") 

full_df_cat['SaleType'] = full_df_cat['SaleType'].fillna(full_df_cat['SaleType'].mode()[0])

full_df_cat['Electrical'] = full_df_cat['Electrical'].fillna("SBrkr")

full_df_cat['MSZoning'] = full_df_cat['MSZoning'].fillna("RL") 
print("Remaining missing values for categorical features in full_df: " + str(full_df_cat.isnull().values.sum()))
full_df_cat_encod = full_df_cat.iloc[:,0:43]

le = LabelEncoder()



for i in full_df_cat_encod:

    full_df_cat_encod[i]=le.fit_transform(full_df_cat_encod[i])
ntrain = train_df.shape[0]
numerical_features = train_df.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

train_df_num = train_df[numerical_features]
print("Total missing values for numerical features in full_df: " + str(full_df_num.isnull().values.sum()))

full_df_num = full_df_num.fillna(full_df_num.median())

print("Remaining missing values for numerical features in full_df: " + str(full_df_num.isnull().values.sum()))
train_df_X = pd.concat([full_df_cat_encod[:1022],full_df_num[:1022]],axis=1)

train_df_X.shape
train_df_X.head()
train_df_y = train_df.SalePrice[:1022]

train_df_y.head()
val_df_X = pd.concat([full_df_cat_encod[1022:ntrain],full_df_num[1022:ntrain]],axis=1)

val_df_X.shape
val_df_y = train_df.SalePrice[1022:ntrain]

val_df_y.shape
test_df_X = pd.concat([full_df_cat_encod[ntrain:],full_df_num[ntrain:]],axis=1)

test_df_X.shape
test_df_X.head()
SVM_regressor = SVR(kernel='rbf')

SVM_regressor.fit(train_df_X,train_df_y)
SVM_val_df_pred = SVM_regressor.predict(val_df_X)
SVM_variance_score = explained_variance_score(val_df_y, SVM_val_df_pred)

SVM_r2_score = r2_score(val_df_y, SVM_val_df_pred)



print("SVM_variance_score: %.5f" %SVM_variance_score)

print("SVM_r2_score: %.5f" %SVM_r2_score)
gbr = GradientBoostingRegressor()

gbr.fit(train_df_X,train_df_y)
gbr_val_df_pred = gbr.predict(val_df_X)
GBR_variance_score = explained_variance_score(val_df_y, gbr_val_df_pred)

GBR_r2_score = r2_score(val_df_y, gbr_val_df_pred)



print("GBR_variance_score: %.5f" %GBR_variance_score)

print("GBR_r2_score: %.5f" %GBR_r2_score)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(train_df_X,train_df_y)
xg_val_df_pred = xg_reg.predict(val_df_X)
xg_variance_score = explained_variance_score(val_df_y, xg_val_df_pred)

xg_r2_score = r2_score(val_df_y, xg_val_df_pred)



print("xg_variance_score: %.5f" %xg_variance_score)

print("xg_r2_score: %.5f" %xg_r2_score)
test_df_y = SVM_regressor.predict(test_df_X)
submission_df = pd.read_csv(os.path.join(input_data_dir, "sample_submission.csv"))
submission_df['SalePrice'] = test_df_y
#Save to csv

submission_df.to_csv('submission.csv',index=False)

submission_df.head(103)