import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import spacy

df = pd.read_csv("../input/train.csv")



pd.set_option('display.max_rows', 5000)

#df.info()

df.columns
# Y 

y=df.SalePrice



#filling up empty cells with df.fillna

df_clean = df.drop(['SalePrice', 'PoolQC', 'Fence', 'MiscFeature', 'Alley', 'FireplaceQu', 'Utilities'], axis=1)

df_clean.notnull().all()

df_clean['LotFrontage'] = df_clean['LotFrontage'].fillna(df_clean['LotFrontage'].mean())

df_clean['MasVnrType'] = df_clean['MasVnrType'].fillna(df_clean['MasVnrType'].mode()[0])

df_clean['MasVnrArea'] = df_clean['MasVnrArea'].fillna(df_clean['MasVnrArea'].mean())

df_clean['BsmtQual'] = df_clean['BsmtQual'].fillna(df_clean['BsmtQual'].mode()[0])

df_clean['BsmtCond'] = df_clean['BsmtCond'].fillna(df_clean['BsmtCond'].mode()[0])

df_clean['BsmtExposure'] = df_clean['BsmtExposure'].fillna(df_clean['BsmtExposure'].mode()[0])

df_clean['BsmtFinType1'] = df_clean['BsmtFinType1'].fillna(df_clean['BsmtFinType1'].mode()[0])

df_clean['BsmtFinType2'] = df_clean['BsmtFinType2'].fillna(df_clean['BsmtFinType2'].mode()[0])

df_clean['Electrical'] = df_clean['Electrical'].fillna(df_clean['Electrical'].mode()[0])

df_clean['GarageType'] = df_clean['GarageType'].fillna(df_clean['GarageType'].mode()[0])

df_clean['GarageYrBlt'] = df_clean['GarageYrBlt'].fillna(df_clean['GarageYrBlt'].mean())

df_clean['GarageFinish'] = df_clean['GarageFinish'].fillna(df_clean['GarageFinish'].mode()[0])

df_clean['GarageQual'] = df_clean['GarageQual'].fillna(df_clean['GarageQual'].mode()[0])

df_clean['GarageCond'] = df_clean['GarageCond'].fillna(df_clean['GarageCond'].mode()[0])

#drop duplicate IDs and NANs with df.drop_duplicates and df.dropna respectively

df_clean.dropna(subset=df_clean.columns, inplace=True)

#df_clean.drop_duplicates(subset=df_clean['Id'], keep='first')





X_objects = df_clean.select_dtypes(include=[object])





import matplotlib.pyplot as plt

import xgboost as xgb

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge



"""Data Cleaning"""

#One-hot encoding

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#1) Instantiate

le = LabelEncoder()

#2) Firt and transform

X_le = X_objects.apply(le.fit_transform)

print(X_le.head())



#one hot encoder

#1) instantiate

oe = OneHotEncoder(categories='auto')

#2) fit

oe.fit(X_le)

#3) transform

onehotlabels = oe.transform(X_le).toarray()

print(onehotlabels.shape)

print(onehotlabels)

#Define a list of categorical columns that need to be encoded

#cat_columns = ['MSZoning', 'Street','Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig','LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType','HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

#       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1','BsmtFinType2', 'Heating','HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual','Functional', 'FireplaceQu','GarageType','GarageQual',

#       'GarageCond', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType',

#       'SaleCondition'

print(df_clean.columns)    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



print(df_clean.columns)

print(X_objects.columns)

df_clean.drop(['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',

       'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',

       'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',

       'SaleType', 'SaleCondition'], inplace=True, axis=1)

# next step - move to OneHotEncoding

ohe = OneHotEncoder(categories = 'auto')



ohe.fit(X_le) #see the cell above for X_le

onehotted = ohe.transform(X_le).toarray()

#print(onehotted)



one_hot_df = pd.DataFrame(onehotted)

one_hot_df.info()



df_new = pd.concat([df_clean, one_hot_df], axis=1)

#df_new.drop(df_new.select_dtypes['object'], inplace=True, axis=1)

df_new.head()
#Try pd.get_dummies for onehotencoding here

#optional

#train_test



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_new, y, test_size=0.25)

df_new.head()
#DF is ready. perform feature selection for identifying the best features in the data set

#Using XGBoost for feature seection



from xgboost import XGBClassifier

from xgboost import XGBRegressor

from xgboost import plot_importance

import matplotlib.pyplot as plt



xgbr = XGBRegressor()

xgbr.fit(X_train, y_train)

#print(plot_importance(xgbr))

feat_importance = pd.Series(xgbr.feature_importances_, index=X_train.columns)

feat_importance.nlargest(20).plot(kind='barh')

plt.show()
#new dataframe after feature engineering



df_feat_imp = df_new[['YearBuilt', 184, 144, 'BsmtFinSF1', 'YearRemodAdd', 'HalfBath', 186, 'Fireplaces', '1stFlrSF', '2ndFlrSF', 3, 177, 'FullBath', 'GrLivArea', 'TotalBsmtSF', 203, 130, 'GarageCars', 142, 'OverallQual']]
#train_test_split again



X_train_feat_imp, X_test_feat_imp, y_train_feat_imp, y_test_feat_imp = train_test_split(df_feat_imp, y, test_size = 0.25)



#covariance calculation

from sklearn.metrics import explained_variance_score



#train the model

xgb_imp = XGBRegressor()

xgb_imp.fit(X_train, y_train)

y_pred = xgb_imp.predict(X_test)

print("covariance score: ", explained_variance_score(y_pred, y_test))



#RandomSearchCV
