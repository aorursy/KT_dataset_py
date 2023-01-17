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


import matplotlib.pyplot as plt

import seaborn as sns

sns.set()





from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler





#import Models

from sklearn.ensemble  import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
train_df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
pd.set_option('display.max_columns', None)

train_df.shape, test_df.shape
train_df.head()
#MSSubClass has categorical value as numbers

train_df.MSSubClass=train_df.MSSubClass.astype(str)

test_df.MSSubClass=test_df.MSSubClass.astype(str)
Y_train= train_df["SalePrice"]

train_df.drop("Id", axis=1, inplace=True)

test_Id=test_df.Id

test_df.drop("Id",axis=1, inplace=True)
train_df.describe()
#creating dataframe with categorical data only

train_df_category=train_df.select_dtypes(include="O")

test_df_category=test_df.select_dtypes(include="O")
train_df_category.info()
train_df.shape
#checking null values in test data

print(test_df_category.isna().sum().sort_values(ascending=False).head(),

train_df_category.isna().sum().sort_values(ascending=False).head())
# MAPPING name to NA according to feature discription in the data_discription file





values_na_categorical={"PoolQC":"no_pool","Alley":"no_alley", 'BsmtQual':"no_bsmt", 

        'BsmtCond':"no_bsmt",'BsmtExposure':"no_bsmt", 'BsmtFinType1':"no_bsmt", 

        'BsmtFinType2':"no_bsmt", 'MasVnrType':"None", 'Electrical':"SBrkr", 

        'FireplaceQu':"no_fireplace", 'GarageType':"no_garage", 'GarageFinish':"no_garage",

        'GarageQual':"no_garage", 'GarageCond':"no_garage", 'Fence':"no_fence", 'MiscFeature':"None"} 



train_df.fillna(values_na_categorical, inplace=True)

train_df_category.fillna(values_na_categorical, inplace=True)



test_df.fillna(values_na_categorical, inplace=True)

test_df_category.fillna(values_na_categorical, inplace=True)
train_df_category.columns
# Assigning mode value to NA, where there is no desciption given in data_discription file



values_missing_categorical = train_df_category.columns



# combine=[train_df,test_df]



# for dataset in combine:

#     dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode())





for feature in values_missing_categorical:

    test_df_category[feature] = test_df_category[feature].fillna(train_df[feature].mode().iloc[0])

    test_df[feature] = test_df[feature].fillna(train_df[feature].mode().iloc[0])         
#Checking for any null values in both test and train category data



print(test_df_category.isna().sum().sort_values(ascending=False).head(),

train_df_category.isna().sum().sort_values(ascending=False).head())
#Just for fun, checking values in each feature



train_df_category_unique=pd.DataFrame(train_df_category.columns.values, columns=["Feature"])

for values in train_df_category_unique.Feature:

    train_df_category_unique.loc[train_df_category_unique.Feature==values, "unique"]=[[train_df_category[values].unique()]]

train_df_category_unique
#creating dataframe with categorical data only

train_df_numeric=train_df.select_dtypes(exclude="O")

test_df_numeric=test_df.select_dtypes(exclude="O")

train_df_numeric
corr_df=train_df_numeric.corr()



mask_df = np.zeros_like(corr_df, dtype=bool)

mask_df[np.triu_indices_from(mask_df)]=True



f,ax=plt.subplots(figsize=(15,12))



heatmap= sns.heatmap(corr_df, mask=mask_df, vmin=-1, vmax=1, square=True, cmap="coolwarm",

                     linewidths=1, annot=True, cbar_kws = {"shrink": 1, 

                                "ticks": [-1, -.5, 0, 0.5, 1]}, annot_kws={"size":8})
print(train_df_numeric.isna().sum().sort_values(ascending=False).head(),"\n\n",

test_df_numeric.isna().sum().sort_values(ascending=False).head())
#removing GaarageYrBlt as it is collinear with YearBuilt



train_df_numeric.drop(["GarageYrBlt"], axis=1, inplace=True)

test_df_numeric.drop(["GarageYrBlt"], axis=1, inplace=True)

train_df.drop(["GarageYrBlt"], axis=1, inplace=True)

test_df.drop(["GarageYrBlt"], axis=1, inplace=True)
#dummy is the avg of ratio of lotarea and lot frontage

dummy_Lot= (train_df_numeric["LotArea"]/train_df_numeric["LotFrontage"]).mean()

dummy_Lot



#filling LotFrontage value by using LotArea and dummy

train_df.LotFrontage.fillna(dummy_Lot, inplace=True)

train_df_numeric.LotFrontage.fillna(dummy_Lot, inplace=True)

test_df.LotFrontage.fillna(dummy_Lot, inplace=True)

test_df_numeric.LotFrontage.fillna(dummy_Lot, inplace=True)
#checking remaining null values



print(train_df_numeric.columns[train_df_numeric.isna().any()].tolist(),

      test_df_numeric.columns[test_df_numeric.isna().any()].tolist())
#filling remaing null values with mean as no of na values are very less



missing_numeric_features_train=train_df_numeric.columns



for feature in missing_numeric_features_train:

    train_df_numeric[feature].fillna(train_df_numeric[feature].mean(), inplace=True)

    train_df[feature].fillna(train_df[feature].mean(), inplace=True)





missing_numeric_features_test=test_df_numeric.columns[test_df_numeric.isna().any()].tolist()



for feature in missing_numeric_features_test:

    test_df_numeric[feature].fillna(train_df[feature].mean(), inplace=True)

    test_df[feature].fillna(train_df[feature].mean(), inplace=True)
print(train_df_numeric.isna().sum().sort_values(ascending=False).head(),"\n\n",

test_df_numeric.isna().sum().sort_values(ascending=False).head())
# train_df_numeric["Years"]= train_df_numeric["YearBuilt"]-train_df_numeric["GarageYrBlt"]
# Time between selling and building



train_df_numeric["Yr(S-B)"]=(train_df_numeric.YrSold-train_df_numeric.YearBuilt)*12+train_df_numeric.MoSold

train_df["Yr(S-B)"]=(train_df.YrSold-train_df.YearBuilt)*12+train_df.MoSold



test_df_numeric["Yr(S-B)"]=(test_df_numeric.YrSold-test_df_numeric.YearBuilt)*12+test_df_numeric.MoSold

test_df["Yr(S-B)"]=(test_df.YrSold-test_df.YearBuilt)*12+test_df.MoSold



# Time between seliing and remodelling



train_df_numeric["Yr(S-R)"]=(train_df_numeric.YrSold-train_df_numeric.YearRemodAdd)*12+train_df_numeric.MoSold

train_df["Yr(S-R)"]=(train_df.YrSold-train_df.YearRemodAdd)*12+train_df.MoSold



test_df_numeric["Yr(S-R)"]=(test_df_numeric.YrSold-test_df_numeric.YearRemodAdd)*12+test_df_numeric.MoSold

test_df["Yr(S-R)"]=(test_df.YrSold-test_df.YearRemodAdd)*12+test_df.MoSold



#removing unnecessary features



train_df.drop(["YrSold", "YearRemodAdd", "YearBuilt", "MoSold"], axis=1, inplace=True)

train_df_numeric.drop(["YrSold", "YearRemodAdd", "YearBuilt", "MoSold"], axis=1, inplace=True)

test_df.drop(["YrSold", "YearRemodAdd", "YearBuilt", "MoSold"], axis=1, inplace=True)

test_df_numeric.drop(["YrSold", "YearRemodAdd", "YearBuilt", "MoSold"], axis=1, inplace=True)
train_df_numeric.head()
corrmat=train_df_numeric.corr()
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice').index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
corr_df=train_df_numeric.corr()



mask_df = np.zeros_like(corr_df, dtype=bool)

mask_df[np.triu_indices_from(mask_df)]=True



f,ax=plt.subplots(figsize=(15,12))



heatmap= sns.heatmap(corr_df, mask=mask_df, vmin=-1, vmax=1, square=True, cmap="coolwarm",

                     linewidths=1, annot=True, cbar_kws = {"shrink": 1, 

                                "ticks": [-1, -.5, 0, 0.5, 1]}, annot_kws={"size":8})
train_df.shape, test_df.shape
# Seprating target value from independent variables



Y_train= train_df["SalePrice"]

train_df.drop("SalePrice", axis=1,inplace=True)

train_df_numeric.drop("SalePrice", axis=1,inplace=True)
#Applying scalar on numerical data



main_scalar=MinMaxScaler()

train_df[train_df_numeric.columns]=main_scalar.fit_transform(train_df_numeric)

test_df[train_df_numeric.columns]=main_scalar.transform(test_df_numeric)

train_df.head(2)
#crating single dataframe for dummies, so that they have same number of columns 

all_df=pd.concat([train_df,test_df]).reset_index(drop=True)

all_df.shape
#Creating dummies for categorical features



all_scaled=pd.get_dummies(all_df, drop_first=True)

print(train_df.shape, test_df.shape)
#Sepration both data from combined dataframe

train_df = pd.DataFrame(all_scaled[:1460])

test_df = pd.DataFrame(all_scaled[1460:2920])



print(train_df.shape, test_df.shape)
#Seprating given train data to Train Data and Validation Data

x_train, x_valid, y_train, y_valid= train_test_split(train_df, Y_train, test_size=0.1, random_state=20)
randomf_reg=RandomForestRegressor()

randomf_reg.fit(x_train,y_train)

randomf_reg.score(x_valid,y_valid)
dtree_reg=DecisionTreeRegressor()

dtree_reg.fit(x_train,y_train)

dtree_reg.score(x_valid,y_valid)
dtree_reg=DecisionTreeRegressor()

adaboost_reg= AdaBoostRegressor(dtree_reg, n_estimators=500, learning_rate=0.1)

adaboost_reg.fit(x_train,y_train)

adaboost_reg.score(x_valid,y_valid)
gboost_reg= GradientBoostingRegressor(random_state=0)

gboost_reg.fit(x_train,y_train)

gboost_reg.score(x_valid,y_valid)
xgb_reg= XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=1000,reg_alpha=0.001,reg_lambda=0.000001,n_jobs=-1,min_child_weight=3)

xgb_reg.fit(x_train,y_train)

xgb_reg.score(x_valid,y_valid)

LGBM = LGBMRegressor(n_estimators = 1000)

LGBM.fit(x_train,y_train)

LGBM.score(x_valid,y_valid)
# solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

# solution.to_csv("ridge_sol.csv", index = False)
#Submitting predictions from XGBoost Regression as it has highest score with validation data



predictions = xgb_reg.predict(test_df)



output = pd.DataFrame({'Id': test_Id, 'SalePrice': predictions})

output.to_csv('my_submission.csv', index=False)