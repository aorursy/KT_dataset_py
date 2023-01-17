
import numpy as np 
import pandas as pd 
%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import cross_val_score

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)
print("train has {} data | test has {} data".format(train.shape[0], test.shape[0]))
train.head()
test.head()
print("train has {} columns".format(len(train.columns)))
print("test has {} columns".format(len(test.columns)))
print("Target data is SalePrice")
# check unique data in train
for i in train.columns:
    print("{} has {} unique data".format(i, len(train[i].unique())))
#check missing data in train
missing_train={}
for i in train.columns:
    x = (len(train[train[i].isnull()]) / (train.shape[0])) *100
    if x !=0:
        print("{0} has {1:.2f}% missing data".format(i, (len(train[train[i].isnull()]) / train.shape[0]) *100)) 
        missing_train[i] = (len(train[train[i].isnull()]) / (train.shape[0]) *100)
#check missing data in train
missing_test ={}
for i in test.columns:
    x = (len(test[test[i].isnull()]) / (test.shape[0])) *100
    if x !=0:
        print("{0} has {1:.2f}% missing data".format(i, (len(test[test[i].isnull()]) / test.shape[0]) *100))
        missing_test[i] =(len(test[test[i].isnull()]) / (test.shape[0]) *100)
num = []
cat = []
for i in train.columns:
    if train[i].dtype == object:
        cat.append(i)
    else:
        num.append(i)
num
num.remove("Id")
num.remove("SalePrice")
cat[0:10]
train[num].head()
train.loc[train["Alley"].isnull(),"Alley"] = "None"
train["Alley"].value_counts()
test.loc[test["Alley"].isnull(),"Alley"] = "None"
test["Alley"].value_counts()
train["BsmtCond"].value_counts()
test["BsmtCond"].value_counts()
train.loc[train["BsmtCond"].isnull(), "BsmtCond"] = 'None'
test.loc[test["BsmtCond"].isnull(), "BsmtCond"] = 'None'
train["BsmtExposure"].value_counts()
test["BsmtFinType1"].value_counts()
train.loc[train["BsmtFinType1"].isnull(), "BsmtFinType1"] = 'None'
test.loc[test["BsmtFinType1"].isnull(), "BsmtFinType1"] = 'None'
train["BsmtFinType2"].value_counts()
test["BsmtFinType2"].value_counts()
train.loc[train["BsmtFinType2"].isnull(), "BsmtFinType2"] = 'None'
test.loc[test["BsmtFinType2"].isnull(), "BsmtFinType2"] = 'None'
train["BsmtQual"].value_counts()
test["BsmtQual"].value_counts()
train.loc[train["BsmtQual"].isnull(), "BsmtQual"] = 'None'
train.loc[train["BsmtQual"].isnull(), "BsmtQual"] = 'None'
train["Electrical"].value_counts()
test["Electrical"].value_counts()
train.loc[train["Electrical"].isnull(), "Electrical"] = train["Electrical"].mode()[0]
test.loc[test["Electrical"].isnull(), "Electrical"] = test["Electrical"].mode()[0]
train["Fence"].value_counts()
test["Fence"].value_counts()
train.loc[train["Fence"].isnull(), "Fence"] = "None"
test.loc[test["Fence"].isnull(), "Fence"] = "None"
train["FireplaceQu"].value_counts()
test["FireplaceQu"].value_counts()
train.loc[train["FireplaceQu"].isnull(), "FireplaceQu"] = "None"
test.loc[test["FireplaceQu"].isnull(), "FireplaceQu"] = "None"
train["GarageCond"].value_counts()
train["GarageFinish"].value_counts()
train["GarageQual"].value_counts()
train["GarageType"].value_counts()
train.loc[train["GarageCond"].isnull(), "GarageCond"] = "None"
train.loc[train["GarageFinish"].isnull(), "GarageFinish"] = "None"
train.loc[train["GarageQual"].isnull(), "GarageQual"] = "None"
train.loc[train["GarageType"].isnull(), "GarageType"] = "None"
train.loc[train["GarageYrBlt"].isnull(), "GarageYrBlt"] = 0
test["GarageCond"].value_counts()
test.loc[test["GarageCond"].isnull(), "GarageCond"] = "None"
test.loc[test["GarageFinish"].isnull(), "GarageFinish"] = "None"
test.loc[test["GarageQual"].isnull(), "GarageQual"] = "None"
test.loc[test["GarageType"].isnull(), "GarageType"] = "None"
test.loc[test["GarageYrBlt"].isnull(), "GarageYrBlt"] = 0
train.loc[train["LotFrontage"]==0, "LotFrontage"] = round(train["LotFrontage"].mean())
test.loc[test["LotFrontage"]== 0, "LotFrontage"] = round(test["LotFrontage"].mean())
train.loc[train["MasVnrArea"].isnull(),"MasVnrArea"] = 0
train.loc[train["MasVnrType"].isnull(),"MasVnrType"] ="None"
test.loc[test["MasVnrArea"].isnull(),"MasVnrArea"] = 0
test.loc[test["MasVnrType"].isnull(),"MasVnrType"] ="None"
train.loc[train["MiscFeature"].isnull(),"MiscFeature"] ="None"
test.loc[test["MiscFeature"].isnull(),"MiscFeature"] ="None"
train.loc[train["PoolQC"].isnull(),"PoolQC"] ="None"
test.loc[test["PoolQC"].isnull(),"PoolQC"] ="None"
test["MSZoning"].value_counts()
test.loc[test["MSZoning"].isnull(),"MSZoning"] =test["MSZoning"].mode()[0]
test["Utilities"].value_counts()
test.loc[test["Utilities"].isnull(),"Utilities"] ="None"
test.loc[test["Exterior1st"].isnull(),"Exterior1st"] ="None"
test.loc[test["Exterior2nd"].isnull(),"Exterior2nd"] ="None"
test.loc[test["BsmtFinSF1"].isnull(),"BsmtFinSF1"] = 0
test.loc[test["BsmtFinSF2"].isnull(),"BsmtFinSF2"] = 0
test.loc[test["BsmtUnfSF"].isnull(),"BsmtUnfSF"] = 0
test.loc[test["TotalBsmtSF"].isnull(),"TotalBsmtSF"] = 0
test.loc[test["BsmtFullBath"].isnull(),"BsmtFullBath"] = 0
test.loc[test["BsmtHalfBath"].isnull(),"BsmtHalfBath"] = 0
test.loc[test["GarageCars"].isnull(),"GarageCars"] = 0
test.loc[test["GarageArea"].isnull(),"GarageArea"] = 0
test.loc[test["KitchenQual"].isnull(),"KitchenQual"] ="None"
test.loc[test["Functional"].isnull(),'Functional'] = test["Functional"].mode()[0]
test.loc[test["SaleType"].isnull(),'SaleType'] = test["SaleType"].mode()[0]
cat_train = train[cat]
cat_train.shape
cat_test = test[cat]
cat_all = pd.concat([cat_train, cat_test], axis= 0)
# one_hot_encoding
cat_dum_all = pd.get_dummies(cat_all)
cat_dum_all.shape
cat_dum_train = cat_dum_all[0:1460]
cat_dum_train.shape
cat_dum_test = cat_dum_all[1460:2921]
cat_dum_test.shape
train.drop(cat_train,inplace=True, axis =1)
train_dum = pd.concat([train,cat_dum_train], axis=1)
train_dum.head()
test.drop(cat_test,inplace=True, axis =1)
test_dum = pd.concat([test,cat_dum_test], axis=1)
test_dum.head()
figure, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8),(ax9,ax10,ax11,ax12),
        (ax13,ax14,ax15,ax16),(ax17,ax18,ax19,ax20),(ax21,ax22,ax23,ax24),
        (ax25,ax26,ax27,ax28),(ax29,ax30,ax31,ax32),(ax33,ax34,ax35,ax36)) =plt.subplots(nrows=9, ncols =4)
figure.set_size_inches(20,20)
sns.pointplot(train["MSSubClass"],train["SalePrice"],ax=ax1)
sns.pointplot(train["LotFrontage"],train["SalePrice"],ax=ax2)
sns.pointplot(train["LotArea"],train["SalePrice"],ax=ax3)
sns.pointplot(train["OverallQual"],train["SalePrice"],ax=ax4)
sns.pointplot(train["OverallCond"],train["SalePrice"],ax=ax5)
sns.pointplot(train["YearBuilt"],train["SalePrice"],ax=ax6)
sns.pointplot(train["YearRemodAdd"],train["SalePrice"],ax=ax7)
sns.pointplot(train["MasVnrArea"],train["SalePrice"],ax=ax8)
sns.pointplot(train["BsmtFinSF1"],train["SalePrice"],ax=ax9)
sns.pointplot(train["BsmtFinSF2"],train["SalePrice"],ax=ax10)
sns.pointplot(train["BsmtUnfSF"],train["SalePrice"],ax=ax11)
sns.pointplot(train["TotalBsmtSF"],train["SalePrice"],ax=ax12)
sns.pointplot(train["1stFlrSF"],train["SalePrice"],ax=ax13)
sns.pointplot(train["2ndFlrSF"],train["SalePrice"],ax=ax14)
sns.pointplot(train["LowQualFinSF"],train["SalePrice"],ax=ax15)
sns.pointplot(train["GrLivArea"],train["SalePrice"],ax=ax16)
sns.pointplot(train["BsmtFullBath"],train["SalePrice"],ax=ax17)
sns.pointplot(train["BsmtHalfBath"],train["SalePrice"],ax=ax18)
sns.pointplot(train["FullBath"],train["SalePrice"],ax=ax19)
sns.pointplot(train["HalfBath"],train["SalePrice"],ax=ax20)
sns.pointplot(train["BedroomAbvGr"],train["SalePrice"],ax=ax21)
sns.pointplot(train["KitchenAbvGr"],train["SalePrice"],ax=ax22)
sns.pointplot(train["TotRmsAbvGrd"],train["SalePrice"],ax=ax23)
sns.pointplot(train["Fireplaces"],train["SalePrice"],ax=ax24)
sns.pointplot(train["GarageYrBlt"],train["SalePrice"],ax=ax25)
sns.pointplot(train["GarageCars"],train["SalePrice"],ax=ax26)
sns.pointplot(train["GarageArea"],train["SalePrice"],ax=ax27)
sns.pointplot(train["WoodDeckSF"],train["SalePrice"],ax=ax28)
sns.pointplot(train["OpenPorchSF"],train["SalePrice"],ax=ax29)
sns.pointplot(train["EnclosedPorch"],train["SalePrice"],ax=ax30)
sns.pointplot(train["3SsnPorch"],train["SalePrice"],ax=ax31)
sns.pointplot(train["ScreenPorch"],train["SalePrice"],ax=ax32)
sns.pointplot(train["PoolArea"],train["SalePrice"],ax=ax33)
sns.pointplot(train["MiscVal"],train["SalePrice"],ax=ax34)
sns.pointplot(train["MoSold"],train["SalePrice"],ax=ax35)
sns.pointplot(train["YrSold"],train["SalePrice"],ax=ax36)

#train
train_dum["OverallQual-s2"] = train_dum["OverallQual"] ** 2
train_dum["OverallQual-s3"] = train_dum["OverallQual"] ** 3
train_dum["OverallQual-Sq"] = np.sqrt(train_dum["OverallQual"])


train_dum["OverallCond-s2"] = train_dum["OverallCond"] ** 2
train_dum["OverallCond-s3"] = train_dum["OverallCond"] ** 3
train_dum["OverallCond-Sq"] = np.sqrt(train_dum["OverallCond"])

train_dum["YearRemodAdd-s2"] = train_dum["YearRemodAdd"] ** 2
train_dum["YearRemodAdd-s3"] = train_dum["YearRemodAdd"] ** 3
train_dum["YearRemodAdd-Sq"] = np.sqrt(train_dum["YearRemodAdd"])

train_dum["FullBath-s2"] = train_dum["FullBath"] ** 2
train_dum["FullBath-s3"] = train_dum["FullBath"] ** 3
train_dum["FullBath-Sq"] = np.sqrt(train_dum["FullBath"])

train_dum["TotRmsAbvGrd-s2"] = train_dum["TotRmsAbvGrd"] ** 2
train_dum["TotRmsAbvGrd-s3"] = train_dum["TotRmsAbvGrd"] ** 3
train_dum["TotRmsAbvGrd-Sq"] = np.sqrt(train_dum["TotRmsAbvGrd"])


train_dum["Fireplaces-s2"] = train_dum["Fireplaces"] ** 2
train_dum["Fireplaces-s3"] = train_dum["Fireplaces"] ** 3
train_dum["Fireplaces-Sq"] = np.sqrt(train_dum["Fireplaces"])


#train
test_dum["OverallQual-s2"] = test_dum["OverallQual"] ** 2
test_dum["OverallQual-s3"] = test_dum["OverallQual"] ** 3
test_dum["OverallQual-Sq"] = np.sqrt(test_dum["OverallQual"])


test_dum["OverallCond-s2"] = test_dum["OverallCond"] ** 2
test_dum["OverallCond-s3"] = test_dum["OverallCond"] ** 3
test_dum["OverallCond-Sq"] = np.sqrt(test_dum["OverallCond"])

test_dum["YearRemodAdd-s2"] = test_dum["YearRemodAdd"] ** 2
test_dum["YearRemodAdd-s3"] = test_dum["YearRemodAdd"] ** 3
test_dum["YearRemodAdd-Sq"] = np.sqrt(test_dum["YearRemodAdd"])

test_dum["FullBath-s2"] = test_dum["FullBath"] ** 2
test_dum["FullBath-s3"] = test_dum["FullBath"] ** 3
test_dum["FullBath-Sq"] = np.sqrt(test_dum["FullBath"])

test_dum["TotRmsAbvGrd-s2"] = test_dum["TotRmsAbvGrd"] ** 2
test_dum["TotRmsAbvGrd-s3"] = test_dum["TotRmsAbvGrd"] ** 3
test_dum["TotRmsAbvGrd-Sq"] = np.sqrt(test_dum["TotRmsAbvGrd"])


test_dum["Fireplaces-s2"] = test_dum["Fireplaces"] ** 2
test_dum["Fireplaces-s3"] = test_dum["Fireplaces"] ** 3
test_dum["Fireplaces-Sq"] = np.sqrt(test_dum["Fireplaces"])

add = ["OverallQual-s2","OverallQual-s3","OverallQual-Sq","OverallCond-s2","OverallCond-s3","OverallCond-Sq",
 "YearRemodAdd-s2","YearRemodAdd-s3","YearRemodAdd-Sq","FullBath-s2","FullBath-s3","FullBath-Sq",
 "TotRmsAbvGrd-s2","TotRmsAbvGrd-s3","TotRmsAbvGrd-Sq","Fireplaces-s2","Fireplaces-s3","Fireplaces-Sq"]
num = num + add
train_dum.loc[train_dum["LotFrontage"].isnull(),"LotFrontage"] = train_dum["LotFrontage"].mean()
test_dum.loc[test_dum["LotFrontage"].isnull(),"LotFrontage"] = test_dum["LotFrontage"].mean()
scaler = MinMaxScaler()
train_dum[num] = scaler.fit_transform(train_dum[num])
test_dum.loc[test_dum["BsmtFinSF1"].isnull(),"BsmtFinSF1"] =0
test_dum.loc[test_dum["BsmtFinSF2"].isnull(),"BsmtFinSF2"] = 0
test_dum.loc[test_dum["BsmtUnfSF"].isnull(),"BsmtUnfSF"] = 0
test_dum.loc[test_dum["TotalBsmtSF"].isnull(),"TotalBsmtSF"] =0
test_dum[num] = scaler.fit_transform(test_dum[num])
x_train = train_dum.drop("SalePrice",axis=1)
x_train.set_index("Id",inplace=True)
x_test = test_dum.set_index("Id")
y_train = train["SalePrice"]
y_train = np.log(y_train)
select_feat = ['LotArea',
 'GrLivArea',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 'GarageArea',
 'BsmtFinSF1',
 'MasVnrArea',
 'WoodDeckSF',
 '2ndFlrSF',
 'OpenPorchSF',
 'YearRemodAdd',
 'YearBuilt',
 'LotFrontage',
 'GarageYrBlt',
 'MoSold',
 'YrSold',
 'OverallQual',
 'TotRmsAbvGrd',
 'MSSubClass',
 'OverallCond',
 'FireplaceQu_TA',
 'HeatingQC_TA',
 'BedroomAbvGr',
 'Fireplaces',
 'LotShape_IR1',
 'BsmtExposure_No',
 'GarageFinish_Fin',
 'GarageFinish_RFn',
 'EnclosedPorch',
 'GarageFinish_Unf',
 'MasVnrType_BrkFace',
 'BsmtQual_Gd',
 'KitchenQual_TA',
 'LotShape_Reg',
 'KitchenQual_Gd',
 'Foundation_CBlock',
 'FullBath',
 'BsmtFinType1_GLQ',
 'HeatingQC_Gd',
 'BsmtFullBath',
 'BsmtFinType1_ALQ',
 'LotConfig_Inside',
 'HeatingQC_Ex',
 'HalfBath',
 'GarageType_Detchd',
 'Neighborhood_CollgCr',
 'FireplaceQu_Gd',
 'LotConfig_Corner',
 'BsmtExposure_Av']

x_train = x_train[select_feat]
x_test = x_test[select_feat]
model = xgb.XGBRegressor(nthread = 4,learning_rate =0.038264, reg_alpha =0.203944,
                        n_estimators=276)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
predictions = model.predict(x_test)
predictions = np.exp(predictions)
x_train = x_train[select_feat]
x_test = x_test[select_feat]
model = xgb.XGBRegressor(nthread = 4,learning_rate =0.038264, reg_alpha =0.203944,
                        n_estimators=276)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
predictions = np.exp(predictions)
predictions[0:10]
predictions[0:100]
submit = pd.read_csv("../input/sample_submission.csv")
submit["SalePrice"] = predictions
submit.set_index("Id",inplace=True)
submit.to_csv("submit.csv")
