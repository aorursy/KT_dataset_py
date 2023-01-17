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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from IPython.display import display

import category_encoders as ce

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error

import xgboost as xgb

%matplotlib inline
df_house = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

df_house_test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")

df_house.head()
print(df_house.columns.values[0])

for i in range(80):

    print(df_house.columns.values[i], len(pd.unique(df_house[df_house.columns.values[i]])))
plt.scatter(df_house["GrLivArea"], df_house["SalePrice"], alpha=0.9)
df_house = df_house[df_house["GrLivArea"]<4200]
plt.hist(df_house["SalePrice"],bins=40)
df_house["SalePrice"]=np.log(df_house["SalePrice"])

plt.hist(df_house["SalePrice"],bins=40)
pd.set_option("display.max_rows", 300)

print(df_house.isnull().sum())
print(df_house_test.columns.values[0])

for i in range(80):

    print(df_house_test.columns.values[i], len(pd.unique(df_house_test[df_house_test.columns.values[i]])))
print(df_house_test.isnull().sum())
df_house.groupby("MSSubClass").count()
df_house_test.groupby("MSSubClass").count()
df_house["Alley"].fillna("Nothing", inplace=True)

df_house["MasVnrType"].fillna("None", inplace=True)

df_house["MasVnrArea"].fillna(0,inplace=True)

df_house["Electrical"].fillna("Mix", inplace=True)

df_house["GarageYrBlt"].fillna(df_house["YearBuilt"], inplace=True)



df_house_test["Alley"].fillna("Nothing", inplace=True)

df_house_test["MasVnrType"].fillna("None", inplace=True)

df_house_test["MasVnrArea"].fillna(0,inplace=True)

df_house_test["Electrical"].fillna("Mix", inplace=True)

df_house_test["GarageYrBlt"].fillna(df_house_test["YearBuilt"], inplace=True)

df_house.loc[df_house["BsmtExposure"].isnull() & df_house["BsmtCond"].notnull(),'BsmtExposure']="Av"

df_house_test.loc[df_house_test["BsmtExposure"].isnull() & df_house_test["BsmtCond"].notnull(),'BsmtExposure']="Av"

df_house.loc[df_house["BsmtFinType2"].isnull() & df_house["BsmtCond"].notnull(), 'BsmtFinType2' ] ="Rec"

df_house_test.loc[df_house_test["BsmtFinType2"].isnull() & df_house_test["BsmtCond"].notnull(), 'BsmtFinType2' ] ="Rec"

df_house["BsmtQual"].fillna("Nothing", inplace=True)

df_house["BsmtCond"].fillna("Nothing", inplace=True)

df_house["BsmtExposure"].fillna("Nothing", inplace=True)

df_house["BsmtFinType1"].fillna("Nothing", inplace=True)

df_house["BsmtFinType2"].fillna("Nothing", inplace=True)

df_house["FireplaceQu"].fillna("Nothing", inplace=True)

df_house["GarageType"].fillna("Nothing", inplace=True)

df_house["GarageFinish"].fillna("Nothing", inplace=True)

df_house["GarageQual"].fillna("Nothing", inplace=True)

df_house["GarageCond"].fillna("Nothing", inplace=True)

df_house["PoolQC"].fillna("Nothing", inplace=True)

df_house["Fence"].fillna("Nothing", inplace=True)

df_house["MiscFeature"].fillna("Nothing", inplace=True)



df_house_test["BsmtQual"].fillna("Nothing", inplace=True)

df_house_test["BsmtCond"].fillna("Nothing", inplace=True)

df_house_test["BsmtExposure"].fillna("Nothing", inplace=True)

df_house_test["BsmtFinType1"].fillna("Nothing", inplace=True)

df_house_test["BsmtFinType2"].fillna("Nothing", inplace=True)

df_house_test["FireplaceQu"].fillna("Nothing", inplace=True)

df_house_test["GarageType"].fillna("Nothing", inplace=True)

df_house_test["GarageFinish"].fillna("Nothing", inplace=True)

df_house_test["GarageQual"].fillna("Nothing", inplace=True)

df_house_test["GarageCond"].fillna("Nothing", inplace=True)

df_house_test["PoolQC"].fillna("Nothing", inplace=True)

df_house_test["Fence"].fillna("Nothing", inplace=True)

df_house_test["MiscFeature"].fillna("Nothing", inplace=True)

df_house_test["MSZoning"].fillna(df_house_test["MSZoning"].mode().iloc[0], inplace=True)

df_house_test["Utilities"].fillna(df_house_test["Utilities"].mode().iloc[0], inplace=True)

df_house_test["Exterior1st"].fillna(df_house_test["Exterior1st"].mode().iloc[0], inplace=True)

df_house_test["Exterior2nd"].fillna(df_house_test["Exterior2nd"].mode().iloc[0], inplace=True)

df_house_test["BsmtFinSF1"].fillna(0, inplace=True)

df_house_test["BsmtFinSF2"].fillna(0, inplace=True)

df_house_test["BsmtUnfSF"].fillna(0, inplace=True)

df_house_test["TotalBsmtSF"].fillna(0, inplace=True)

df_house_test["BsmtFullBath"].fillna(0, inplace=True)

df_house_test["BsmtHalfBath"].fillna(0, inplace=True)

df_house_test["KitchenQual"].fillna(df_house_test["KitchenQual"].mode().iloc[0], inplace=True)

df_house_test["Functional"].fillna(df_house_test["Functional"].mode().iloc[0], inplace=True)

df_house_test["GarageCars"].fillna(df_house_test["GarageCars"].mode().iloc[0], inplace=True)

df_house_test["GarageArea"].fillna(df_house_test["GarageArea"].mean(), inplace=True)

df_house_test["SaleType"].fillna(df_house_test["SaleType"].mode().iloc[0], inplace=True)
print(df_house_test.isnull().sum())
print(df_house.isnull().sum())
te_col=["MSSubClass", "MSZoning", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2",

            "BldgType", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation",

            "Heating", "Electrical", "PavedDrive", "MiscFeature", "SaleType", "SaleCondition"]
ordinal_col=["Alley", "LotShape", "Utilities", "LandSlope", "HouseStyle", "ExterQual", "ExterCond",

            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC",

            "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",

            "GarageCond", "PoolQC", "Fence"]



ordinal_mapping = [{'col':'Alley', 'mapping':{"Nothing":0, 'Pave':1, 'Grvl':2}},

                   {'col':'LotShape', 'mapping':{'IR3':0, 'IR2':1, 'IR1':2, 'Reg':3}},

                   {'col':'Utilities', 'mapping':{'ELO':0, 'NoSeWa':1, 'NoSewr':2, 'AllPub':3}},

                   {'col':'LandSlope', 'mapping':{'Sev':0, 'Mod':1, 'Gtl':2}},

                   {'col':'HouseStyle', 'mapping':{'1Story':0, '1.5Unf':1, '1.5Fin':2, '2Story':3, '2.5Unf':4, '2.5Fin':5, 'SFoyer':6, 'SLvl':7}},

                   {'col':'ExterQual', 'mapping':{'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}},

                   {'col':'ExterCond', 'mapping':{'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}},

                   {'col':'BsmtQual', 'mapping':{"Nothing":0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}},

                   {'col':'BsmtCond', 'mapping':{"Nothing":0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}},

                   {'col':'BsmtExposure', 'mapping':{"Nothing":0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}},

                   {'col':'BsmtFinType1', 'mapping':{"Nothing":0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}},

                   {'col':'BsmtFinType2', 'mapping':{"Nothing":0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}},

                   {'col':'HeatingQC', 'mapping':{'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}},

                   {'col':'KitchenQual', 'mapping':{'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}},

                   {'col':'Functional', 'mapping':{'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7}},

                   {'col':'FireplaceQu', 'mapping':{"Nothing":0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}},

                   {'col':'GarageType', 'mapping':{"Nothing":0, 'Detchd':1, 'CarPort':2, 'BuiltIn':3, 'Basment':4, 'Attchd':5, '2Types':6}},

                   {'col':'GarageFinish', 'mapping':{"Nothing":0, 'Unf':1, 'RFn':2, 'Fin':3}},

                   {'col':'GarageQual', 'mapping':{"Nothing":0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}},

                   {'col':'GarageCond', 'mapping':{"Nothing":0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}},

                   {'col':'PoolQC', 'mapping':{"Nothing":0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}},

                   {'col':'Fence', 'mapping':{"Nothing":0, 'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4}}

                  ]   
ce_te = ce.TargetEncoder(cols=te_col, handle_unknown='impute')

df_house1 = ce_te.fit_transform(X=df_house, y=df_house["SalePrice"])

df_house_test1 = df_house_test

# trainとtestの列をそろえるために、便宜的にダミー列を追加

df_house_test1["dummy"]=df_house["SalePrice"]

df_house_test1 = ce_te.transform(X=df_house_test)

df_house1.head()
df_house_test1.head()
print(df_house1.isnull().sum())
print(df_house_test1.isnull().sum())
ce_oe = ce.OrdinalEncoder(cols=ordinal_col, mapping=ordinal_mapping, handle_unknown='impute')

ce_oe.fit(df_house1)

df_house2 = ce_oe.transform(df_house1)

df_house_test2 = ce_oe.transform(df_house_test1)

print(df_house2.isnull().sum())
df_house3=df_house2

df_house3.loc[df_house3["Street"]=="Grvl",'Street']=0

df_house3.loc[df_house3["Street"]=="Pave",'Street']=1

df_house3.loc[df_house3["CentralAir"]=="N",'CentralAir']=0

df_house3.loc[df_house3["CentralAir"]=="Y",'CentralAir']=1



df_house_test3=df_house_test2

df_house_test3.loc[df_house_test3["Street"]=="Grvl",'Street']=0

df_house_test3.loc[df_house_test3["Street"]=="Pave",'Street']=1

df_house_test3.loc[df_house_test3["CentralAir"]=="N",'CentralAir']=0

df_house_test3.loc[df_house_test3["CentralAir"]=="Y",'CentralAir']=1
df_house3=df_house3.set_index("Id")



df_house_test3=df_house_test3.set_index("Id")



df_house3.head()
print(df_house3.isnull().sum())
mean = df_house3["LotFrontage"].mean()

count = df_house3["LotFrontage"].count()

mean_test =df_house_test3["LotFrontage"].mean()

count_test = df_house_test3["LotFrontage"].count()

LotFrontage_average = (mean*count+mean_test*count_test)/(count+count_test)

print(LotFrontage_average)
df_house4 = df_house3.fillna({'LotFrontage':LotFrontage_average})

df_house_test4 = df_house_test3.fillna({'LotFrontage':LotFrontage_average})

pd.set_option("display.max_rows", 300)

print(df_house4.isnull().sum())
print(df_house_test4.isnull().sum())
df_house_test4["MSSubClass"].fillna(df_house_test["MSSubClass"].mean(), inplace=True)

print(df_house_test4.isnull().sum())
price=df_house4["SalePrice"].values

df_house5=df_house4

del df_house5["SalePrice"]

del df_house_test4["dummy"]

feature = df_house5.values

feature_test = df_house_test4.values

df_house5.head()
df_house_test4.head()
price.shape
feature.shape
feature_test.shape
X_train, X_test, y_train, y_test = train_test_split(feature, price, random_state=0)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
mod = xgb.XGBRegressor(eval_metric="mae",

                       learning_rate=0.01,

                       max_depth=2,

                       n_estimators=3460,

                       subsample=0.8,

                       colsample_bytree=0.8,

                       reg_lambda=0.5,

                       reg_alpha=0.5,

                       objective='reg:squarederror',

                       random_state=42)

mod.fit(X_train, y_train)

print("train score:{:.3f}" .format(mod.score(X_train, y_train)))

print("test score :{:.3f}" .format(mod.score(X_test, y_test)))

preds_XGBR = mod.predict(X_test)

print(mean_absolute_error(np.exp(y_test), np.exp(preds_XGBR)))
preds_XGBR=mod.predict(X_test)

#preds=grid.predict(feature_test)

plt.plot(preds_XGBR)

plt.plot(y_test)
plt.scatter(y_test, preds_XGBR)
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PolynomialFeatures

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor
ridge = make_pipeline(RobustScaler(), RidgeCV(cv=5))

ridge.fit(X_train, y_train)

print("train score:{:.3f}" .format(ridge.score(X_train, y_train)))

print("test score :{:.3f}" .format(ridge.score(X_test, y_test)))

preds_ridge=ridge.predict(X_test)

print(mean_absolute_error(np.exp(y_test), np.exp(preds_ridge)))
plt.plot(preds_ridge)

plt.plot(y_test)
plt.scatter(y_test, preds_ridge)
lasso = make_pipeline(RobustScaler(), LassoCV(cv=5))

lasso.fit(X_train, y_train)

print("train score:{:.3f}" .format(lasso.score(X_train, y_train)))

print("test score :{:.3f}" .format(lasso.score(X_test, y_test)))

preds_lasso=lasso.predict(X_test)

print(mean_absolute_error(np.exp(y_test), np.exp(preds_lasso)))
plt.plot(preds_lasso)

plt.plot(y_test)
plt.scatter(y_test, preds_lasso)
gbr = GradientBoostingRegressor(learning_rate=0.02, n_estimators=3000,

                                max_depth=2, min_samples_split=2,

                                loss='ls', max_features=35)

gbr.fit(X_train, y_train)

print("train score:{:.3f}" .format(gbr.score(X_train, y_train)))

print("test score :{:.3f}" .format(gbr.score(X_test, y_test)))

preds_gbr=gbr.predict(X_test)

print(mean_absolute_error(np.exp(y_test), np.exp(preds_gbr)))
plt.plot(preds_gbr)

plt.plot(y_test)
plt.scatter(y_test, preds_gbr)
# 各モデルの価格帯別性能評価



pred_xgbr_exp=np.exp(mod.predict(X_test))

pred_ridge_exp=np.exp(ridge.predict(X_test))

pred_lasso_exp=np.exp(lasso.predict(X_test))

pred_gbr_exp=np.exp(gbr.predict(X_test))

y_test_exp = np.exp(y_test)

average_pred = (pred_xgbr_exp + pred_ridge_exp + pred_lasso_exp + pred_gbr_exp)/4



pred_price_exp = np.stack([y_test_exp,

                       pred_xgbr_exp,

                       pred_ridge_exp,

                       pred_lasso_exp,

                       pred_gbr_exp,

                       average_pred,

                          ],axis=1)

pred_price_exp.shape
print(np.max(pred_price_exp[:,5]))

print(np.max(pred_price_exp[:,0]))
th1 = 100000

th2 = 140000

th3 = 180000

th4 = 220000

th5 = 300000





price_band_10 = pred_price_exp[(pred_price_exp[:,5]<th1)]

price_band_1015 = pred_price_exp[(pred_price_exp[:,5]>=th1) & (pred_price_exp[:,5]<th2)]

price_band_1520 = pred_price_exp[(pred_price_exp[:,5]>=th2) & (pred_price_exp[:,5]<th3)]

price_band_2025 = pred_price_exp[(pred_price_exp[:,5]>=th3) & (pred_price_exp[:,5]<th4)]

price_band_2535 = pred_price_exp[(pred_price_exp[:,5]>=th4) & (pred_price_exp[:,5]<th5)]

price_band_35 = pred_price_exp[(pred_price_exp[:,5]>=th5)]

print(price_band_10.shape)

print(price_band_1015.shape)

print(price_band_1520.shape)

print(price_band_2025.shape)

print(price_band_2535.shape)

print(price_band_35.shape)
mae_xgbr=[]

mae_ridge=[]

mae_lasso=[]

mae_gbr=[]



mae_xgbr.append(mean_absolute_error(price_band_10[:,0], price_band_10[:,1]))

mae_xgbr.append(mean_absolute_error(price_band_1015[:,0], price_band_1015[:,1]))

mae_xgbr.append(mean_absolute_error(price_band_1520[:,0], price_band_1520[:,1]))

mae_xgbr.append(mean_absolute_error(price_band_2025[:,0], price_band_2025[:,1]))

mae_xgbr.append(mean_absolute_error(price_band_2535[:,0], price_band_2535[:,1]))

mae_xgbr.append(mean_absolute_error(price_band_35[:,0], price_band_35[:,1]))



mae_ridge.append(mean_absolute_error(price_band_10[:,0], price_band_10[:,2]))

mae_ridge.append(mean_absolute_error(price_band_1015[:,0], price_band_1015[:,2]))

mae_ridge.append(mean_absolute_error(price_band_1520[:,0], price_band_1520[:,2]))

mae_ridge.append(mean_absolute_error(price_band_2025[:,0], price_band_2025[:,2]))

mae_ridge.append(mean_absolute_error(price_band_2535[:,0], price_band_2535[:,2]))

mae_ridge.append(mean_absolute_error(price_band_35[:,0], price_band_35[:,2]))



mae_lasso.append(mean_absolute_error(price_band_10[:,0], price_band_10[:,3]))

mae_lasso.append(mean_absolute_error(price_band_1015[:,0], price_band_1015[:,3]))

mae_lasso.append(mean_absolute_error(price_band_1520[:,0], price_band_1520[:,3]))

mae_lasso.append(mean_absolute_error(price_band_2025[:,0], price_band_2025[:,3]))

mae_lasso.append(mean_absolute_error(price_band_2535[:,0], price_band_2535[:,3]))

mae_lasso.append(mean_absolute_error(price_band_35[:,0], price_band_35[:,3]))



mae_gbr.append(mean_absolute_error(price_band_10[:,0], price_band_10[:,4]))

mae_gbr.append(mean_absolute_error(price_band_1015[:,0], price_band_1015[:,4]))

mae_gbr.append(mean_absolute_error(price_band_1520[:,0], price_band_1520[:,4]))

mae_gbr.append(mean_absolute_error(price_band_2025[:,0], price_band_2025[:,4]))

mae_gbr.append(mean_absolute_error(price_band_2535[:,0], price_band_2535[:,4]))

mae_gbr.append(mean_absolute_error(price_band_35[:,0], price_band_35[:,4]))



print(mae_xgbr)

print(mae_ridge)

print(mae_lasso)

print(mae_gbr)

price=[th1,th2,th3,th4,th5,np.max(average_pred)]
fig = plt.figure(figsize=(10,5)) 

ax = fig.add_subplot(111)

ax.plot(price, mae_xgbr, label="xgbr")

ax.plot(price, mae_ridge, label="ridge")

ax.plot(price, mae_lasso, label="lasso")

ax.plot(price, mae_gbr, label="gbr")

ax.legend()
pred_xgbr_exp=np.exp(mod.predict(X_test))

pred_ridge_exp=np.exp(ridge.predict(X_test))

pred_lasso_exp=np.exp(lasso.predict(X_test))

pred_gbr_exp=np.exp(gbr.predict(X_test))



a1=1.0  # xgbr

b1=0.0  # ridge

c1=0.0  # lasso

d1=0.0  # gbr



a2=1.0  # xgbr

b2=1.0  # ridge

c2=1.0  # lasso  

d2=1.0  # gbr



a3=1.0  # xgbr

b3=0.0  # ridge

c3=0.0  # lasso

d3=1.0  # gbr



a4=1.0  # xgbr

b4=0.0  # ridge

c4=0.0  # lasso

d4=1.0  # gbr



a5=0.5  # xgbr

b5=1.0  # ridge

c5=0.0  # lasso

d5=0.5  # gbr



a6=1.0  # xgbr

b6=0.0  # ridge

c6=0.0  # lasso

d6=0.0  # gbr



average_pred = (pred_xgbr_exp + pred_ridge_exp + pred_lasso_exp + pred_gbr_exp)/4

blend_pred = average_pred

blend_pred = np.where(average_pred < th1,

                      (a1*pred_xgbr_exp + b1*pred_ridge_exp + c1*pred_lasso_exp + d1*pred_gbr_exp)/(a1+b1+c1+d1), blend_pred)

blend_pred = np.where((average_pred >=th1) & (average_pred < th2),

                      (a2*pred_xgbr_exp + b2*pred_ridge_exp + c2*pred_lasso_exp + d2*pred_gbr_exp)/(a2+b2+c2+d2),blend_pred)

blend_pred = np.where((average_pred >=th2) & (average_pred < th3), 

                      (a3*pred_xgbr_exp + b3*pred_ridge_exp + c3*pred_lasso_exp + d3*pred_gbr_exp)/(a3+b3+c3+d3),blend_pred)

blend_pred = np.where((average_pred >=th3) & (average_pred < th4),

                      (a4*pred_xgbr_exp + b4*pred_ridge_exp + c4*pred_lasso_exp + d4*pred_gbr_exp)/(a4+b4+c4+d4),blend_pred)

blend_pred = np.where((average_pred >=th4) & (average_pred < th5),

                      (a5*pred_xgbr_exp + b5*pred_ridge_exp + c5*pred_lasso_exp + d5*pred_gbr_exp)/(a5+b5+c5+d5),blend_pred)

blend_pred = np.where(average_pred >= th5, 

                      (a6*pred_xgbr_exp + b6*pred_ridge_exp + c6*pred_lasso_exp + d6*pred_gbr_exp)/(a6+b6+c6+d6), blend_pred)



print("XGBR mae {:.3f}".format(mean_absolute_error(np.exp(y_test), pred_xgbr_exp)))

print("ridge mae {:.3f}".format(mean_absolute_error(np.exp(y_test), pred_ridge_exp)))

print("lasso mae {:.3f}".format(mean_absolute_error(np.exp(y_test), pred_lasso_exp)))

print("gbr mae {:.3f}".format(mean_absolute_error(np.exp(y_test), pred_gbr_exp)))

print("blend mae {:.3f}".format(mean_absolute_error(np.exp(y_test), blend_pred)))
blend_pred.shape
pred_xgbr_exp=np.exp(mod.predict(feature_test))

pred_ridge_exp=np.exp(ridge.predict(feature_test))

pred_lasso_exp=np.exp(lasso.predict(feature_test))

pred_gbr_exp=np.exp(gbr.predict(feature_test))

average_pred = (pred_xgbr_exp + pred_ridge_exp + pred_lasso_exp + pred_gbr_exp)/4

blend_pred = average_pred

blend_pred = np.where(average_pred < th1,

                      (a1*pred_xgbr_exp + b1*pred_ridge_exp + c1*pred_lasso_exp + d1*pred_gbr_exp)/(a1+b1+c1+d1), blend_pred)

blend_pred = np.where((average_pred >=th1) & (average_pred < th2),

                      (a2*pred_xgbr_exp + b2*pred_ridge_exp + c2*pred_lasso_exp + d2*pred_gbr_exp)/(a2+b2+c2+d2),blend_pred)

blend_pred = np.where((average_pred >=th2) & (average_pred < th3), 

                      (a3*pred_xgbr_exp + b3*pred_ridge_exp + c3*pred_lasso_exp + d3*pred_gbr_exp)/(a3+b3+c3+d3),blend_pred)

blend_pred = np.where((average_pred >=th3) & (average_pred < th4),

                      (a4*pred_xgbr_exp + b4*pred_ridge_exp + c4*pred_lasso_exp + d4*pred_gbr_exp)/(a4+b4+c4+d4),blend_pred)

blend_pred = np.where((average_pred >=th4) & (average_pred < th5),

                      (a5*pred_xgbr_exp + b5*pred_ridge_exp + c5*pred_lasso_exp + d5*pred_gbr_exp)/(a5+b5+c5+d5),blend_pred)

blend_pred = np.where(average_pred >= th5, 

                      (a6*pred_xgbr_exp + b6*pred_ridge_exp + c6*pred_lasso_exp + d6*pred_gbr_exp)/(a6+b6+c6+d6), blend_pred)
blend_pred.shape
df_house_test5=pd.DataFrame()

df_house_test5["Id"]=df_house_test["Id"]
df_house_test5["SalePrice"]=pd.DataFrame(blend_pred)
df_house_test5.head()
df_house_test5.to_csv("house_data_submission.csv", index=False)