import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')


from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, mean_squared_error
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
hp = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

hp_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



hp.head()
hp.info()
hp.SalePrice.describe()
def downcast_dtypes(hp):    

   

    float_cols = [c for c in df if hp[c].dtype == "float64"]

    int_cols =   [c for c in df if hp[c].dtype == "int64"]

    

    hp[float_cols] = hp[float_cols].astype(np.float32)

    hp[int_cols]   = hp[int_cols].astype(np.int32)

    

    return hp
plt.figure(figsize=(15,5))



sns.distplot(hp['SalePrice'],kde=False)

plt.xlabel('Sale price')

plt.axis([0,800000,0,180])

objs = [column for column in hp.columns if hp[column].dtype in ['object']]

intfloat = [column for column in hp.columns if hp[column].dtype in ['int64','float64']]

cats = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'BsmtFullBath',

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'OverallQual', 'YrSold', 'MoSold', 'GarageCars','BsmtHalfBath', 'FullBath', 'HalfBath',

        'BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces']
hp['TotalSF'] = hp['TotalBsmtSF'] + hp['1stFlrSF'] + hp['2ndFlrSF']

hp_test['TotalSF'] = hp_test['TotalBsmtSF'] + hp_test['1stFlrSF'] + hp_test['2ndFlrSF']
nan_cols =[]

for i in objs:

    if hp[i].isna().sum() != 0:

#         print(i, hp[i].isna().sum())

        nan_cols.append(i)
for i in intfloat:

    if hp[i].isna().sum() != 0:

#         print(i, hp[i].isna().sum())

        nan_cols.append(i)
for i in nan_cols:

    print(hp[i].value_counts())
hp['Alley']=hp['Alley'].fillna("Grvl")

hp['BsmtQual']=hp['BsmtQual'].fillna("TA")

hp['BsmtCond']=hp['BsmtCond'].fillna("TA")

hp['BsmtExposure']=hp['BsmtExposure'].fillna("No")

hp['BsmtFinType1']=hp['BsmtFinType1'].fillna("Unf")

hp['BsmtFinType2']=hp['BsmtFinType2'].fillna("Unf")

hp['Electrical']=hp['Electrical'].fillna("SBrkr")

hp['FireplaceQu']=hp['FireplaceQu'].fillna("Gd")

hp['GarageType']=hp['GarageType'].fillna("Attchd")

hp['GarageFinish']=hp['GarageFinish'].fillna("Unf")

hp['GarageQual']=hp['GarageQual'].fillna("TA")

hp['GarageCond']=hp['GarageCond'].fillna("TA")

hp['PoolQC']=hp['GarageCond'].fillna("TA")

hp['Fence']=hp['Fence'].fillna("MnPrv")

hp['MiscFeature']=hp['MiscFeature'].fillna("Shed")

hp_test['Alley']=hp_test['Alley'].fillna("Grvl")

hp_test['BsmtQual']=hp_test['BsmtQual'].fillna("TA")

hp_test['BsmtCond']=hp_test['BsmtCond'].fillna("TA")

hp_test['BsmtExposure']=hp_test['BsmtExposure'].fillna("No")

hp_test['BsmtFinType1']=hp_test['BsmtFinType1'].fillna("Unf")

hp_test['BsmtFinType2']=hp_test['BsmtFinType2'].fillna("Unf")

hp_test['Electrical']=hp_test['Electrical'].fillna("SBrkr")

hp_test['FireplaceQu']=hp_test['FireplaceQu'].fillna("Gd")

hp_test['GarageType']=hp_test['GarageType'].fillna("Attchd")

hp_test['GarageFinish']=hp_test['GarageFinish'].fillna("Unf")

hp_test['GarageQual']=hp_test['GarageQual'].fillna("TA")

hp_test['GarageCond']=hp_test['GarageCond'].fillna("TA")

hp_test['PoolQC']=hp_test['GarageCond'].fillna("TA")

hp_test['Fence']=hp_test['Fence'].fillna("MnPrv")

hp_test['MiscFeature']=hp_test['MiscFeature'].fillna("Shed")
hp_cor = hp.loc[:,'MSSubClass':]

corr = hp_cor.select_dtypes(include = ['float64', 'int64'])

corr = corr.corr()

fig, ax = plt.subplots(figsize=(30,30))

# sns.set(font_scale=1) 

sns.heatmap(corr, linewidths=.7, annot=True)

ordered_corr = corr['SalePrice']

ordered_corr.sort_values(axis=0,ascending=False)[1:10]
plt.figure(figsize=(20,15))

plt.subplot(321)

plt.scatter(hp['OverallQual'], hp['SalePrice'])

sns.regplot(x ='OverallQual', y ='SalePrice', data=hp, scatter=False, color ='Green')

plt.subplot(322)

plt.scatter(hp['GrLivArea'], hp['SalePrice'])

sns.regplot(x ='GrLivArea', y ='SalePrice', data=hp, scatter=False, color ='Green')

plt.subplot(323)

plt.scatter(hp['GarageCars'], hp['SalePrice'])

sns.regplot(x ='GarageCars', y ='SalePrice', data=hp, scatter=False, color ='Green')

plt.subplot(324)

plt.scatter(hp['GarageArea'], hp['SalePrice'])

sns.regplot(x ='GarageArea', y ='SalePrice', data=hp, scatter=False, color ='Green')

plt.subplot(325)

plt.scatter(hp['TotalBsmtSF'], hp['SalePrice'])

sns.regplot(x ='TotalBsmtSF', y ='SalePrice', data=hp, scatter=False, color ='Green')





hp.drop(hp[(hp['OverallQual']<5) & (hp['SalePrice']>200000)].index, inplace=True)

hp.drop(hp[(hp['GrLivArea']>4000) & (hp['SalePrice']<300000)].index, inplace=True)

hp.drop(hp[(hp['TotalBsmtSF']>4000) & (hp['SalePrice']<200000)].index, inplace=True)

hp.drop(hp[(hp['TotalBsmtSF']>2000) & (hp['SalePrice']>700000)].index, inplace=True)

hp.drop(hp[(hp['GarageArea']>700) & (hp['SalePrice']>650000)].index, inplace=True)

hp.drop(hp[(hp['GarageArea']>1200) & (hp['SalePrice']>100000)].index, inplace=True)
plt.figure(figsize=(20,15))

plt.subplot(321)

plt.scatter(hp['OverallQual'], hp['SalePrice'])

sns.regplot(x ='OverallQual', y ='SalePrice', data=hp, scatter=False, color ='Green')

plt.subplot(322)

plt.scatter(hp['GrLivArea'], hp['SalePrice'])

sns.regplot(x ='GrLivArea', y ='SalePrice', data=hp, scatter=False, color ='Green')

plt.subplot(323)

plt.scatter(hp['GarageCars'], hp['SalePrice'])

sns.regplot(x ='GarageCars', y ='SalePrice', data=hp, scatter=False, color ='Green')

plt.subplot(324)

plt.scatter(hp['GarageArea'], hp['SalePrice'])

sns.regplot(x ='GarageArea', y ='SalePrice', data=hp, scatter=False, color ='Green')

plt.subplot(325)

plt.scatter(hp['TotalBsmtSF'], hp['SalePrice'])

sns.regplot(x ='TotalBsmtSF', y ='SalePrice', data=hp, scatter=False, color ='Green')



plt.figure(figsize=(9,9))

sns.relplot(x="SalePrice",y="GrLivArea",col="BldgType",hue="OverallQual",kind="scatter",

            height=10,aspect=0.3,data=hp)
plt.figure(figsize=(9,9))

sns.relplot(x="SalePrice",y="GarageArea",col="GarageCars",hue="GarageQual",kind="scatter",

            height=10,aspect=0.3,data=hp)
obj_type_variables=objs+cats

obj_type_variables=list(set(obj_type_variables))



train_data_X_obj = hp[obj_type_variables].apply(lambda column: column.astype('category').cat.codes)

test_data_X_obj=hp_test[obj_type_variables].apply(lambda column: column.astype('category').cat.codes) 

train_data_X_obj.columns
actual_obj = ['Fireplaces','MSSubClass','OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath',

              'FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces',

              'GarageCars','MoSold','YrSold','Id','SalePrice']

for i in actual_obj:

    if i in intfloat:

        intfloat.remove(i)
train_data_X_int=hp[intfloat]

test_data_X_int=hp_test[intfloat]
print(train_data_X_obj.isna().sum())

print(train_data_X_int.isna().sum())
train_data_X_int=train_data_X_int.fillna(train_data_X_int.median())

test_data_X_int=test_data_X_int.fillna(train_data_X_int.median())
train_X=train_data_X_int.merge(train_data_X_obj,left_index=True,right_index=True).reset_index(drop=True)

test_X=test_data_X_int.merge(test_data_X_obj,left_index=True,right_index=True).reset_index(drop=True)
train_Y=hp['SalePrice']
X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y,test_size = 0.2, random_state=0)
rf = RandomForestRegressor()

params = {"max_depth":[15,20,25], "n_estimators":[24,30,36]}

rf_reg = GridSearchCV(rf, params, cv = 10, n_jobs =10)

rf_reg.fit(X_train, Y_train)

print(rf_reg.best_estimator_)

best_estimator=rf_reg.best_estimator_

y_pred_test = best_estimator.predict(X_test)
from sklearn.metrics import r2_score

print('Mean Square Error test = ' + str(mean_squared_error(Y_test, y_pred_test,squared=False)))

print('Root Mean Square Error test = ' + str((mean_squared_error(np.log(Y_test), np.log(y_pred_test),squared=False))))



print ('R2 square '+str(r2_score(Y_test, y_pred_test))) 
param = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}
from sklearn.ensemble import GradientBoostingRegressor

# gbr = GradientBoostingRegressor(max_depth=4, min_samples_split=8, min_samples_leaf=15, subsample=1,max_features='sqrt', 

#                                 random_state=42, loss='huber')

GBoost = GradientBoostingRegressor(n_estimators=1992, learning_rate=0.03, max_depth=3, max_features='sqrt', 

                                   min_samples_leaf=15, min_samples_split=8, loss='huber', random_state =42)

GBoost.fit(X_train,Y_train)

y_pred_test=GBoost.predict(X_test)

print('Mean Square Error test = ' + str(mean_squared_error(Y_test, y_pred_test,squared=False)))

print('Root Mean Square Error test = ' + str((mean_squared_error(np.log(Y_test), np.log(y_pred_test),squared=False))))



print ('R2 square '+str(r2_score(Y_test, y_pred_test)))
pred_test = GBoost.predict(test_X)
sample_submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission=pd.DataFrame({"Id":sample_submission['Id'],"SalePrice":pred_test})

submission.to_csv('submission3.csv',index=False)
import xgboost as xgb



xgbr = xgb.XGBRegressor()

params = {'learning_rate': [0.1, 0.15, 0.2, 0.25], 'max_depth': [3, 5,7,9], }



xgbr_reg = GridSearchCV(xgbr, params, cv = 10, n_jobs =1)

xgbr_reg.fit(X_train, Y_train)



print("Best params:{}".format(xgbr_reg.best_params_))



best_x = xgbr_reg.best_estimator_

y_val_pred_x = best_x.predict(X_test)
print('Mean Square Error test = ' + str(mean_squared_error(Y_test, y_val_pred_x,squared=False)))

print('Root Mean Square Error test = ' + str((mean_squared_error(np.log(Y_test), np.log(y_val_pred_x),squared=False))))



print ('R2 square '+str(r2_score(Y_test, y_val_pred_x)))
from sklearn.preprocessing import MinMaxScaler, Normalizer
normalized_tr =pd.DataFrame(MinMaxScaler().fit_transform(train_X), columns = train_X.columns)

normalized_tr
normalized_test =pd.DataFrame(MinMaxScaler().fit_transform(test_X), columns = test_X.columns)

normalized_test
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(normalized_tr, train_Y, test_size=0.2, random_state=0)
GBoost = GradientBoostingRegressor(n_estimators=1992, learning_rate=0.03, max_depth=3, max_features='sqrt', 

                                   min_samples_leaf=15, min_samples_split=8, loss='huber', random_state =42)

GBoost.fit(X_train2, Y_train2)

y_predgbr = GBoost.predict(X_test2)
print('Mean Square Error test = ' + str(mean_squared_error(Y_test2, y_predgbr,squared=False)))

print('Root Mean Square Error test = ' + str((mean_squared_error(np.log(Y_test2), np.log(y_predgbr),squared=False))))



print ('R2 square '+str(r2_score(Y_test2, y_predgbr)))
pred_test2 = GBoost.predict(test_X)
output = pd.DataFrame({'Id': sample_submission['Id'],

                       'SalePrice': pred_test2})

output.to_csv('submission2.csv', index=False)