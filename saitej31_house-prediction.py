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
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_data['train']  = 1

test_data['train']  = 0

data = pd.concat([train_data,test_data],axis =0)

print(len(data))

data.head()
for i in data.columns:

    print(f'length of unique values in {i}',len(set(data[i])))

    print(f'some of the unique values in {i}',list(set(data[i]))[0:5])

    print('---------------------------------------------------------')
import missingno as msno 

msno.matrix(data.iloc[:,1:20])

msno.matrix(data.iloc[:,20:40])

msno.matrix(data.iloc[:,40:60])

msno.matrix(data.iloc[:,60:])

y = train_data.SalePrice

data =data.drop(['SalePrice'],axis = 1)
objects = data.select_dtypes(include=['object'])

numericals =data.select_dtypes(exclude=['object'])
print(objects.dtypes)
objects.isna().sum()
object_none_fill = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','PoolQC','Fence','MiscFeature','GarageType','GarageFinish','GarageQual','GarageCond']

objects[object_none_fill] = objects[object_none_fill].fillna('none')
object_mode_fill = ['Electrical','KitchenQual','Functional','MasVnrType','Exterior1st','Exterior2nd','MSZoning','Utilities','SaleType']

objects[object_mode_fill] = objects[object_mode_fill].fillna(objects[object_mode_fill].mode().iloc[0])
for i in objects.columns:

    print(i,'-----',len(set(objects[i])),'-----',objects[i].dtype,'\n')

import matplotlib.pyplot as plt

import seaborn as sns

for i in objects.columns:

    if len(set(objects[i])) <= 5:

        plt.title(f"{i}")

        ax = sns.countplot(x= i, data=objects)

        plt.show()
for i in objects.columns:

    if len(set(objects[i])) <= 10 and len(set(objects[i])) > 5:

        plt.title(f"{i}")

        ax = sns.countplot(x= i, data=objects)

        plt.show()
remove = ['Street','Alley','Fence','MiscFeature','PoolQC','Utilities','Exterior1st','Exterior2nd','LotConfig','LandSlope','BldgType','Electrical','RoofMatl','BsmtFinType2','Functional','Condition1','Condition2']
objects = objects.drop(remove,axis =1)
for i in objects.columns:

    print(i,set(list(objects[i])))

MSZoning ={'RL':0, 'FV' :1, 'C (all)':2, 'RM' :3, 'RH':4}

objects['MSZoning'] = objects['MSZoning'].map(MSZoning)







LotShape = {'IR1':0, 'IR2':1, 'IR3':2,'Reg':3}

objects['LotShape'] = objects['LotShape'].map(LotShape)



LandContour = {'Low':0, 'Bnk':1, 'HLS':2, 'Lvl':3}

objects['LandContour'] = objects['LandContour'].map(LandContour)





Neighborhood = {'Veenker':0, 'SWISU':1, 'Blueste':2, 'Gilbert':3, 'MeadowV':4, 'SawyerW':5, 'Blmngtn':6, 'BrDale':7, 'Sawyer':8, 'ClearCr':9, 'IDOTRR':10, 'NAmes':11, 'OldTown':12, 'Somerst':13, 'NridgHt':14, 'NoRidge':15, 'Timber':16, 'StoneBr':17, 'BrkSide':18, 'NPkVill':19, 'Mitchel':20, 'NWAmes':21, 'CollgCr':22, 'Crawfor':23, 'Edwards':24}

objects['Neighborhood'] = objects['Neighborhood'].map(Neighborhood)



BsmtFinType1 ={'BLQ':1, 'ALQ':2, 'LwQ':3, 'Rec':4, 'GLQ':5, 'Unf':6, 'none':0}

objects['BsmtFinType1'] = objects['BsmtFinType1'].map(BsmtFinType1)







HouseStyle = {'2.5Fin':0 , 'SLvl':1, 'SFoyer':2, '2.5Unf':3, '1Story':4, '1.5Fin':5, '2Story':6, '1.5Unf':7}

objects['HouseStyle'] = objects['HouseStyle'].map(HouseStyle)



RoofStyle = {'Gable':0, 'Shed':1, 'Flat':2, 'Mansard':3, 'Hip':4, 'Gambrel':5}

objects['RoofStyle'] = objects['RoofStyle'].map(RoofStyle)



MasVnrType = {'Stone':1, 'BrkFace':2, 'None':0, 'BrkCmn':3}

objects['MasVnrType'] = objects['MasVnrType'].map(MasVnrType)





score = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'none':0,'No':0,'N':0, 'Y':1 }



objects['ExterQual'] = objects['ExterQual'].map(score)

objects['ExterCond'] = objects['ExterCond'].map(score)

objects['BsmtQual'] = objects['BsmtQual'].map(score)

objects['BsmtCond'] = objects['BsmtCond'].map(score)

objects['BsmtExposure'] = objects['BsmtExposure'].map(score)

objects['GarageQual'] = objects['GarageQual'].map(score)

objects['GarageCond'] = objects['GarageCond'].map(score)

objects['PavedDrive'] = objects['PavedDrive'].map(score)

objects['HeatingQC'] = objects['HeatingQC'].map(score)

objects['CentralAir'] = objects['CentralAir'].map(score)

objects['KitchenQual'] = objects['KitchenQual'].map(score)

objects['FireplaceQu'] = objects['FireplaceQu'].map(score)





Heating = {'Floor':0, 'Grav':1, 'GasA':2, 'Wall':3, 'GasW':4, 'OthW':5}

objects['Heating'] = objects['Heating'].map(Heating)



Foundation = {'Slab':0, 'Stone':1, 'Wood':2, 'PConc':3, 'CBlock':4, 'BrkTil':5}

objects['Foundation'] = objects['Foundation'].map(Foundation)





GarageType = {'CarPort':1, 'Detchd':2, 'Basment':3,'2Types':4, 'BuiltIn':5, 'Attchd':6, 'none':0}

objects['GarageType'] = objects['GarageType'].map(GarageType)



GarageFinish ={'none':0, 'Unf':1, 'Fin':2, 'RFn':3}

objects['GarageFinish'] = objects['GarageFinish'].map(GarageFinish)



SaleType = {'WD':0, 'New':1, 'Con':2, 'COD':3, 'Oth':4, 'ConLI':5, 'ConLD':6, 'ConLw':7, 'CWD':8}

objects['SaleType'] = objects['SaleType'].map(SaleType)



SaleCondition = {'Abnorml':0, 'Family':1, 'AdjLand':2, 'Normal':3, 'Alloca':4 ,'Partial':5}

objects['SaleCondition'] = objects['SaleCondition'].map(SaleCondition)

objects.dtypes
objects.head()
numericals.isna().sum()
print(round((numericals['LotFrontage']).mean()))

print(round((numericals['YrSold']-numericals['YearBuilt']).mean()))
numericals['GarageYrBlt'] = numericals['GarageYrBlt'].fillna(numericals['YrSold']-36)

numericals['LotFrontage'] = numericals['LotFrontage'].fillna(69)
numericals = numericals.fillna(0)
numericals.isna().sum()
for i in numericals.columns:

    if len(set(numericals[i])) <= 5:

        plt.title(f"{i}")

        ax = sns.countplot(x= i, data=numericals)

        plt.show()
for i in numericals.columns:

    if len(set(numericals[i])) <= 10 and len(set(numericals[i])) > 5:

        plt.title(f"{i}")

        ax = sns.countplot(x= i, data=numericals)

        plt.show()
data = pd.concat([objects,numericals],axis =1)

data = data.drop(['Id'],axis =1)

data.shape
data['OverallQualandCond']= (data['OverallQual']+data['OverallCond'])

data['ExterQualandCond']= (data['ExterQual']+data['ExterCond'])

data['BsmtAll']= (data['BsmtCond']+data['BsmtQual']+data['BsmtExposure'])

data['GarageQualandCond']= (data['GarageQual']+data['GarageCond'])

data = data.drop(['OverallQual','OverallCond','ExterQual','ExterCond','BsmtCond','BsmtQual','BsmtExposure','GarageQual','GarageCond'],axis =1)
data.shape


train = data[data['train'] == 1]

train = train.drop(['train',],axis=1)



test = data[data['train'] == 0]

test = test.drop(['train',],axis=1)
from sklearn.model_selection import train_test_split

import xgboost as xgb

import math

import sklearn.metrics as metrics



x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.25,random_state=42)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)



xgb_model.fit(x_train,y_train)

predictions = xgb_model.predict(x_test)
print('Root Mean Square Error for  Log of pred and true = ' + str(math.sqrt(metrics.mean_squared_error(np.log(y_test), np.log(predictions)))))
import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)



xgb_model.fit(train,y)

predict = xgb_model.predict(test)
submission = pd.DataFrame({

        "Id": test_data["Id"],

        "SalePrice": predict

    })

submission.to_csv('submission.csv', index=False)