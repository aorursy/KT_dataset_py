import numpy as np 

import pandas as pd 

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

fig_dims = (20,10)

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adadelta

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from keras.wrappers.scikit_learn import KerasRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



train.head(5)
cols = train.columns

cols = cols.delete([0,80])

for i in cols:

    if np.issubdtype(train[i].dtype,np.number) == True:

        mean = train[i].mean()

        std = train[i].std()

        train[i] = (train[i]-mean)/std

        #train[i] = preprocessing.scale(train[i])

cols2 = test.columns

cols2 = cols2.delete([0])

for i in cols2:

    if np.issubdtype(test[i].dtype,np.number) == True:

        mean = test[i].mean()

        std = test[i].std()

        test[i] = (test[i]-mean)/std

        #test[i] = preprocessing.scale(test[i])

train.head(5)
#MSZoning

#sns.barplot(x=train.index,y=train['MSZoning'])

train['MSZoning'] = train["MSZoning"].map({'RL':1,'RM':2,'FV':3,'RH':4,'C (all)':0})

test['MSZoning'] = test["MSZoning"].map({'RL':1,'RM':2,'FV':3,'RH':4,'C (all)':0})



#Street

#sns.barplot(x=train.index,y=train['Street'])

train['Street'] = train["Street"].map({'Pave':1,'Grvl':1})

test['Street'] = test["Street"].map({'Pave':1,'Grvl':1})



#Alley

#sns.barplot(x=train.index,y=train["Alley"])

train['Alley'] = train['Alley'].fillna(0)

test['Alley'] = test['Alley'].fillna(0)

train['Alley'] = train['Alley'].map({'Pave':0,'Grvl':1,0:2})

test['Alley'] = test['Alley'].map({'Pave':0,'Grvl':1,0:2})



#LotShape

#sns.barplot(x=train.index,y=train["LotShape"])

train["LotShape"] = train["LotShape"].map({'Reg':0,'IR1':1,'IR2':2,'IR3':3})

test["LotShape"] = test["LotShape"].map({'Reg':0,'IR1':1,'IR2':2,'IR3':3})



#LandContour

#sns.barplot(x=train.index,y=train["LandContour"])

train["LandContour"] = train["LandContour"].map({'Lvl':0,'Bnk':1,'Low':2,'HLS':3})

test["LandContour"] = test["LandContour"].map({'Lvl':0,'Bnk':1,'Low':2,'HLS':3})



#Utilities

#sns.barplot(x=train.index,y=train["Utilities"])

train["Utilities"] = train["Utilities"].map({'AllPub':0,'NoSeWa':1})

test["Utilities"] = test["Utilities"].map({'AllPub':0,'NoSeWa':1})



#LotConfig

#sns.barplot(x=train.index,y=train["LotConfig"])

train['LotConfig'] = train['LotConfig'].map({'Inside':0,'FR2':1,'Corner':2,'CulDSac':3,'FR3':4})

test['LotConfig'] = test['LotConfig'].map({'Inside':0,'FR2':1,'Corner':2,'CulDSac':3,'FR3':4})



#LandSlope

#sns.barplot(x=train.index,y=train['LandSlope'])

train['LandSlope'] = train['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})

test['LandSlope'] = test['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})



#Neighborhood

uniques = train['Neighborhood'].unique()

i=0

for neigh in uniques:

    train['Neighborhood'] = train['Neighborhood'].replace({neigh:i})

    test['Neighborhood'] = test['Neighborhood'].replace({neigh:i})

    i+=1



#Condition1/Condition2

#sns.barplot(x=train.index,y=train['Condition1'])

#sns.barplot(x=train.index,y=train['Condition2'])

train['Condition1'] = train['Condition1'].map({'Norm':0,'Feedr':1,'PosN':2,'Artery':3,'RRAe':4,'RRNn':5,'RRAn':6,'PosA':7,'RRNe':8})

test['Condition1'] = test['Condition1'].map({'Norm':0,'Feedr':1,'PosN':2,'Artery':3,'RRAe':4,'RRNn':5,'RRAn':6,'PosA':7,'RRNe':8})

train['Condition2'] = train['Condition2'].map({'Norm':0,'Feedr':1,'PosN':2,'Artery':3,'RRAe':4,'RRNn':5,'RRAn':6,'PosA':7})

test['Condition2'] = test['Condition2'].map({'Norm':0,'Feedr':1,'PosN':2,'Artery':3,'RRAe':4,'RRNn':5,'RRAn':6,'PosA':7})



#BldgType

#sns.barplot(x=train.index,y=train['BldgType'])

train['BldgType'] = train['BldgType'].map({'1Fam':0,'2fmCon':1,'Duplex':2,'TwnhsE':3,'Twnhs':4})

test['BldgType'] = test['BldgType'].map({'1Fam':0,'2fmCon':1,'Duplex':2,'TwnhsE':3,'Twnhs':4})



#HouseStyle

#sns.barplot(x=train.index,y=train['HouseStyle'])

train['HouseStyle'].unique()

train['HouseStyle'] = train['HouseStyle'].map({'2Story':0,'1Story':1,'1.5Fin':2,'1.5Unf':3,'SFoyer':4,'SLvl':5,'2.5Unf':6,'2.5Fin':7})

test['HouseStyle'] = test['HouseStyle'].map({'2Story':0,'1Story':1,'1.5Fin':2,'1.5Unf':3,'SFoyer':4,'SLvl':5,'2.5Unf':6,'2.5Fin':7})



#RoofStyle

train['RoofStyle'].unique()

train['RoofStyle'] = train['RoofStyle'].map({'Gable':0,'Hip':1,'Gambrel':2,'Mansard':3,'Flat':4,'Shed':5})

test['RoofStyle'] = test['RoofStyle'].map({'Gable':0,'Hip':1,'Gambrel':2,'Mansard':3,'Flat':4,'Shed':5})



#RoofMatl

train['RoofMatl'].unique()

train['RoofMatl'] = train['RoofMatl'].map({'CompShg':0,'WdShngl':1,'Metal':2,'WdShake':3,'Membran':4,'Tar&Grv':5,'Roll':6,'ClyTile':7})

test['RoofMatl'] = test['RoofMatl'].map({'CompShg':0,'WdShngl':1,'Metal':2,'WdShake':3,'Membran':4,'Tar&Grv':5,'Roll':6,'ClyTile':7})



#Exterior1st

train['Exterior1st'].unique()

train['Exterior1st'] = train['Exterior1st'].map({'VinylSd':0, 'MetalSd':1, 'Wd Sdng':2, 'HdBoard':3, 'BrkFace':4, 'WdShing':5,

       'CemntBd':6, 'Plywood':7, 'AsbShng':8, 'Stucco':9, 'BrkComm':10, 'AsphShn':11,

       'Stone':12, 'ImStucc':13, 'CBlock':14})

test['Exterior1st'] = test['Exterior1st'].map({'VinylSd':0, 'MetalSd':1, 'Wd Sdng':2, 'HdBoard':3, 'BrkFace':4, 'WdShing':5,

       'CemntBd':6, 'Plywood':7, 'AsbShng':8, 'Stucco':9, 'BrkComm':10, 'AsphShn':11,

       'Stone':12, 'ImStucc':13, 'CBlock':14})



#Exterior2nd

train['Exterior2nd'].unique()

train['Exterior2nd'] = train['Exterior2nd'].map({'VinylSd':0, 'MetalSd':1, 'Wd Shng':2, 'HdBoard':3, 'Plywood':4, 'Wd Sdng':5,

       'CmentBd':6, 'BrkFace':7, 'Stucco':8, 'AsbShng':9, 'Brk Cmn':10, 'ImStucc':11,

       'AsphShn':12, 'Stone':13, 'Other':14, 'CBlock':15})

test['Exterior2nd'] = test['Exterior2nd'].map({'VinylSd':0, 'MetalSd':1, 'Wd Shng':2, 'HdBoard':3, 'Plywood':4, 'Wd Sdng':5,

       'CmentBd':6, 'BrkFace':7, 'Stucco':8, 'AsbShng':9, 'Brk Cmn':10, 'ImStucc':11,

       'AsphShn':12, 'Stone':13, 'Other':14, 'CBlock':15})



#MasVnrType

train['MasVnrType'].unique()

train['MasVnrType'] = train['MasVnrType'].fillna(0)

test['MasVnrType'] = test['MasVnrType'].fillna(0)

train['MasVnrType'] = train['MasVnrType'].map({'BrkFace':1, 'None':0, 'Stone':2, 'BrkCmn':3,0:0})

test['MasVnrType'] = test['MasVnrType'].map({'BrkFace':1, 'None':0, 'Stone':2, 'BrkCmn':3,0:0})



#ExterQual

train['ExterQual'].unique()

train['ExterQual'] = train['ExterQual'].map({'Gd':0, 'TA':1, 'Ex':2, 'Fa':3})

test['ExterQual'] = test['ExterQual'].map({'Gd':0, 'TA':1, 'Ex':2, 'Fa':3})



#ExterCond

train['ExterCond'].unique()

train['ExterCond'] = train['ExterCond'].map({'TA':0, 'Gd':1, 'Fa':2, 'Po':3, 'Ex':4})

test['ExterCond'] = test['ExterCond'].map({'TA':0, 'Gd':1, 'Fa':2, 'Po':3, 'Ex':4})



#Foundation

train['Foundation'].unique()

train['Foundation'] = train['Foundation'].map({'PConc':0, 'CBlock':1, 'BrkTil':2, 'Wood':3, 'Slab':4, 'Stone':5})

test['Foundation'] = test['Foundation'].map({'PConc':0, 'CBlock':1, 'BrkTil':2, 'Wood':3, 'Slab':4, 'Stone':5})



#BsmtQual

train['BsmtQual'].unique()

train['BsmtQual'] = train['BsmtQual'].fillna(0)

test['BsmtQual'] = test['BsmtQual'].fillna(0)

train['BsmtQual'] = train['BsmtQual'].map({'Gd':1, 'TA':2, 'Ex':3, 0:0, 'Fa':4})

test['BsmtQual'] = test['BsmtQual'].map({'Gd':1, 'TA':2, 'Ex':3, 0:0, 'Fa':4})



#BsmtCond

train['BsmtCond'].unique()

train['BsmtCond'] = train['BsmtCond'].fillna(0)

test['BsmtCond'] = test['BsmtCond'].fillna(0)

train['BsmtCond'] = train['BsmtCond'].map({'Gd':1, 'TA':2, 'Po':3, 0:0, 'Fa':4})

test['BsmtCond'] = test['BsmtCond'].map({'Gd':1, 'TA':2, 'Po':3, 0:0, 'Fa':4})



#BsmtExposure

train['BsmtExposure'].unique()

train['BsmtExposure'] = train['BsmtExposure'].fillna(0)

test['BsmtExposure'] = test['BsmtExposure'].fillna(0)

train['BsmtExposure'] = train['BsmtExposure'].map({'Gd':1, 'No':2, 'Mn':3, 0:0, 'Av':4})

test['BsmtExposure'] = test['BsmtExposure'].map({'Gd':1, 'No':2, 'Mn':3, 0:0, 'Av':4})



#BsmtFinType1/2

train['BsmtFinType1'].unique()

train['BsmtFinType1'] = train['BsmtFinType1'].fillna(0)

test['BsmtFinType1'] = test['BsmtFinType1'].fillna(0)

train['BsmtFinType1'] = train['BsmtFinType1'].map({'GLQ':0, 'ALQ':1, 'Unf':2, 'Rec':3, 'BLQ':4, 0:0, 'LwQ':5})

test['BsmtFinType1'] = test['BsmtFinType1'].map({'GLQ':0, 'ALQ':1, 'Unf':2, 'Rec':3, 'BLQ':4, 0:0, 'LwQ':5})

train['BsmtFinType2'] = train['BsmtFinType2'].fillna(0)

train['BsmtFinType2'] = train['BsmtFinType2'].map({'GLQ':0, 'ALQ':1, 'Unf':2, 'Rec':3, 'BLQ':4, 0:0, 'LwQ':5})

test['BsmtFinType2'] = test['BsmtFinType2'].map({'GLQ':0, 'ALQ':1, 'Unf':2, 'Rec':3, 'BLQ':4, 0:0, 'LwQ':5})



#Heating

train['Heating'].unique()

train['Heating'] = train['Heating'].map({'GasA':0, 'GasW':1, 'Grav':2, 'Wall':3, 'OthW':4, 'Floor':5})

test['Heating'] = test['Heating'].map({'GasA':0, 'GasW':1, 'Grav':2, 'Wall':3, 'OthW':4, 'Floor':5})



#HeatingQC

train['HeatingQC'].unique()

train['HeatingQC'] = train['HeatingQC'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4})

test['HeatingQC'] = test['HeatingQC'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4})



#CentralAir

train['CentralAir'].unique()

train['CentralAir'] = train['CentralAir'].map({'Y':1,'N':0})

test['CentralAir'] = test['CentralAir'].map({'Y':1,'N':0})



#Electrical

train['Electrical'].unique()

train['Electrical'] = train['Electrical'].fillna(0)

test['Electrical'] = test['Electrical'].fillna(0)

train['Electrical'] = train['Electrical'].map({'SBrkr':0, 'FuseF':1, 'FuseA':2, 'FuseP':3, 'Mix':4, 0:0})

test['Electrical'] = test['Electrical'].map({'SBrkr':0, 'FuseF':1, 'FuseA':2, 'FuseP':3, 'Mix':4, 0:0})



#KitchenQual

train['KitchenQual'].unique()

train['KitchenQual'] = train['KitchenQual'].map({'Gd':0, 'TA':1, 'Ex':2, 'Fa':3})

test['KitchenQual'] = test['KitchenQual'].map({'Gd':0, 'TA':1, 'Ex':2, 'Fa':3})



#Functional

train['Functional'].unique()

train['Functional'] = train['Functional'].map({'Typ':0, 'Min1':1, 'Maj1':2, 'Min2':3, 'Mod':4, 'Maj2':5, 'Sev':6})

test['Functional'] = test['Functional'].map({'Typ':0, 'Min1':1, 'Maj1':2, 'Min2':3, 'Mod':4, 'Maj2':5, 'Sev':6})



#FireplaceQu

train['FireplaceQu'].unique()

train['FireplaceQu'] = train['FireplaceQu'].fillna(0)

test['FireplaceQu'] = test['FireplaceQu'].fillna(0)

train['FireplaceQu'] = train['FireplaceQu'].map({0:0, 'TA':1, 'Gd':2, 'Fa':3, 'Ex':4, 'Po':5})

test['FireplaceQu'] = test['FireplaceQu'].map({0:0, 'TA':1, 'Gd':2, 'Fa':3, 'Ex':4, 'Po':5})



#GarageType

train['GarageType'].unique()

train['GarageType'] = train['GarageType'].fillna(0)

test['GarageType'] = test['GarageType'].fillna(0)

train['GarageType'] = train['GarageType'].map({'Attchd':1, 'Detchd':2, 'BuiltIn':3, 'CarPort':4, 0:0, 'Basment':5, '2Types':6})

test['GarageType'] = test['GarageType'].map({'Attchd':1, 'Detchd':2, 'BuiltIn':3, 'CarPort':4, 0:0, 'Basment':5, '2Types':6})



#GarageFinish

train['GarageFinish'].unique()

train['GarageFinish'] = train['GarageFinish'].fillna(0)

test['GarageFinish'] = test['GarageFinish'].fillna(0)

train['GarageFinish'] = train['GarageFinish'].map({'RFn':1, 'Unf':2, 'Fin':3, 0:0})

test['GarageFinish'] = test['GarageFinish'].map({'RFn':1, 'Unf':2, 'Fin':3, 0:0})



#GarageQual

train['GarageQual'].unique()

train['GarageQual'] = train['GarageQual'].fillna(0)

test['GarageQual'] = test['GarageQual'].fillna(0)

train['GarageQual'] = train['GarageQual'].map({0:0, 'TA':1, 'Gd':2, 'Fa':3, 'Ex':4, 'Po':5})

test['GarageQual'] = test['GarageQual'].map({0:0, 'TA':1, 'Gd':2, 'Fa':3, 'Ex':4, 'Po':5})



#GarageCond

train['GarageCond'].unique()

train['GarageCond'] = train['GarageCond'].fillna(0)

test['GarageCond'] = test['GarageCond'].fillna(0)

train['GarageCond'] = train['GarageCond'].map({0:0, 'TA':1, 'Gd':2, 'Fa':3, 'Ex':4, 'Po':5})

test['GarageCond'] = test['GarageCond'].map({0:0, 'TA':1, 'Gd':2, 'Fa':3, 'Ex':4, 'Po':5})



#PavedDrive

train['PavedDrive'].unique()

train['PavedDrive'] = train['PavedDrive'].map({'Y':1,'N':0,'P':2})

test['PavedDrive'] = test['PavedDrive'].map({'Y':1,'N':0,'P':2})



#PoolQC

train['PoolQC'].unique()

train['PoolQC'] = train['PoolQC'].fillna(0)

test['PoolQC'] = test['PoolQC'].fillna(0)

train['PoolQC'] = train['PoolQC'].map({0:0, 'Ex':1,'Fa':2,'Gd':3})

test['PoolQC'] = test['PoolQC'].map({0:0, 'Ex':1,'Fa':2,'Gd':3})



#Fence

train['Fence'].unique()

train['Fence'] = train['Fence'].fillna(0)

test['Fence'] = test['Fence'].fillna(0)

train['Fence'] = train['Fence'].map({0:0,'MnPrv':1, 'GdWo':2, 'GdPrv':3, 'MnWw':4})

test['Fence'] = test['Fence'].map({0:0,'MnPrv':1, 'GdWo':2, 'GdPrv':3, 'MnWw':4})



#MiscFeature

train['MiscFeature'].unique()

train['MiscFeature'] = train['MiscFeature'].fillna(0)

test['MiscFeature'] = test['MiscFeature'].fillna(0)

train['MiscFeature'] = train['MiscFeature'].map({0:0,'Shed':1, 'Gar2':2, 'Othr':3, 'TenC':4})

test['MiscFeature'] = test['MiscFeature'].map({0:0,'Shed':1, 'Gar2':2, 'Othr':3, 'TenC':4})



#SaleType

train['SaleType'].unique()

train['SaleType'] = train['SaleType'].map({'WD':0, 'New':1, 'COD':2, 'ConLD':3, 'ConLI':4, 'CWD':5, 'ConLw':6, 'Con':7, 'Oth':8})

test['SaleType'] = test['SaleType'].map({'WD':0, 'New':1, 'COD':2, 'ConLD':3, 'ConLI':4, 'CWD':5, 'ConLw':6, 'Con':7, 'Oth':8})



#SaleCondition

train['SaleCondition'].unique()

train['SaleCondition'] = train['SaleCondition'].map({'Normal':0, 'Abnorml':1, 'Partial':2, 'AdjLand':3, 'Alloca':4, 'Family':5})

test['SaleCondition'] = test['SaleCondition'].map({'Normal':0, 'Abnorml':1, 'Partial':2, 'AdjLand':3, 'Alloca':4, 'Family':5})



train.head(5)
print(len(test.select_dtypes(include=['float64','int64']).columns))
for col in train.columns:

    if train[col].isnull().values.any() == True:

        mean = round(train[col].mean(),0)

        train[col] = train[col].fillna(mean)

for col in test.columns:

    if test[col].isnull().values.any() == True:

        mean = round(test[col].mean(),0)

        test[col] = test[col].fillna(mean)
train_copy = train.copy()

x = train_copy.drop(columns=["SalePrice",'Id'])

y = train["SalePrice"]

train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=42)
def model():

    model = Sequential()

    model.add(Dense(25,input_dim=79,kernel_initializer='normal',activation='relu'))

    model.add(Dense(50,kernel_initializer='normal',activation='relu'))

    model.add(Dense(50,kernel_initializer='normal',activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_absolute_error',optimizer='adam')

    return model

keras = model()

keras.fit(train_x,train_y,epochs=500,batch_size=50)

keras_preds = keras.predict(val_x)

keras_error = r2_score(val_y,keras_preds)

print(keras_error)
data = test.drop(columns=['Id'])

predictions = keras.predict(data)

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission.iloc[:,1] = predictions

submission.to_csv('submission.csv',index=False)