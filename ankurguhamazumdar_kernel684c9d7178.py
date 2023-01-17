# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
print(train["SalePrice"].skew())
plt.hist(train["SalePrice"],color="red")
plt.show()

train['SalePrice']=np.log(train['SalePrice'])
print(train["SalePrice"].skew())
plt.hist(train["SalePrice"],color="red")
plt.show()
Numeric_Data=train.select_dtypes(include=[np.number])
print(Numeric_Data.dtypes)

#%% Checking variation of features with results
#Mssubclass
print(Numeric_Data['MSSubClass'].unique())
quality_pivot=train.pivot_table(index="MSSubClass",values='SalePrice', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.xticks(rotation=0)
plt.show()
#%%
#LotFrontage
print(Numeric_Data['LotFrontage'].unique())

plt.scatter(Numeric_Data['LotFrontage'],Numeric_Data["SalePrice"])
plt.show()
#%%
#LotArea
print(Numeric_Data['LotArea'].unique())
plt.scatter(Numeric_Data['LotArea'],Numeric_Data['SalePrice'])

#%%
#overallcond
print(Numeric_Data['OverallCond'].unique())
Cond_pivot=Numeric_Data.pivot_table(index='OverallCond',values='SalePrice',aggfunc=np.median)
plt.xticks(rotation=0)
Cond_pivot.plot(kind='bar',color='Red')

#%%
#YearRemodAdd
print(Numeric_Data["YearRemodAdd"].unique())
plt.scatter(Numeric_Data['YearRemodAdd'],Numeric_Data['SalePrice'])
del train['YearRemodAdd']
del test['YearRemodAdd']
#%%
#MasVnrArea
print(Numeric_Data['MasVnrArea'].unique())

plt.scatter(Numeric_Data['MasVnrArea'],Numeric_Data['SalePrice'])
del train['MasVnrArea']
del test['MasVnrArea']
#%%
#BsmtFinSF1 
print(Numeric_Data['BsmtFinSF1'].unique())
plt.scatter(Numeric_Data['BsmtFinSF1'],Numeric_Data['SalePrice'])
#%%
#BsmtHalfBath
print(Numeric_Data['BsmtHalfBath'].unique())
bath_pivot=Numeric_Data.pivot_table(index='BsmtHalfBath',values='SalePrice',aggfunc=np.median)
print(bath_pivot)
plt.xticks(rotation=0)
bath_pivot.plot(kind='bar',color='red')
#%%
#BsmtFinSF2
print(Numeric_Data['BsmtFinSF2'].unique())
plt.scatter(Numeric_Data['BsmtFinSF2'],Numeric_Data['SalePrice'])
del train['BsmtFinSF2']
del test['BsmtFinSF2']
#%%
#FullBath
print(Numeric_Data['FullBath'].value_counts())
Full_pivot=Numeric_Data.pivot_table(index='FullBath',values='SalePrice',aggfunc=np.median)

Full_pivot.plot(kind='bar',color='red')
plt.xticks(rotation=0)
#%%
#BedroomAbvGr
print(Numeric_Data['BedroomAbvGr'].unique())
BedroomAbvGr_pivot=Numeric_Data.pivot_table(index='BedroomAbvGr',values='SalePrice',aggfunc=np.median)
BedroomAbvGr_pivot.plot(kind="bar",color='b')
#%%
#Fireplaces
print(Numeric_Data['Fireplaces'].unique())

#%%
#GarageArea 
print(Numeric_Data['GarageArea'].unique())
plt.scatter(Numeric_Data['GarageArea'],Numeric_Data['SalePrice'])
#%%
#OpenPorchSF
print(Numeric_Data['OpenPorchSF'].unique())
plt.scatter(Numeric_Data['OpenPorchSF'],Numeric_Data['SalePrice'])
del train['OpenPorchSF']
del test['OpenPorchSF']
#%%
#EnclosedPorch
print(Numeric_Data['EnclosedPorch'].unique())
plt.scatter(Numeric_Data['EnclosedPorch'],Numeric_Data['SalePrice'])
#%%
#ScreenPorch 
print(Numeric_Data['ScreenPorch'].unique())
plt.scatter(Numeric_Data['ScreenPorch'],Numeric_Data['SalePrice'])
del train['ScreenPorch']
del test['ScreenPorch']
#%%
#PoolArea 
print(Numeric_Data['PoolArea'].unique())
#%%
#MiscVal
print(Numeric_Data['MiscVal'].unique())
#%%
#TotalBsmtSF 
print(Numeric_Data['TotalBsmtSF'].unique())
plt.scatter(Numeric_Data['TotalBsmtSF'],Numeric_Data['SalePrice'])

# Categoricals Data:
Categoricals=train.select_dtypes(exclude=[np.number])
print(Categoricals.dtypes)
#print(Categoricals.describe)
#%%
#MSZoning 
print(Categoricals['MSZoning'].unique())
MSZoning_pivot=train.pivot_table(index='MSZoning',values='SalePrice',aggfunc=np.median)
print(MSZoning_pivot)
print(Categoricals['MSZoning'].value_counts())
MSZoning_pivot.plot(kind='bar',color='b')
def changes(x):
   return 1 if x=='RL' else 0
train['MSZoning']=train.MSZoning.apply(changes)
test['MSZoning']=test.MSZoning.apply(changes)
print(train['MSZoning'].unique())
#%%
#Street
print(train['Street'].value_counts())
def st(x):
    return 1 if x=='Pave' else 0
train['Street']=train.Street.apply(st)
test['Street']=test.Street.apply(st)



#%%
#LandContour
print(train['LandContour'].value_counts())
def land(x):
    return 1 if x=='Lvl' else 0
train['LandContour']=train.LandContour.apply(land) 
test['LandContour']=test.LandContour.apply(land)
print(train['LandContour'].value_counts())

#%%
#Utilities
print(train['Utilities'].value_counts())
def Uti(x):
    return 1 if x=='AllPub' else 0
train['Utilities']=train.Utilities.apply(Uti)
test['Utilities']=test.Utilities.apply(Uti)
#%%
#LotConfig
print(train['LotConfig'].value_counts())
def lot(x):
    return 1 if x=='Inside' else 0
train['LotConfig']=train.LotConfig.apply(lot)
test['LotConfig']=train.LotConfig.apply(lot)
print(train['LotConfig'].value_counts())
#%%
#LandSlope
print(train['LandSlope'].value_counts())
def lan(x):
    return 1 if x=='Gtl' else 0
train['LandSlope']=train.LandSlope.apply(lan)
test['LandSlope']=test.LandSlope.apply(lan)
#%%
#SaleCondition
print(train['SaleCondition'].value_counts())
def nor(x):
    return 1 if x=='Normal' else 0
train['SaleCondition']=train.SaleCondition.apply(nor)
test['SaleCondition']=test.SaleCondition.apply(nor)
#%%
#SaleType 
print(train['SaleType'].value_counts())  
def wd(x):
    return 1 if x=='WD' else 0
train['SaleType']=train.SaleType.apply(wd)
test['SaleType']=test.SaleType.apply(wd)
#%%
#MiscFeature
print(train['MiscFeature'].value_counts())
print(train['MiscFeature'].isnull().value_counts())
del train['MiscFeature']
del test['MiscFeature']
#%%
#Fence
print(train['Fence'].value_counts())
print(train['Fence'].isnull().value_counts())
del train['Fence']
del test['Fence']
#%%
#PoolQC 
print(train['PoolQC'].value_counts())  
print(train['PoolQC'].isnull().value_counts())  
del train['PoolQC']
del test['PoolQC']

#%%
#PavedDrive
print(train['PavedDrive'].value_counts())     
def Pave(x):
    return 1 if x=='Y' else 0
train['PavedDrive']=train.PavedDrive.apply(Pave)
test['PavedDrive']=test.PavedDrive.apply(Pave)
#%%
#GarageCond
print(train['GarageCond'].value_counts())
def gar(x):
    return 1 if x=='TA' else 0
train['GarageCond']=train.GarageCond.apply(gar)
test['GarageCond']=test.GarageCond.apply(gar)
#%%
#GarageQual 
print(train['GarageQual'].value_counts())
train['GarageQual']=train.GarageQual.apply(gar)
test['GarageQual']=test.GarageQual.apply(gar)
#%%
#GarageFinish
dummies1=pd.get_dummies(train.GarageFinish)
dummies2=pd.get_dummies(test.GarageFinish)
train=pd.concat([train,dummies1],axis='columns')
train=train.drop(['GarageFinish','Unf'],axis='columns')
test=pd.concat([test,dummies2],axis='columns')
test=test.drop(['GarageFinish','Unf'],axis='columns')

#%%
#GarageType 
#print(train['GarageType'].value_counts())
dummies3=pd.get_dummies(train.GarageType)
dummies4=pd.get_dummies(test.GarageType)
train=pd.concat([train,dummies3],axis='columns')
test=pd.concat([train,dummies4],axis='columns')
train=train.drop(['GarageType','Attchd'],axis='columns')
test=test.drop(['GarageType','Attchd'],axis='columns')


#%%
#FireplaceQu 
#print(test['FireplaceQu'].value_counts())
dummies5=pd.get_dummies(train.FireplaceQu)
dummies6=pd.get_dummies(test.FireplaceQu)
train=pd.concat([train,dummies5],axis='columns')
test=pd.concat([test,dummies6],axis='columns')
train=train.drop(['FireplaceQu','Gd'],axis='columns')
test=test.drop(['FireplaceQu','Gd'],axis='columns')

#%%
#Functional
#print(train['Functional'].value_counts())
def fun(x):
    return 1 if x=='Typ'else 0
train['Functional']=train.Functional.apply(fun)
test['Functional']=test.Functional.apply(fun)
#%%
#KitchenQual 
#print(train['KitchenQual'].value_counts())
dummies7=pd.get_dummies(train.KitchenQual)
dummies8=pd.get_dummies(test.KitchenQual)
train=pd.concat([train,dummies7],axis='columns')
test=pd.concat([test,dummies8],axis='columns')
train=train.drop(['KitchenQual','TA'],axis='columns')
test=test.drop(['KitchenQual','TA'],axis='columns')

#%%
#Electrical
print(train['Electrical'].value_counts()) 
def ele(x):
    return 1 if x=='SBrkr' else 0
train['Electrical']=train.Electrical.apply(ele)
test['Electrical']=test.Electrical.apply(ele)
#%%
#  CentralAir
print(train['CentralAir'].value_counts())
def cen(x):
    return 1 if x=='Y' else 0
train['CentralAir']=train.CentralAir.apply(cen)
test['CentralAir']=test.CentralAir.apply(cen)
#%%
#HeatingQC
#print(train['HeatingQC'].value_counts())
dummies9=pd.get_dummies(train.HeatingQC)
dummies10=pd.get_dummies(test.HeatingQC)
train=pd.concat([train,dummies9],axis='columns')
test=pd.concat([test,dummies10],axis='columns')
train=train.drop(['HeatingQC','Ex'],axis='columns')
test=test.drop(['HeatingQC','Ex'],axis='columns')

#%%
#Heating
print(train['Heating'].value_counts())
def heat(x):
    return 1 if x=='GasA' else 0
train['Heating']=train.Heating.apply(heat)
test['Heating']=test.Heating.apply(heat)
#%%
#BsmtFinType2
print(train['BsmtFinType2'].value_counts())
def Bsmt(x):
    return 1 if x=='Unf' else 0
train ['BsmtFinType2']=train.BsmtFinType2.apply(Bsmt)
test['BsmtFinType2']=test.BsmtFinType2.apply(Bsmt)      
#%%  
#BsmtFinType1
#print(train['BsmtFinType1'].value_counts())
dummies11=pd.get_dummies(train.BsmtFinType1)
dummies12=pd.get_dummies(test.BsmtFinType1)
train=pd.concat([train,dummies11],axis='columns')
test=pd.concat([test,dummies12],axis='columns')
train=train.drop(['BsmtFinType1','Unf'],axis='columns')
test=test.drop(['BsmtFinType1','Unf'],axis='columns')

#%%
#BsmtExposure 
#print(train['BsmtExposure'].value_counts())
dummies13=pd.get_dummies(train.BsmtExposure)
dummies14=pd.get_dummies(test.BsmtExposure)
train=pd.concat([train,dummies13],axis='columns')
test=pd.concat([test,dummies14],axis='columns')
train=train.drop(['BsmtExposure','Mn'],axis='columns')
test=test.drop(['BsmtExposure','Mn'],axis='columns')
#%%
#BsmtCond
print(train['BsmtCond'].value_counts())
def bsm(x):
    return 1 if x=='TA' else 0
train['BsmtCond']=train.BsmtCond.apply(bsm)
test['BsmtCond']=test.BsmtCond.apply(bsm)
#%%
#BsmtQual
#print(train['BsmtQual'].value_counts())
dummies15=pd.get_dummies(train.BsmtQual)
dummies16=pd.get_dummies(test.BsmtQual)
train=pd.concat([train,dummies15],axis='columns')
test=pd.concat([test,dummies16],axis='columns')
train=train.drop(['BsmtQual','Ex'],axis='columns')
test=test.drop(['BsmtQual','Ex'],axis='columns')

#%%
#Foundation
print(train['Foundation'].value_counts())
def fond(x):
    return 1 if x=='PConc' or x=='CBlock' else 0
train['Foundation']=train.Foundation.apply(fond)
test['Foundation']=test.Foundation.apply(fond)
#%%
#ExterCond
print(train['ExterCond'].value_counts())
train['ExterCond']=train.ExterCond.apply(bsm)
test['ExterCond']=test.ExterCond.apply(bsm)
#%%
#ExterQual
#print(train['ExterQual'].value_counts())
dummies17=pd.get_dummies(train.ExterQual)
dummies18=pd.get_dummies(test.ExterQual)
train=pd.concat([train,dummies17],axis='columns')
test=pd.concat([test,dummies18],axis='columns')
train=train.drop(['ExterQual','Fa'],axis='columns')
test=test.drop(['ExterQual','Fa'],axis='columns')

#%%
#MasVnrType 
print(test['MasVnrType'].value_counts())
dummies19=pd.get_dummies(train.MasVnrType)
dummies20=pd.get_dummies(test.MasVnrType)
train=pd.concat([train,dummies19],axis='columns')
test=pd.concat([test,dummies20],axis='columns')
train=train.drop(['MasVnrType','Stone'],axis='columns')
test=test.drop(['MasVnrType','Stone'],axis='columns')

#%%
#Exterior2nd
#print(test['Exterior2nd'].value_counts())
dummies21=pd.get_dummies(train.Exterior2nd)
dummies22=pd.get_dummies(test.Exterior2nd)
train=pd.concat([train,dummies21],axis='columns')
test=pd.concat([test,dummies22],axis='columns')
train=train.drop(['Exterior2nd','Stone'],axis='columns')
test=test.drop(['Exterior2nd','Stone'],axis='columns')
#%%
#Exterior1st
#print(train['Exterior1st'].value_counts())
dummies23=pd.get_dummies(train.Exterior1st)
dummies24=pd.get_dummies(test.Exterior1st)
train=pd.concat([train,dummies23],axis='columns')
test=pd.concat([test,dummies24],axis='columns')
train=train.drop(['Exterior1st','HdBoard'],axis='columns')
test=test.drop(['Exterior1st','HdBoard'],axis='columns')
#%%
#RoofMatl
print(train['RoofMatl'].value_counts())
def roof(x):
    return 1 if x=='CompShg' else 0
train['RoofMatl']=train.RoofMatl.apply(roof)
test['RoofMatl']=test.RoofMatl.apply(roof)
#%%
#RoofStyle
print(train['RoofStyle'].value_counts())
def style(x):
    return 1 if x=="Gable" else 0
train['RoofStyle']=train.RoofStyle.apply(style)
test['RoofStyle']=test.RoofStyle.apply(style)
#%%
#HouseStyle
#print(train['HouseStyle'].value_counts())
dummies25=pd.get_dummies(train.HouseStyle)
dummies26=pd.get_dummies(test.HouseStyle)
train=pd.concat([train,dummies25],axis='columns')
test=pd.concat([test,dummies26],axis='columns')
train=train.drop(['HouseStyle','2.5Unf'],axis='columns')
test=test.drop(['HouseStyle','2.5Unf'],axis='columns')
#%%
#BldgType
print(train['BldgType'].value_counts())
def bl(x):
    return 1 if x=='1Fam' else 0
train['BldgType']=train.BldgType.apply(bl)
test['BldgType']=test.BldgType.apply(bl)
#%%
#Condition2
print(train['Condition2'].value_counts())
def con(x):
    return 1 if x=='Norm' else 0
train['Condition2']=train.Condition2.apply(con)
test['Condition2']=test.Condition2.apply(con)
#%%
#Condition1
print(train['Condition1'].value_counts())
def con(x):
    return 1 if x=='Norm' else 0
train['Condition1']=train.Condition1.apply(con)
test['Condition1']=test.Condition1.apply(con)
#%%
#Neighborhood  
#print(test['Neighborhood'].value_counts())

dummies27=pd.get_dummies(train.Neighborhood)
dummies28=pd.get_dummies(test.Neighborhood)
train=pd.concat([train,dummies27],axis='columns')
test=pd.concat([test,dummies28],axis='columns')
train=train.drop(['Neighborhood','Edwards'],axis='columns')
test=test.drop(['Neighborhood','Edwards'],axis='columns')


#%%
#Alley
#print(train['Alley'].value_counts())
dummies29=pd.get_dummies(train.Alley)
dummies30=pd.get_dummies(test.Alley)
train=pd.concat([train,dummies29],axis='columns')
test=pd.concat([test,dummies30],axis='columns')
train=train.drop(['Alley','Grvl'],axis='columns')
test=test.drop(['Alley','Grvl'],axis='columns')
#%%
#LotShape
#print(train['LotShape'].value_counts())
dummies31=pd.get_dummies(train.LotShape)
dummies32=pd.get_dummies(test.LotShape)
train=pd.concat([train,dummies31],axis='columns')
test=pd.concat([test,dummies32],axis='columns')
train=train.drop(['LotShape','IR1'],axis='columns')
test=test.drop(['LotShape','IR1'],axis='columns')

train=train.interpolate().dropna()
print(sum(train.isnull().sum() != 0))




#%%
y= train.SalePrice
X=train.drop(['SalePrice','Id'],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,
                                                test_size=0.33)

from sklearn import linear_model
lr=linear_model.LinearRegression()

model=lr.fit(X_train,y_train)

print('R^2: ',model.score(X_train,y_train))

predictions=model.predict(X_test)

from sklearn.metrics import mean_squared_error
print('RMSE: ',mean_squared_error(y_test,predictions))


submission = pd.DataFrame()
submission['Id'] = test.Id
feats = test.drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions = np.exp(predictions)
print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])
submission['SalePrice'] = final_predictions
submission.head()
submission.to_csv('submission2.csv', index=False)