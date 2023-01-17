import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame

%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.info()
train = train.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1)
test = test.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1)
#encoding MSZoning to numeric values
labels = train['MSZoning'].astype('category').cat.categories.tolist()
replace_map = {'MSZoning' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
train.replace(replace_map,inplace=True)



#encoding Street to numeric values
labels = train['Street'].astype('category').cat.categories.tolist()
replace_map = {'Street' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
train.replace(replace_map,inplace=True)



#encoding LotShape to numeric values
replace_map = {'LotShape' : {'Reg':4,'IR1':3,'IR2':2,'IR3':1}}
train.replace(replace_map,inplace=True)



#encoding LandContour to numeric values
replace_map = {'LandContour' : {'Lvl':4,'Bnk':3,'HLS':2,'Low':1}}
train.replace(replace_map,inplace=True)



#encoding Utilities to numeric values
replace_map = {'Utilities' : {'AllPub':4,'NoSewr':3,'NoSeWa':2,'ELO':1}}
train.replace(replace_map,inplace=True)



#encoding LotConfig to numeric values
replace_map = {'LotConfig' : {'Inside':5,'Corner':4,'CulDSac':3,'FR2':2,'FR3':1}}
train.replace(replace_map,inplace=True)



#encoding LandSlope to numeric values
replace_map = {'LandSlope' : {'Gtl':3,'Mod':2,'Sev':1}}
train.replace(replace_map,inplace=True)



#encoding Neighborhood to numeric values
labels = train['Neighborhood'].astype('category').cat.categories.tolist()
replace_map = {'Neighborhood' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
train.replace(replace_map,inplace=True)



#encoding Condition1 to numeric values
labels = train['Condition1'].astype('category').cat.categories.tolist()
replace_map = {'Condition1' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
train.replace(replace_map,inplace=True)



#encoding Condition2 to numeric values
labels = train['Condition2'].astype('category').cat.categories.tolist()
replace_map = {'Condition2' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
train.replace(replace_map,inplace=True)



#encoding BldgType to numeric values
replace_map = {'BldgType' : {'1Fam':5,'2fmCon':4,'Duplex':3,'TwnhsE':2,'Twnhs':1}}
train.replace(replace_map,inplace=True)



#encoding HouseStyle to numeric values
replace_map = {'HouseStyle' : {'1Story':8,'1.5Fin':7,'1.5Unf':6,'2Story':5, \
                               '2.5Fin':4, '2.5Unf':3, 'SFoyer':2, 'SLvl':1}}
train.replace(replace_map,inplace=True)



#encoding RoofStyle to numeric values
replace_map = {'RoofStyle' : {'Flat':6,'Gable':5,'Gambrel':4,'Hip':3, 'Mansard':2, 'Shed':1}}
train.replace(replace_map,inplace=True)



#encoding RoofMatl to numeric values
replace_map = {'RoofMatl' : {'ClyTile':1,'CompShg':2,'Membran':3,'Metal':4, \
                             'Roll':5, 'Tar&Grv':6, 'WdShake':7, 'WdShngl':8}}
train.replace(replace_map,inplace=True)



#encoding Exterior1st to numeric values
labels = train['Exterior1st'].astype('category').cat.categories.tolist()
replace_map = {'Exterior1st' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
train.replace(replace_map,inplace=True)



#encoding Exterior2nd to numeric values
labels = train['Exterior2nd'].astype('category').cat.categories.tolist()
replace_map = {'Exterior2nd' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
train.replace(replace_map,inplace=True)



#encoding MasVnrType to numeric values
replace_map = {'MasVnrType' : {'BrkCmn':1,'BrkFace':2,'CBlock':3,'None':-1,'Stone':4}}
train.replace(replace_map,inplace=True)



#encoding ExterQual to numeric values
replace_map = {'ExterQual' : {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':-1}}
train.replace(replace_map,inplace=True)



#encoding ExterCond to numeric values
replace_map = {'ExterCond' : {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':-1}}
train.replace(replace_map,inplace=True)



#encoding Foundation to numeric values
replace_map = {'Foundation' : {'BrkTil':1,'CBlock':2,'PConc':3,'Slab':4,'Stone':5, 'Wood':6}}
train.replace(replace_map,inplace=True)



#encoding BsmtCond to numeric values
replace_map = {'BsmtCond' : {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2, 'NA':-1}}
train.replace(replace_map,inplace=True)



#encoding BsmtExposure to numeric values
replace_map = {'BsmtExposure' : {'Gd':5,'Av':4,'Mn':3,'No':1, 'NA':-1}}
train.replace(replace_map,inplace=True)



#encoding BsmtFinType1 to numeric values
replace_map = {'BsmtFinType1' : {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1, 'NA':-1}}
train.replace(replace_map,inplace=True)



#encoding BsmtFinType2 to numeric values
replace_map = {'BsmtFinType2' : {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1, 'NA':-1}}
train.replace(replace_map,inplace=True)



#encoding Heating to numeric values
replace_map = {'Heating' : {'Floor':6,'GasA':5,'GasW':4,'Grav':3,'OthW':2,'Wall':1}}
train.replace(replace_map,inplace=True)



#encoding HeatingQC to numeric values
replace_map = {'HeatingQC' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
train.replace(replace_map,inplace=True)



#encoding CentralAir to numeric values
replace_map = {'CentralAir' : {'N':0,'Y':1}}
train.replace(replace_map,inplace=True)



#encoding Electrical to numeric values
replace_map = {'Electrical' : {'SBrkr':5,'FuseA':4,'FuseF':3,'FuseP':2,'Mix':1}}
train.replace(replace_map,inplace=True)



#encoding KitchenQual to numeric values
replace_map = {'KitchenQual' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
train.replace(replace_map,inplace=True)



#encoding Functional to numeric values
replace_map = {'Functional' : {'Typ':4,'Min1':3,'Min2':2,'Mod':1,'Maj1':-1, 'Maj2':-2 , 'Sev':-3, 'Sal':-4 }}
train.replace(replace_map,inplace=True)



#encoding GarageType to numeric values
replace_map = {'GarageType' : {'2Types':6,'Attchd':5,'Basment':4,'BuiltIn':3,'CarPort':2, 'Detchd': 1, 'NA': -1 }}
train.replace(replace_map,inplace=True)



#encoding GarageFinish to numeric values
replace_map = {'GarageFinish' : {'Fin':3,'RFn':2, 'Unf':1,'NA':-1}}
train.replace(replace_map,inplace=True)



#encoding GarageQual to numeric values
replace_map = {'GarageQual' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
train.replace(replace_map,inplace=True)



#encoding GarageCond to numeric values
replace_map = {'GarageCond' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
train.replace(replace_map,inplace=True)



#encoding PavedDrive to numeric values
replace_map = {'PavedDrive' : {'Y':3,'P':2,'N':1}}
train.replace(replace_map,inplace=True)



#encoding SaleType to numeric values
replace_map = {'SaleType' : {'WD':10,'CWD':9,'VWD':8,'New':7,'COD':6,'Con':5,'ConLw':4,'ConLI':3,'ConLD':2,'Oth':1}}
train.replace(replace_map,inplace=True)



#encoding SaleCondition to numeric values
replace_map = {'SaleCondition' : {'Normal':6,'Abnorml':5,'AdjLand':4, 'Alloca':3, 'Family':2, 'Partial':1}}
train.replace(replace_map,inplace=True)
#encoding BsmtQual to numeric values
replace_map = {'BsmtQual' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
train.replace(replace_map,inplace=True)
#encoding MSZoning to numeric values
labels = test['MSZoning'].astype('category').cat.categories.tolist()
replace_map = {'MSZoning' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
test.replace(replace_map,inplace=True)



#encoding Street to numeric values
labels = test['Street'].astype('category').cat.categories.tolist()
replace_map = {'Street' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
test.replace(replace_map,inplace=True)



#encoding LotShape to numeric values
replace_map = {'LotShape' : {'Reg':4,'IR1':3,'IR2':2,'IR3':1}}
test.replace(replace_map,inplace=True)



#encoding LandContour to numeric values
replace_map = {'LandContour' : {'Lvl':4,'Bnk':3,'HLS':2,'Low':1}}
test.replace(replace_map,inplace=True)



#encoding Utilities to numeric values
replace_map = {'Utilities' : {'AllPub':4,'NoSewr':3,'NoSeWa':2,'ELO':1}}
test.replace(replace_map,inplace=True)



#encoding LotConfig to numeric values
replace_map = {'LotConfig' : {'Inside':5,'Corner':4,'CulDSac':3,'FR2':2,'FR3':1}}
test.replace(replace_map,inplace=True)



#encoding LandSlope to numeric values
replace_map = {'LandSlope' : {'Gtl':3,'Mod':2,'Sev':1}}
test.replace(replace_map,inplace=True)



#encoding Neighborhood to numeric values
labels = test['Neighborhood'].astype('category').cat.categories.tolist()
replace_map = {'Neighborhood' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
test.replace(replace_map,inplace=True)



#encoding Condition1 to numeric values
labels = test['Condition1'].astype('category').cat.categories.tolist()
replace_map = {'Condition1' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
test.replace(replace_map,inplace=True)



#encoding Condition2 to numeric values
labels = test['Condition2'].astype('category').cat.categories.tolist()
replace_map = {'Condition2' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
test.replace(replace_map,inplace=True)



#encoding BldgType to numeric values
replace_map = {'BldgType' : {'1Fam':5,'2fmCon':4,'Duplex':3,'TwnhsE':2,'Twnhs':1}}
test.replace(replace_map,inplace=True)



#encoding HouseStyle to numeric values
replace_map = {'HouseStyle' : {'1Story':8,'1.5Fin':7,'1.5Unf':6,'2Story':5, \
                               '2.5Fin':4, '2.5Unf':3, 'SFoyer':2, 'SLvl':1}}
test.replace(replace_map,inplace=True)



#encoding RoofStyle to numeric values
replace_map = {'RoofStyle' : {'Flat':6,'Gable':5,'Gambrel':4,'Hip':3, 'Mansard':2, 'Shed':1}}
test.replace(replace_map,inplace=True)



#encoding RoofMatl to numeric values
replace_map = {'RoofMatl' : {'ClyTile':1,'CompShg':2,'Membran':3,'Metal':4, \
                             'Roll':5, 'Tar&Grv':6, 'WdShake':7, 'WdShngl':8}}
test.replace(replace_map,inplace=True)



#encoding Exterior1st to numeric values
labels = test['Exterior1st'].astype('category').cat.categories.tolist()
replace_map = {'Exterior1st' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
test.replace(replace_map,inplace=True)



#encoding Exterior2nd to numeric values
labels = test['Exterior2nd'].astype('category').cat.categories.tolist()
replace_map = {'Exterior2nd' : {k:v for k,v in zip(labels,range(1,len(labels)+1))}}
test.replace(replace_map,inplace=True)



#encoding MasVnrType to numeric values
replace_map = {'MasVnrType' : {'BrkCmn':1,'BrkFace':2,'CBlock':3,'None':-1,'Stone':4}}
test.replace(replace_map,inplace=True)



#encoding ExterQual to numeric values
replace_map = {'ExterQual' : {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':-1}}
test.replace(replace_map,inplace=True)



#encoding ExterCond to numeric values
replace_map = {'ExterCond' : {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':-1}}
test.replace(replace_map,inplace=True)



#encoding Foundation to numeric values
replace_map = {'Foundation' : {'BrkTil':1,'CBlock':2,'PConc':3,'Slab':4,'Stone':5, 'Wood':6}}
test.replace(replace_map,inplace=True)



#encoding BsmtCond to numeric values
replace_map = {'BsmtCond' : {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2, 'NA':-1}}
test.replace(replace_map,inplace=True)



#encoding BsmtExposure to numeric values
replace_map = {'BsmtExposure' : {'Gd':5,'Av':4,'Mn':3,'No':1, 'NA':-1}}
test.replace(replace_map,inplace=True)



#encoding BsmtFinType1 to numeric values
replace_map = {'BsmtFinType1' : {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1, 'NA':-1}}
test.replace(replace_map,inplace=True)



#encoding BsmtFinType2 to numeric values
replace_map = {'BsmtFinType2' : {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1, 'NA':-1}}
test.replace(replace_map,inplace=True)



#encoding Heating to numeric values
replace_map = {'Heating' : {'Floor':6,'GasA':5,'GasW':4,'Grav':3,'OthW':2,'Wall':1}}
test.replace(replace_map,inplace=True)



#encoding HeatingQC to numeric values
replace_map = {'HeatingQC' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
test.replace(replace_map,inplace=True)



#encoding CentralAir to numeric values
replace_map = {'CentralAir' : {'N':0,'Y':1}}
test.replace(replace_map,inplace=True)



#encoding Electrical to numeric values
replace_map = {'Electrical' : {'SBrkr':5,'FuseA':4,'FuseF':3,'FuseP':2,'Mix':1}}
test.replace(replace_map,inplace=True)



#encoding KitchenQual to numeric values
replace_map = {'KitchenQual' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
test.replace(replace_map,inplace=True)



#encoding Functional to numeric values
replace_map = {'Functional' : {'Typ':4,'Min1':3,'Min2':2,'Mod':1,'Maj1':-1, 'Maj2':-2 , 'Sev':-3, 'Sal':-4 }}
test.replace(replace_map,inplace=True)



#encoding GarageType to numeric values
replace_map = {'GarageType' : {'2Types':6,'Attchd':5,'Basment':4,'BuiltIn':3,'CarPort':2, 'Detchd': 1, 'NA': -1 }}
test.replace(replace_map,inplace=True)



#encoding GarageFinish to numeric values
replace_map = {'GarageFinish' : {'Fin':3,'RFn':2, 'Unf':1,'NA':-1}}
test.replace(replace_map,inplace=True)



#encoding GarageQual to numeric values
replace_map = {'GarageQual' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
test.replace(replace_map,inplace=True)



#encoding GarageCond to numeric values
replace_map = {'GarageCond' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
test.replace(replace_map,inplace=True)



#encoding PavedDrive to numeric values
replace_map = {'PavedDrive' : {'Y':3,'P':2,'N':1}}
test.replace(replace_map,inplace=True)



#encoding SaleType to numeric values
replace_map = {'SaleType' : {'WD':10,'CWD':9,'VWD':8,'New':7,'COD':6,'Con':5,'ConLw':4,'ConLI':3,'ConLD':2,'Oth':1}}
test.replace(replace_map,inplace=True)



#encoding SaleCondition to numeric values
replace_map = {'SaleCondition' : {'Normal':6,'Abnorml':5,'AdjLand':4, 'Alloca':3, 'Family':2, 'Partial':1}}
test.replace(replace_map,inplace=True)
#encoding BsmtQual to numeric values
replace_map = {'BsmtQual' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}}
test.replace(replace_map,inplace=True)
test.info()
train['YearOld'] = train['YrSold'] - train['YearRemodAdd']
test['YearOld'] = test['YrSold'] - test['YearRemodAdd']
train = train.drop(['YearBuilt','YrSold','YearRemodAdd'],axis=1)
test = test.drop(['YearBuilt','YrSold','YearRemodAdd'],axis=1)
train['ExterQualCond'] = (train['ExterQual']+train['ExterCond'])/2
test['ExterQualCond'] = (test['ExterQual']+test['ExterCond'])/2
train['BsmtQualCond'] = (train['BsmtQual']+train['BsmtCond'])/2
test['BsmtQualCond'] = (test['BsmtQual']+test['BsmtCond'])/2
train['GarageQualCond'] = (train['GarageQual']+train['GarageCond'])/2
test['GarageQualCond'] = (test['GarageQual']+test['GarageCond'])/2
train = train.drop(['ExterQual','ExterCond','BsmtQual','BsmtCond','GarageQual','GarageCond'],axis=1)
test = test.drop(['ExterQual','ExterCond','BsmtQual','BsmtCond','GarageQual','GarageCond'],axis=1)
train['TotPorch'] = train['OpenPorchSF']+train['EnclosedPorch']+train['3SsnPorch']+train['ScreenPorch']
test['TotPorch'] = test['OpenPorchSF']+test['EnclosedPorch']+test['3SsnPorch']+test['ScreenPorch']
train = train.drop(['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'],axis=1)
test = test.drop(['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'],axis=1)
SalePrice = train['SalePrice']
train = train.drop(['SalePrice'],axis=1)
train['SalePrice'] = SalePrice
train.head()
train = train.drop(['Id'],axis=1)
test = test.drop(['Id'],axis=1)
correlation_matrix = train.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation_matrix, vmin=-0.75, vmax=0.75, square=True)

plt.scatter(train['LotArea'],train['SalePrice'])
train = train[train['LotArea']<55000]
plt.scatter(train['GrLivArea'],train['SalePrice'])
train = train[train['GrLivArea']<4500]
plt.scatter(train['TotalBsmtSF'],train['SalePrice'])
train = train[train['TotalBsmtSF']<2500]
train = train[train['TotalBsmtSF']>0]
plt.scatter(train['GarageCars'],train['SalePrice'])
train = train[train['GarageCars']<3.75]
plt.scatter(train['GarageArea'],train['SalePrice'])
train = train[train['GarageArea']<1200]
train = train[train['GarageArea']>0]
sns.distplot(train['SalePrice'])
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'])
plt.figure(figsize=(20,20))
sns.jointplot(train['LotFrontage'],train['SalePrice'],kind="hex")
sns.jointplot(train['LotArea'],train['SalePrice'],kind="hex",color="indianred")
sns.jointplot(train['OverallQual'],train['SalePrice'],kind='kde')
x = train.iloc[:,:-1].values
y = train.iloc[:,-1].values

x_eval = test.iloc[:,:].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(x[:, :])
x[:,:] = imputer.transform(x[:,:])
from sklearn.preprocessing import Imputer
imputer1 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer1 = imputer1.fit(x_eval[:, :])
x_eval[:,:] = imputer1.transform(x_eval[:,:])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

rmse_dict = {}

regressors = [
    GradientBoostingRegressor(n_estimators = 5000, random_state = 0),
    RandomForestRegressor(n_estimators = 5000, random_state = 0),
    DecisionTreeRegressor(random_state=0),
    SVR(kernel='rbf')
]

for reg in regressors:
    name = reg.__class__.__name__
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    error = mean_squared_error(y_test,y_pred) 
    rmse_dict[name] = error
names=[]
rmse=[]
for key,value in rmse_dict.items():
    names.append(key)
    rmse.append(value)
df_vals = DataFrame(list(zip(names,rmse)),columns=['RMSE','Regressors'])
plt.figure(figsize=(15,5))
plt.title('RMSE values of different Regressors on our data')
sns.set_color_codes("muted")
sns.barplot(x='Regressors', y='RMSE', data=df_vals, palette="Blues")
regressor1 = RandomForestRegressor(n_estimators = 5000, random_state = 0)
regressor1.fit(x_train,y_train)
y_pred = regressor1.predict(x_test)


y_test = np.expm1(y_test)
y_pred = np.expm1(y_pred)
DataFrame([y_test,y_pred])
predictions = regressor1.predict(x_eval)
predictions = np.expm1(predictions)
predictions = DataFrame(predictions)
predictions.to_csv("pred_new.csv")
predictions