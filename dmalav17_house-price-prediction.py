#importing the libraries 

import numpy as np 

import matplotlib.pyplot as plt

import pandas as pd

#importing the train and test data 

train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")
train_data.head()
test_data.head()
#shape of train and test data \

train_data.shape,test_data.shape
#take care of missing data in training set

train_data.isnull().sum()[train_data.isnull().sum()>0]




#take care of missing data in test set

test_data.isnull().sum()[test_data.isnull().sum()>0]



#draw a correlation matrices b/w the variables 

corr=train_data.corr()

plt.figure(figsize=(10,10))

#importing seaborn library 

import seaborn as sns

sns.heatmap(train_data[corr.index[abs(corr['SalePrice'])>0.4]].corr(),annot=True)





#filling the missing values 

fillnull={'Alley':'No Alley','PoolQC':'NoPool','Fence':'NoFence','MiscFeature':'NoMisc','FireplaceQu':'NoFire'}

train_data.fillna(fillnull,inplace=True)

test_data.fillna(fillnull,inplace=True)
train_data.head()
test_data.head()


#Detection of outlier

plt.scatter(train_data['Id'],train_data['GrLivArea'],color='red')

#train_data.drop(train)
train_data.drop(train_data[train_data.GrLivArea>4000].index.values,inplace=True)
train_data.shape

#shape is reduced
#checking for outliers 

plt.figure(figsize=(5,5))

plt.scatter(train_data['Id'],train_data['MSSubClass'],color='blue')
plt.scatter(train_data['LotFrontage'],train_data['LotArea'],color='green')

plt.xlabel('LotFrontage')

plt.ylabel('LotArea')
train_data.drop(train_data[train_data.LotFrontage>250].index.values,inplace=True)

train_data.drop(train_data[train_data.LotArea>100000].index.values,inplace=True)

train_data.shape
plt.scatter(train_data['LotFrontage'],train_data['LotArea'],color='green')

plt.xlabel('LotFrontage')

plt.ylabel('LotArea')
from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

train_data['LotFrontage']=imputer.fit_transform(train_data['LotFrontage'].values.reshape(-1,1))

test_data['LotFrontage']=imputer.fit_transform(test_data['LotFrontage'].values.reshape(-1,1))

test_data.GarageYrBlt.describe()
test_data[test_data.GarageYrBlt>2018].GarageYrBlt
test_data.GarageYrBlt.replace({2207:2007},inplace=True)
fillnull={'BsmtQual':'NoBsmt','BsmtCond':'NoBsmt','BsmtExposure':'NoBsmt','BsmtFinType1':'NoBsmt','BsmtFinSF1':'NoBsmt','BsmtFinType2':'NoBsmt','GarageType':'NoGrg','GarageFinish':'NoGrg','GarageQual':'NoGrg','GarageCond':'NoGrg'}

train_data.fillna(fillnull,inplace=True)

test_data.fillna(fillnull,inplace=True)
train_data.MasVnrType.fillna('None',inplace=True)

test_data.MasVnrType.fillna('None',inplace=True)

train_data.MasVnrArea.fillna(0,inplace=True)

test_data.MasVnrArea.fillna(0,inplace=True)

train_data.Electrical.fillna('SBrkr',inplace=True)
fillwithmode=['MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType']

for column in fillwithmode:

    mode=train_data[column].mode()[0]

    test_data[column].fillna(mode,inplace=True)

fillwithzero=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']

for column in fillwithzero:

    test_data[column].fillna(0,inplace=True)

fillwithmean=['GarageArea','GarageCars']

for column in fillwithmean:

    mean=train_data[column].mean()

    test_data[column].fillna(mean,inplace=True)

test_data.isnull().sum()[test_data.isnull().sum()>0]
train_data.isnull().sum()[train_data.isnull().sum()>0]
#Labeling, creating dummies and feature engineering

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

dummies=pd.get_dummies(train_data.MSZoning,prefix='MSZoning')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.MSZoning,prefix='MSZoning')

test_data=pd.concat([test_data,dummies],axis=1)

train_data.loc[:,'Street']=le.fit_transform(train_data.Street.values)

test_data.loc[:,'Street']=le.fit_transform(test_data.Street.values)

mapping={'No Alley':0,'Grvl':1,'Pave':2}

train_data.Alley.replace(mapping,inplace=True)

test_data.Alley.replace(mapping,inplace=True)


mapping={'Low':0,'HLS':1,'Bnk':2,'Lvl':3}

train_data.LandContour.replace(mapping,inplace=True)

test_data.LandContour.replace(mapping,inplace=True)

mapping={'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':0}

train_data.Utilities.replace(mapping,inplace=True)

test_data.Utilities.replace(mapping,inplace=True)

dummies=pd.get_dummies(train_data.LotConfig,prefix='LotConfig')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.LotConfig,prefix='LotConfig')

test_data=pd.concat([test_data,dummies],axis=1)
mappingg={'IR3':0,'IR2':1,'IR1':2,'Reg':3}

train_data.LotShape.replace(mapping,inplace=True)

test_data.LotShape.replace(mapping,inplace=True)


dummies=pd.get_dummies(train_data.Neighborhood,prefix='Neighborhood')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.Neighborhood,prefix='Neighborhood')

test_data=pd.concat([test_data,dummies],axis=1)

dummies=pd.get_dummies(train_data.Condition1,prefix='Cond1')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.Condition1,prefix='Cond1')

test_data=pd.concat([test_data,dummies],axis=1)

dummies=pd.get_dummies(train_data.Condition2,prefix='Cond2')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.Condition2,prefix='Cond2')

test_data=pd.concat([test_data,dummies],axis=1)



mapping={'Gtl':0,'Mod':1,'Sev':2}

train_data.LandSlope.replace(mapping,inplace=True)

test_data.LandSlope.replace(mapping,inplace=True)
dummies=pd.get_dummies(train_data.BldgType,prefix='BldgType')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.BldgType,prefix='BldgType')

test_data=pd.concat([test_data,dummies],axis=1)
dummies=pd.get_dummies(train_data.HouseStyle,prefix='HouseStyle')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.HouseStyle,prefix='HouseStyle')

test_data=pd.concat([test_data,dummies],axis=1)

dummies=pd.get_dummies(train_data.RoofStyle,prefix='RoofStyle')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.RoofStyle,prefix='RoofStyle')

test_data=pd.concat([test_data,dummies],axis=1)

dummies=pd.get_dummies(train_data.RoofMatl,prefix='RoofMatl')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.RoofMatl,prefix='RoofMatl')

test_data=pd.concat([test_data,dummies],axis=1)

train_data.Exterior1st.unique()
test_data.Exterior1st.unique()
train_data.Exterior2nd.unique()
test_data.Exterior2nd.unique()
#we can see some wrong spell data in Exterior2nd attributes

mapping={ 'Wd Shng':'WdShing','Brk Cmn': 'BrkComm','CmentBd': 'CemntBd'}

train_data.Exterior2nd.replace(mapping,inplace=True)

test_data.Exterior2nd.replace(mapping,inplace=True)

a=train_data.Exterior1st.value_counts()

b=train_data.Exterior2nd.value_counts()

c=pd.concat([a,b],axis=1,sort=True)

c.plot.bar(stacked=False)
a=test_data.Exterior1st.value_counts()

b=test_data.Exterior2nd.value_counts()

c=pd.concat([a,b],axis=1,sort=True)

c.plot.bar(stacked=False)
dummies=pd.get_dummies(train_data.Exterior1st,prefix='Ext1')

train_data['Ext1_Other']=0

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.Exterior1st,prefix='Ext1')

dummies['Ext1_ImStucc']=0

dummies['Ext1_Stone']=0

dummies['Ext1_Other']=0

test_data=pd.concat([test_data,dummies],axis=1)

dummies=pd.get_dummies(train_data.Exterior2nd,prefix='Ext2')

dummies['Ext2_Other']=0

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.Exterior2nd,prefix='Ext2')

test_data=pd.concat([test_data,dummies],axis=1)
dummies=pd.get_dummies(train_data.MasVnrType,prefix='MasVnrType')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.MasVnrType,prefix='MasVnrType')

test_data=pd.concat([test_data,dummies],axis=1)
plt.scatter(train_data['Id'],train_data['MasVnrArea'],color='yellow')

plt.show()
train_data.drop(train_data[train_data.MasVnrArea>1200].index.values,inplace=True)

train_data.shape
mapping={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1 }

train_data.ExterQual.replace(mapping,inplace=True)

test_data.ExterQual.replace(mapping,inplace=True)

train_data.ExterCond.replace(mapping,inplace=True)

test_data.ExterCond.replace(mapping,inplace=True)

dummies=pd.get_dummies(train_data.Foundation,prefix='Foundation')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.Foundation,prefix='Foundation')

test_data=pd.concat([test_data,dummies],axis=1)

train_data['NoBsmt']=(train_data.BsmtQual=='NoBsmt')*1

test_data['NoBsmt']=(test_data.BsmtQual=='NoBsmt')*1

mapping={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5,'NoBsmt':0}

train_data.BsmtCond.replace(mapping,inplace=True)

test_data.BsmtCond.replace(mapping,inplace=True)

mapping={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5,'NoBsmt':0}

train_data.BsmtQual.replace(mapping,inplace=True)

test_data.BsmtQual.replace(mapping,inplace=True)

mapping={'Gd':4,'Av':3,'Mn':2,'No':1,'NoBsmt':0}

train_data.BsmtExposure.replace(mapping,inplace=True)

test_data.BsmtExposure.replace(mapping,inplace=True)
mapping={'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBsmt':0}

train_data.BsmtFinType1.replace(mapping,inplace=True)

test_data.BsmtFinType1.replace(mapping,inplace=True)

train_data.BsmtFinType2.replace(mapping,inplace=True)

test_data.BsmtFinType2.replace(mapping,inplace=True)
dummies=pd.get_dummies(train_data.Heating,prefix='Heating')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.Heating,prefix='Heating')

test_data=pd.concat([test_data,dummies],axis=1)

mapping={ 'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1 }

train_data.HeatingQC.replace(mapping,inplace=True)

test_data.HeatingQC.replace(mapping,inplace=True)
train_data.loc[:,'CentralAir']=le.fit_transform(train_data.CentralAir.values)

test_data.loc[:,'CentralAir']=le.fit_transform(test_data.CentralAir.values)
dummies=pd.get_dummies(train_data.Electrical,prefix='Electrical')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.Electrical,prefix='Electrical')

test_data=pd.concat([test_data,dummies],axis=1)

mapping={ 'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1 }

train_data.KitchenQual.replace(mapping,inplace=True)

test_data.KitchenQual.replace(mapping,inplace=True)
mapping= {'Maj1':2,'Maj2':1,'Min1':5,'Min2':4,'Mod':3,'Sev':0,'Typ':6}

train_data.Functional.replace(mapping,inplace=True)

test_data.Functional.replace(mapping,inplace=True)
mapping={ 'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1 ,'NoFire':0}

train_data.FireplaceQu.replace(mapping,inplace=True)

test_data.FireplaceQu.replace(mapping,inplace=True)
train_data['GarageType']=(train_data.GarageType=='NoGrg')*1

test_data['GarageType']=(test_data.GarageType=='NoGrg')*1

dummies=pd.get_dummies(train_data.GarageType,prefix='GarageType')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.GarageType,prefix='GarageType')

test_data=pd.concat([test_data,dummies],axis=1)

mapping={'Fin':3,'RFn':2,'Unf':1,'NoGrg':0}

train_data.GarageFinish.replace(mapping,inplace=True)

test_data.GarageFinish.replace(mapping,inplace=True)

mapping={ 'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1 ,'NoGrg':0}

train_data.GarageQual.replace(mapping,inplace=True)

test_data.GarageQual.replace(mapping,inplace=True)

train_data.GarageCond.replace(mapping,inplace=True)

test_data.GarageCond.replace(mapping,inplace=True)
mapping={'Y':3,'P':2,'N':1}

train_data.PavedDrive.replace(mapping,inplace=True)

test_data.PavedDrive.replace(mapping,inplace=True)
mapping={'Ex':3,'Gd':2,'Fa':1,'NoPool':0}

train_data.PoolQC.replace(mapping,inplace=True)

test_data.PoolQC.replace(mapping,inplace=True)

mapping={'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'NoFence':0}

train_data.Fence.replace(mapping,inplace=True)

test_data.Fence.replace(mapping,inplace=True)
dummies=pd.get_dummies(train_data.MiscFeature,prefix='MiscFeature')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.MiscFeature,prefix='MiscFeature')

test_data=pd.concat([test_data,dummies],axis=1)
dummies=pd.get_dummies(train_data.SaleType,prefix='SaleType')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.SaleType,prefix='SaleType')

test_data=pd.concat([test_data,dummies],axis=1)

dummies=pd.get_dummies(train_data.SaleCondition,prefix='SaleCondition')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.SaleCondition,prefix='SaleCondition')

test_data=pd.concat([test_data,dummies],axis=1)

dummies=pd.get_dummies(train_data.YrSold,prefix='YrSold')

train_data=pd.concat([train_data,dummies],axis=1)

dummies=pd.get_dummies(test_data.YrSold,prefix='YrSold')

test_data=pd.concat([test_data,dummies],axis=1)
train_data['BsmtQual'].unique


train_data.head()
column_drop=['Id','MSZoning','LotConfig','LotShape','BsmtFinSF1','ExterCond','Fence','PoolQC','GarageCond','PavedDrive','GarageQual','Functional','GarageFinish','KitchenQual','FireplaceQu','HeatingQC','BsmtQual','BsmtCond','BsmtFinType2','BsmtFinType1','BsmtExposure','Foundation','Condition1','Condition2','HouseStyle','RoofStyle','Exterior1st','Exterior2nd','ExterQual','MasVnrType','Foundation','Electrical','GarageType','MiscFeature','YrSold','SaleType','SaleCondition','GarageYrBlt','Neighborhood','BldgType','RoofMatl','Heating','MoSold']

train_data.drop(column_drop,axis=1,inplace=True)

test_data.drop(column_drop,axis=1,inplace=True)

Y=train_data.SalePrice

train_data.drop('SalePrice',axis=1,inplace=True)

plt.hist(Y,normed=True,bins=30)

plt.xlabel('SalePrice')

plt.ylabel('Frequency_count')
from scipy.stats import skew

skew(Y)
#we can see our dependent variable is higly positively skewed

Y=np.log1p(Y)
skew(Y)
plt.hist(Y,normed=True,bins=30)

plt.xlabel('SalePrice')

plt.ylabel('Frequency_count')

train_data.head()

lst=list(range(56))

to_extend=[153,165]

lst.extend(to_extend)

skew_features=train_data.iloc[:,lst].apply(lambda train_data:skew(train_data)).sort_values(ascending=False)

skew_features_test=test_data.iloc[:,lst].apply(lambda test_data:skew(test_data)).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skew_features})

skewness_test = pd.DataFrame({'Skew' :skew_features_test})

skewness.head()



from scipy.special import boxcox1p

skewness = skewness[abs(skewness) > 0.5].dropna()

feats=skewness.index.values.tolist()

lam=0.1

for feat in feats:

    train_data[feat]=boxcox1p(train_data[feat].index.values.tolist(), lam)

    test_data[feat]=boxcox1p(test_data[feat].index.values.tolist(), lam)
train_data.shape,test_data.shape
X=train_data

X_test=test_data

for column in X.columns:

    if column not in X_test.columns:

        X.drop([column], axis=1, inplace=True)
for column in X_test.columns:

    if column not in X.columns:

        X_test.drop([column], axis=1, inplace=True)
train_data.shape,test_data.shape
# Gradient Boosting Regression :-

from sklearn.model_selection import GridSearchCV,KFold,cross_val_score

from sklearn.ensemble import GradientBoostingRegressor

#parameters = {'max_depth':[2,3,4],'n_estimators':[2800,3000,3200],'max_features':['sqrt'],'loss':['huber'],'min_samples_leaf':[14,15,16],'min_samples_split':[9,10,11],'random_state':[0]}

parameters = {'max_depth':[3],'n_estimators':[3000],'max_features':['sqrt'],'loss':['huber'],'min_samples_leaf':[15],'min_samples_split':[10],'random_state':[0]}

model=GridSearchCV(GradientBoostingRegressor(),parameters,scoring='neg_mean_squared_error',cv=KFold(n_splits=7))

model.fit(train_data,Y)

Y_test1=model.predict(test_data)

Y_test1=np.exp(Y_test1)-1

((model.best_score_)*(-1))**0.5
# Lasso Regression :-



from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Lasso

from sklearn.preprocessing import  RobustScaler

model=make_pipeline(RobustScaler(),Lasso(0.0004,random_state=0))

rmslerror=(-cross_val_score(model,train_data,Y,scoring='neg_mean_squared_error',cv=KFold(n_splits=7)))**0.5

print(rmslerror.mean())

model.fit(train_data,Y)

Y_test2=np.expm1(model.predict(test_data))
Y_test=(Y_test1+Y_test2)/2

submission=pd.DataFrame({'Id':range(1461,2920),'SalePrice':Y_test})

submission.to_csv('submit.csv',index=False)