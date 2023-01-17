# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import LabelEncoder, RobustScaler# LabelEncoder: Encode target labels with value between 0 and n_classes-1.
from sklearn.model_selection import GridSearchCV,KFold,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from scipy.stats import skew # for a normal distributed data skew is zero, a skewness value greater than zero
                            # means that there is more weight in the right tail of the distribution

from scipy.special import boxcox1p

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
X_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
X = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
X.head()
# Printing top 7 rows
X_test.head(7)
# Printing shape with '.shape'
X.shape,X_test.shape
X.isnull().sum()[X.isnull().sum()>0] # Counting null values and printing them of X_train
X_test.isnull().sum()[X_test.isnull().sum()>0] # Counting null values and printing them of X_test
corr=X.corr() # corr() is used to find the pairwise correlation of all columns in X
plt.figure(figsize=(10,10))
sns.heatmap(X[corr.index[abs(corr['SalePrice'])>0.4]].corr(),annot=True) # for simpler view
plt.show()
# Filling null values.
fillnull={ 'PoolQC':'NoPool', 'MiscFeature':'NoMisc', 'Alley':'NoAlley', 'Fence':'NoFence', 'FireplaceQu':'NoFire' }
X.fillna(fillnull,inplace=True)
X_test.fillna(fillnull,inplace=True)
X.drop(X[X.GrLivArea>4000].index.values,inplace=True)
plt.plot((X.LotArea)**0.5,X.LotFrontage,'.', color='red') # LotArea is square unit and LotFrontage is not so using square-root
plt.xlabel('LotArea', size=13)
plt.ylabel('LotFrontage', size=13)
plt.show()
X.drop(X[X.LotFrontage>300].index.values,inplace=True)
X.drop(X[X.LotArea>100000].index.values,inplace=True)
X.shape
plt.plot((X.LotArea)**0.5,X.LotFrontage,'.')
plt.xlabel('LotArea', size=13)
plt.ylabel('LotFrontage', size=13)
plt.show()
reg=LinearRegression()
reg.fit(((X[(X.LotArea<35000) & (X.LotFrontage<200)].LotArea)**0.5).values.reshape(-1,1),X[(X.LotArea<35000) & (X.LotFrontage<200)].LotFrontage.values)
reg.intercept_,reg.coef_[0] # The coef_ contain the coefficients for the prediction of each of the targets
X.loc[X.LotFrontage.isnull(),'LotFrontage']=reg.predict((X[X.LotFrontage.isnull()].LotArea.values.reshape(-1,1))**0.5)
X_test.loc[X_test.LotFrontage.isnull(),'LotFrontage']=reg.predict((X_test[X_test.LotFrontage.isnull()].LotArea.values.reshape(-1,1))**0.5)
X_test.GarageYrBlt.describe()
X_test[X_test.GarageYrBlt>2018].GarageYrBlt
X_test.GarageYrBlt.replace({2207:2007},inplace=True)
fillnull={ 'GarageType':'NoGarage', 'GarageFinish':'NoGarage', 'GarageQual':'NoGarage', 'GarageCond':'NoGarage', 'BsmtQual':'NoBsmt', 'BsmtCond':'NoBsmt', 'BsmtExposure':'NoBsmt', 'BsmtFinType1':'NoBsmt', 'BsmtFinType2':'NoBsmt' }
X.fillna(fillnull,inplace=True)
X_test.fillna(fillnull,inplace=True)
X.MasVnrType.fillna("None",inplace=True)
X_test.MasVnrType.fillna("None",inplace=True)
X.MasVnrArea.fillna(0,inplace=True)
X_test.MasVnrArea.fillna(0,inplace=True)
X.Electrical.fillna("SBrkr",inplace=True)
fillwithmode=['MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType']
for column in fillwithmode:
    mode=X[column].mode()[0]
    X_test[column].fillna(mode,inplace=True)
fillwithzero=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
for column in fillwithzero:
    X_test[column].fillna(0,inplace=True)
fillwithmedian=['GarageArea','GarageCars']
for column in fillwithmedian:
    median=X[column].median()
    X_test[column].fillna(median,inplace=True)
X_test.isnull().sum()[X_test.isnull().sum()>0]
X.isnull().sum()[X.isnull().sum()>0]
le=LabelEncoder()
dummies=pd.get_dummies(X.MSZoning,prefix='MSZoning')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.MSZoning,prefix='MSZoning')
X_test=pd.concat([X_test,dummies],axis=1)
X.loc[:,'Street']=le.fit_transform(X.Street.values)
X_test.loc[:,'Street']=le.transform(X_test.Street.values)
mapping={'NoAlley':0,'Grvl':1,'Pave':2}
X.Alley.replace(mapping,inplace=True)
X_test.Alley.replace(mapping,inplace=True)
mapping={'IR3':0,'IR2':1,'IR1':2,'Reg':3}
X.LotShape.replace(mapping,inplace=True)
X_test.LotShape.replace(mapping,inplace=True)
mapping={'Low':0,'HLS':1,'Bnk':2,'Lvl':3}
X.LandContour.replace(mapping,inplace=True)
X_test.LandContour.replace(mapping,inplace=True)
X.drop('Utilities',axis=1,inplace=True)
X_test.drop('Utilities',axis=1,inplace=True)
dummies=pd.get_dummies(X.LotConfig,prefix='LotConfig')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.LotConfig,prefix='LotConfig')
X_test=pd.concat([X_test,dummies],axis=1)
X.loc[:,'LandSlope']=le.fit_transform(X.LandSlope.values)
X_test.loc[:,'LandSlope']=le.transform(X_test.LandSlope.values)
dummies=pd.get_dummies(X.Neighborhood,prefix='Neigh')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.Neighborhood,prefix='Neigh')
X_test=pd.concat([X_test,dummies],axis=1)
dummies=pd.get_dummies(X.Condition1,prefix='Cond1')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.Condition1,prefix='Cond1')
X_test=pd.concat([X_test,dummies],axis=1)
dummies=pd.get_dummies(X.Condition2,prefix='Cond2')
X=pd.concat([X,dummies],axis=1)
X['Cond2_RRNe']=0
dummies=pd.get_dummies(X_test.Condition2,prefix='Cond2')
X_test=pd.concat([X_test,dummies],axis=1)
X_test['Cond2_RRAe']=0
X_test['Cond2_RRAn']=0
X_test['Cond2_RRNn']=0
X_test['Cond2_RRNe']=0
conditions=X.Condition1.unique()
for cond in conditions:
    X['Cond_'+cond]=((X['Cond1_'+cond]+X['Cond2_'+cond])>0)*1
    X_test['Cond_'+cond]=((X_test['Cond1_'+cond]+X_test['Cond2_'+cond])>0)*1
    X.drop(['Cond1_'+cond,'Cond2_'+cond],axis=1,inplace=True)
    X_test.drop(['Cond1_'+cond,'Cond2_'+cond],axis=1,inplace=True)
dummies=pd.get_dummies(X.BldgType,prefix='BldgType')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.BldgType,prefix='BldgType')
X_test=pd.concat([X_test,dummies],axis=1)
dummies=pd.get_dummies(X.HouseStyle,prefix='HouseStyle')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.HouseStyle,prefix='HouseStyle')
dummies['HouseStyle_2.5Fin']=0
dummies.sort_index(axis=1,inplace=True)
X_test=pd.concat([X_test,dummies],axis=1)
dummies=pd.get_dummies(X.RoofStyle,prefix='RoofStyle')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.RoofStyle,prefix='RoofStyle')
X_test=pd.concat([X_test,dummies],axis=1)
dummies=pd.get_dummies(X.RoofMatl,prefix='RoofMatl')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.RoofMatl,prefix='RoofMatl')
dummies['RoofMatl_Roll']=0
dummies['RoofMatl_Membran']=0
dummies['RoofMatl_Metal']=0
dummies.sort_index(axis=1,inplace=True)
X_test=pd.concat([X_test,dummies],axis=1)
X.Exterior1st.unique()
X_test.Exterior1st.unique()
X.Exterior2nd.unique()
X_test.Exterior2nd.unique()
mapping={ 'Wd Shng':'WdShing','Brk Cmn':'BrkComm','CmentBd':'CemntBd' }
X.Exterior2nd.replace(mapping,inplace=True)
X_test.Exterior2nd.replace(mapping,inplace=True)
a=X.Exterior1st.value_counts()
b=X.Exterior2nd.value_counts()
c=pd.concat([a,b],axis=1,sort=True)
c.plot.bar(stacked=True)
plt.show()
dummies=pd.get_dummies(X.Exterior1st,prefix='Ext1')
dummies['Ext1_Other']=0
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.Exterior1st,prefix='Ext1')
dummies['Ext1_ImStucc']=0
dummies['Ext1_Stone']=0
dummies['Ext1_Other']=0
X_test=pd.concat([X_test,dummies],axis=1)
dummies=pd.get_dummies(X.Exterior2nd,prefix='Ext2')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.Exterior2nd,prefix='Ext2')
dummies['Ext2_Other']=0
X_test=pd.concat([X_test,dummies],axis=1)
exteriors=X.Exterior2nd.unique()
for ext in exteriors:
    X['Ext_'+ext]=((X['Ext1_'+ext]+X['Ext2_'+ext])>0)*1
    X_test['Ext_'+ext]=((X_test['Ext1_'+ext]+X_test['Ext2_'+ext])>0)*1
    X.drop(['Ext1_'+ext,'Ext2_'+ext],axis=1,inplace=True)
    X_test.drop(['Ext1_'+ext,'Ext2_'+ext],axis=1,inplace=True)
dummies=pd.get_dummies(X.MasVnrType,prefix='MVT')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.MasVnrType,prefix='MVT')
X_test=pd.concat([X_test,dummies],axis=1)
mapping={ 'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0 }
X.ExterQual.replace(mapping,inplace=True)
X_test.ExterQual.replace(mapping,inplace=True)
X.ExterCond.replace(mapping,inplace=True)
X_test.ExterCond.replace(mapping,inplace=True)
dummies=pd.get_dummies(X.Foundation,prefix='Foundation')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.Foundation,prefix='Foundation')
X_test=pd.concat([X_test,dummies],axis=1)
X['NoBsmt']=(X.BsmtQual=='NoBsmt')*1
X_test['NoBsmt']=(X_test.BsmtQual=='NoBsmt')*1
mapping={ 'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0,'NoBsmt':-1 }
X.BsmtQual.replace(mapping,inplace=True)
X_test.BsmtQual.replace(mapping,inplace=True)
X.BsmtCond.replace(mapping,inplace=True)
X_test.BsmtCond.replace(mapping,inplace=True)
mapping={ 'Gd':4,'Av':3,'Mn':2,'No':1,'NoBsmt':0 }
X.BsmtExposure.replace(mapping,inplace=True)
X_test.BsmtExposure.replace(mapping,inplace=True)
X['BsmtFinSF']=X.BsmtFinSF1+X.BsmtFinSF2
X_test['BsmtFinSF']=X_test.BsmtFinSF1+X_test.BsmtFinSF2
mapping={'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBsmt':0}
X.BsmtFinType1.replace(mapping,inplace=True)
X_test.BsmtFinType1.replace(mapping,inplace=True)
X.BsmtFinType2.replace(mapping,inplace=True)
X_test.BsmtFinType2.replace(mapping,inplace=True)
dummies=pd.get_dummies(X.Heating,prefix='Heating')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.Heating,prefix='Heating')
dummies['Heating_OthW']=0
dummies['Heating_Floor']=0
dummies.sort_index(axis=1,inplace=True)
X_test=pd.concat([X_test,dummies],axis=1)
mapping={ 'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0 }
X.HeatingQC.replace(mapping,inplace=True)
X_test.HeatingQC.replace(mapping,inplace=True)
X.loc[:,'CentralAir']=le.fit_transform(X.CentralAir.values)
X_test.loc[:,'CentralAir']=le.transform(X_test.CentralAir.values)
dummies=pd.get_dummies(X.Electrical,prefix='Electrical')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.Electrical,prefix='Electrical')
dummies['Electrical_Mix']=0
dummies.sort_index(axis=1,inplace=True)
X_test=pd.concat([X_test,dummies],axis=1)
X['TotalSF']=X['TotalBsmtSF']+X['1stFlrSF']+X['2ndFlrSF']
X_test['TotalSF']=X_test['TotalBsmtSF']+X_test['1stFlrSF']+X_test['2ndFlrSF']
mapping={ 'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0 }
X.KitchenQual.replace(mapping,inplace=True)
X_test.KitchenQual.replace(mapping,inplace=True)
mapping= {'Maj1':2,'Maj2':1,'Min1':5,'Min2':4,'Mod':3,'Sev':0,'Typ':6}
X.Functional.replace(mapping,inplace=True)
X_test.Functional.replace(mapping,inplace=True)
mapping={ 'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0,'NoFire':-1 }
X.FireplaceQu.replace(mapping,inplace=True)
X_test.FireplaceQu.replace(mapping,inplace=True)
X['NoGarage']=(X.GarageType=='NoGarage')*1
X_test['NoGarage']=(X_test.GarageType=='NoGarage')*1
dummies=pd.get_dummies(X.GarageType,prefix='GarageType')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.GarageType,prefix='GarageType')
X_test=pd.concat([X_test,dummies],axis=1)
mapping={'Fin':3,'RFn':2,'Unf':1,'NoGarage':0}
X.GarageFinish.replace(mapping,inplace=True)
X_test.GarageFinish.replace(mapping,inplace=True)
mapping={ 'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0,'NoGarage':-1 }
X.GarageQual.replace(mapping,inplace=True)
X_test.GarageQual.replace(mapping,inplace=True)
X.GarageCond.replace(mapping,inplace=True)
X_test.GarageCond.replace(mapping,inplace=True)
X.loc[:,'PavedDrive']=le.fit_transform(X.PavedDrive.values)
X_test.loc[:,'PavedDrive']=le.transform(X_test.PavedDrive.values)
mapping={'Ex':3,'Gd':2,'Fa':1,'NoPool':0}
X.PoolQC.replace(mapping,inplace=True)
X_test.PoolQC.replace(mapping,inplace=True)
dummies=pd.get_dummies(X.MiscFeature,prefix='Misc')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.MiscFeature,prefix='Misc')
dummies['Misc_TenC']=0
X_test=pd.concat([X_test,dummies],axis=1)
mapping={'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'NoFence':0}
X.Fence.replace(mapping,inplace=True)
X_test.Fence.replace(mapping,inplace=True)
mapping={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
X.MoSold.replace(mapping,inplace=True)
X_test.MoSold.replace(mapping,inplace=True)
dummies=pd.get_dummies(X.MoSold,prefix='MoSold')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.MoSold,prefix='MoSold')
X_test=pd.concat([X_test,dummies],axis=1)
dummies=pd.get_dummies(X.YrSold,prefix='YrSold')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.YrSold,prefix='YrSold')
X_test=pd.concat([X_test,dummies],axis=1)
dummies=pd.get_dummies(X.SaleType,prefix='SaleType')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.SaleType,prefix='SaleType')
X_test=pd.concat([X_test,dummies],axis=1)
dummies=pd.get_dummies(X.SaleCondition,prefix='SaleCondition')
X=pd.concat([X,dummies],axis=1)
dummies=pd.get_dummies(X_test.SaleCondition,prefix='SaleCondition')
X_test=pd.concat([X_test,dummies],axis=1)
X.head()
X_test.head(6)
column_drop=['Id','MSZoning','LotConfig','Condition1','Condition2','HouseStyle','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','Foundation','Electrical','GarageType','MiscFeature','YrSold','SaleType','SaleCondition','GarageYrBlt','Neighborhood','BldgType','RoofMatl','Heating','MoSold']
X.drop(column_drop,axis=1,inplace=True)
X_test.drop(column_drop,axis=1,inplace=True)
Y=X.SalePrice
X.drop('SalePrice',axis=1,inplace=True)
skew(Y)
Y=np.log1p(Y)
skew(Y)
lst=list(range(56))
to_extend=[153,165]
lst.extend(to_extend)
skew_features=X.iloc[:,lst].apply(lambda x:skew(x)).sort_values(ascending=False)
skew_features_test=X_test.iloc[:,lst].apply(lambda x:skew(x)).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skew_features})
skewness_test = pd.DataFrame({'Skew' :skew_features_test})
skewness.head()
skewness = skewness[abs(skewness) > 0.5].dropna()
feats=skewness.index.values.tolist()
lam=0.1
for feat in feats:
    X[feat]=boxcox1p(X[feat], lam)
    X_test[feat]=boxcox1p(X_test[feat], lam)
X.shape,X_test.shape
X.head()
X_test.head()
#parameters = {'max_depth':[2,3,4],'n_estimators':[2800,3000,3200],'max_features':['sqrt'],'loss':['huber'],'min_samples_leaf':[14,15,16],'min_samples_split':[9,10,11],'random_state':[0]}
parameters = {'max_depth':[3],'n_estimators':[3000],'max_features':['sqrt'],'loss':['huber'],'min_samples_leaf':[15],'min_samples_split':[10],'random_state':[0]}
model=GridSearchCV(GradientBoostingRegressor(),parameters,scoring='neg_mean_squared_error',cv=KFold(n_splits=7))
model.fit(X,Y)
Y_test1=model.predict(X_test)
Y_test1=np.exp(Y_test1)-1
((model.best_score_)*(-1))**0.5
model=make_pipeline(RobustScaler(),Lasso(0.0004,random_state=0))
rmslerror=(-cross_val_score(model,X,Y,scoring='neg_mean_squared_error',cv=KFold(n_splits=7)))**0.5
print(rmslerror.mean())
model.fit(X,Y)
Y_test2=np.expm1(model.predict(X_test))
Y_test=(Y_test1+Y_test2)/2
submission=pd.DataFrame({'Id':range(1461,2920),'SalePrice':Y_test})
submission.to_csv('submit.csv',index=False)