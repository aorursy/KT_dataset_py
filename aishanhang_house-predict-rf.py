# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
submission=pd.read_csv('../input/sample_submission.csv')
print(test.iloc[:,0].size)
print(submission.iloc[:,0].size)
print(train.shape)
print(test.shape)
from sklearn.ensemble import RandomForestRegressor
Y_label=train['SalePrice']
#Y_label.head()
#train.describe(include="all")
#train.info()
#print('train columns with null values:\n',train.isnull().sum())
#print('*'*10)
#print('test columns with null values:\n',test.isnull().sum())
train=train.drop('SalePrice',axis=1)
#train.columns
#特征处理
data_cleaner=[train,test]
for dataset in data_cleaner:
    dataset['LotFrontage'].fillna(dataset['LotFrontage'].median(),inplace=True)#将距离用平均数替代
    dataset['Alley'].fillna('None',inplace=True)
    dataset['MasVnrType'].fillna('None',inplace=True)
    dataset['MasVnrArea'].fillna('0',inplace=True)
    dataset['BsmtQual'].fillna('None',inplace=True)
    dataset['BsmtCond'].fillna('None',inplace=True)
    dataset['BsmtExposure'].fillna('None',inplace=True)
    dataset['BsmtFinType1'].fillna('None',inplace=True)
    dataset['BsmtFinType2'].fillna('None',inplace=True)
    dataset['Electrical'].fillna(dataset['Electrical'].mode()[0],inplace=True)
    dataset['FireplaceQu'].fillna('None',inplace=True)
    dataset['GarageType'].fillna('0',inplace=True)
    dataset['GarageYrBlt'].fillna('0',inplace=True)
    dataset['GarageFinish'].fillna('0',inplace=True)
    dataset['GarageQual'].fillna('None',inplace=True)
    dataset['GarageCond'].fillna('None',inplace=True)
    dataset['PoolQC'].fillna('None',inplace=True)
    dataset['Fence'].fillna('None',inplace=True)
    dataset['MiscFeature'].fillna('None',inplace=True)

 
#train.info()
#test.info()
#test['BsmtFinSF1'].fillna(0,inplace=True)
#test['BsmtFinSF2'].fillna(0,inplace=True)
#test['BsmtUnfSF'].fillna(0,inplace=True)
#test['TotalBsmtSF'].fillna(0,inplace=True)

test.fillna(0,inplace=True)
cen_map={'Y':1,'N':0}
for dataset in data_cleaner:
    dataset['CentralAir']=dataset['CentralAir'].map(cen_map)
    dataset['GarageYrBlt']=dataset['GarageYrBlt'].astype(int)
    dataset['MasVnrArea']=dataset['MasVnrArea'].astype(int)
#train['CentralAir'].isnull().sum()
#test['CentralAir'].isnull().sum()
#train['GarageYrBlt']=train['GarageYrBlt'].astype(int)
#train['MasVnrArea']=train['MasVnrArea'].astype(int)
#数据归一化
from sklearn.preprocessing import StandardScaler
dataset=[train,test]
for data in dataset:
    MSSubClass_scale=StandardScaler().fit(data['MSSubClass'].reshape(-1,1))
    data['MSSubClass']=StandardScaler().fit_transform(data['MSSubClass'].reshape(-1,1),MSSubClass_scale)
    LotFrontage_scale=StandardScaler().fit(data['LotFrontage'].reshape(-1,1))
    data['LotFrontage']=StandardScaler().fit_transform(data['LotFrontage'].reshape(-1,1),LotFrontage_scale)
    LotArea_scale=StandardScaler().fit(data['LotArea'].reshape(-1,1))
    data['LotArea']=StandardScaler().fit_transform(data['LotArea'].reshape(-1,1),LotArea_scale)
    
    OverallQual_scale=StandardScaler().fit(data['OverallQual'].reshape(-1,1))
    data['OverallQual']=StandardScaler().fit_transform(data['OverallQual'].reshape(-1,1),OverallQual_scale)
    
    OverallCond_scale=StandardScaler().fit(data['OverallCond'].reshape(-1,1))
    data['OverallCond']=StandardScaler().fit_transform(data['OverallCond'].reshape(-1,1),OverallCond_scale)
    
    YearBuilt_scale=StandardScaler().fit(data['YearBuilt'].reshape(-1,1))
    data['YearBuilt']=StandardScaler().fit_transform(data['YearBuilt'].reshape(-1,1),YearBuilt_scale)
    
    YearRemodAdd_scale=StandardScaler().fit(data['YearRemodAdd'].reshape(-1,1))
    data['YearRemodAdd']=StandardScaler().fit_transform(data['YearRemodAdd'].reshape(-1,1),YearRemodAdd_scale)
    
    MasVnrArea_scale=StandardScaler().fit(data['MasVnrArea'].reshape(-1,1))
    data['MasVnrArea']=StandardScaler().fit_transform(data['MasVnrArea'].reshape(-1,1),MasVnrArea_scale)
    
    BsmtFinSF1_scale=StandardScaler().fit(data['BsmtFinSF1'].reshape(-1,1))
    data['BsmtFinSF1']=StandardScaler().fit_transform(data['BsmtFinSF1'].reshape(-1,1),BsmtFinSF1_scale)
    
    BsmtFinSF2_scale=StandardScaler().fit(data['BsmtFinSF2'].reshape(-1,1))
    data['BsmtFinSF2']=StandardScaler().fit_transform(data['BsmtFinSF2'].reshape(-1,1),BsmtFinSF2_scale)
    
    BsmtUnfSF_scale=StandardScaler().fit(data['BsmtUnfSF'].reshape(-1,1))
    data['BsmtUnfSF']=StandardScaler().fit_transform(data['BsmtUnfSF'].reshape(-1,1),BsmtUnfSF_scale)
    
    TotalBsmtSF_scale = StandardScaler().fit(data['TotalBsmtSF'].reshape(-1,1))
    data['TotalBsmtSF']=StandardScaler().fit_transform(data['TotalBsmtSF'].reshape(-1,1),TotalBsmtSF_scale)
    
    stFlrSF_scale = StandardScaler().fit(data['1stFlrSF'].reshape(-1,1))
    data['1stFlrSF'] = StandardScaler().fit_transform(data['1stFlrSF'].reshape(-1,1),stFlrSF_scale)
    
    ndFlrSF_scale = StandardScaler().fit(data['2ndFlrSF'].reshape(-1,1))
    data['2ndFlrSF'] = StandardScaler().fit_transform(data['2ndFlrSF'].reshape(-1,1),ndFlrSF_scale)
    
    LowQualFinSF_scale=StandardScaler().fit(data['LowQualFinSF'].reshape(-1,1))
    data['LowQualFinSF']=StandardScaler().fit_transform(data['LowQualFinSF'].reshape(-1,1),LowQualFinSF_scale)
    
    GrLivArea_scale=StandardScaler().fit(data['GrLivArea'].reshape(-1,1))
    data['GrLivArea']=StandardScaler().fit_transform(data['GrLivArea'].reshape(-1,1),GrLivArea_scale)
    
    BsmtFullBath_scale=StandardScaler().fit(data['BsmtFullBath'].reshape(-1,1))
    data['BsmtFullBath']=StandardScaler().fit_transform(data['BsmtFullBath'].reshape(-1,1),BsmtFullBath_scale)
    
    FullBath_scale=StandardScaler().fit(data['FullBath'].reshape(-1,1))
    data['FullBath']=StandardScaler().fit_transform(data['FullBath'].reshape(-1,1),FullBath_scale)
    
    HalfBath_scale=StandardScaler().fit(data['HalfBath'].reshape(-1,1))
    data['HalfBath']=StandardScaler().fit_transform(data['HalfBath'].reshape(-1,1),HalfBath_scale)
    
    BedroomAbvGr_scale=StandardScaler().fit(data['BedroomAbvGr'].reshape(-1,1))
    data['BedroomAbvGr']=StandardScaler().fit_transform(data['BedroomAbvGr'].reshape(-1,1),BedroomAbvGr_scale)
    
    KitchenAbvGr_scale=StandardScaler().fit(data['KitchenAbvGr'].reshape(-1,1))
    data['KitchenAbvGr']=StandardScaler().fit_transform(data['KitchenAbvGr'].reshape(-1,1),KitchenAbvGr_scale)
    
    TotRmsAbvGrd_scale=StandardScaler().fit(data['TotRmsAbvGrd'].reshape(-1,1))
    data['TotRmsAbvGrd']=StandardScaler().fit_transform(data['TotRmsAbvGrd'].reshape(-1,1),TotRmsAbvGrd_scale)
    
    GarageYrBlt_scale=StandardScaler().fit(data['GarageYrBlt'].reshape(-1,1))
    data['GarageYrBlt']=StandardScaler().fit_transform(data['GarageYrBlt'].reshape(-1,1),GarageYrBlt_scale)
    
    GarageCars_scale=StandardScaler().fit(data['GarageCars'].reshape(-1,1))
    data['GarageCars']=StandardScaler().fit_transform(data['GarageCars'].reshape(-1,1),GarageCars_scale)
    
    GarageYrBlt_scale=StandardScaler().fit(data['GarageYrBlt'].reshape(-1,1))
    data['GarageYrBlt']=StandardScaler().fit_transform(data['GarageYrBlt'].reshape(-1,1),GarageYrBlt_scale)
    
    GarageArea_scale=StandardScaler().fit(data['GarageArea'].reshape(-1,1))
    data['GarageArea']=StandardScaler().fit_transform(data['GarageArea'].reshape(-1,1),GarageArea_scale)
    
    WoodDeckSF_scale=StandardScaler().fit(data['WoodDeckSF'].reshape(-1,1))
    data['WoodDeckSF']=StandardScaler().fit_transform(data['WoodDeckSF'].reshape(-1,1),WoodDeckSF_scale)
    
    
    OpenPorchSF_scale=StandardScaler().fit(data['OpenPorchSF'].reshape(-1,1))
    data['OpenPorchSF']=StandardScaler().fit_transform(data['OpenPorchSF'].reshape(-1,1),OpenPorchSF_scale)
    EnclosedPorch_scale=StandardScaler().fit(data['EnclosedPorch'].reshape(-1,1))
    data['EnclosedPorch']=StandardScaler().fit_transform(data['EnclosedPorch'].reshape(-1,1),EnclosedPorch_scale)
    
    SsnPorch_scale=StandardScaler().fit(data['3SsnPorch'].reshape(-1,1))
    data['3SsnPorch']=StandardScaler().fit_transform(data['3SsnPorch'].reshape(-1,1),SsnPorch_scale)
    
    ScreenPorch_scale=StandardScaler().fit(data['ScreenPorch'].reshape(-1,1))
    data['ScreenPorch']=StandardScaler().fit_transform(data['ScreenPorch'].reshape(-1,1),ScreenPorch_scale)
    
    PoolArea_scale=StandardScaler().fit(data['PoolArea'].reshape(-1,1))
    data['PoolArea']=StandardScaler().fit_transform(data['PoolArea'].reshape(-1,1),PoolArea_scale)
    
    MiscVal_scale=StandardScaler().fit(data['MiscVal'].reshape(-1,1))
    data['MiscVal']=StandardScaler().fit_transform(data['MiscVal'].reshape(-1,1),MiscVal_scale)
    
    MoSold_scale=StandardScaler().fit(data['MoSold'].reshape(-1,1))
    data['MoSold']=StandardScaler().fit_transform(data['MoSold'].reshape(-1,1),MoSold_scale)
    
    YrSold_scale=StandardScaler().fit(data['YrSold'].reshape(-1,1))
    data['YrSold']=StandardScaler().fit_transform(data['YrSold'].reshape(-1,1),YrSold_scale)
'''train_cov=pd.get_dummies(train[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd',
                                'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
                                'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                                'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',
                                'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold',
                                'MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',
                                'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                                'Exterior1st','Exterior2nd','MasVnrType','ExterQual','Foundation','BsmtQual','BsmtExposure',
                                'BsmtFinType1','BsmtFinType2','Heating','HeatingQC','Electrical','KitchenQual','Functional',
                                'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence',
                                'MiscFeature','SaleType','SaleCondition','PavedDrive']])'''

'''test_cov=pd.get_dummies(train[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd',
                                'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
                                'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                                'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',
                                'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold',
                                'MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',
                                'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                                'Exterior1st','Exterior2nd','MasVnrType','ExterQual','Foundation','BsmtQual','BsmtExposure',
                                'BsmtFinType1','BsmtFinType2','Heating','HeatingQC','Electrical','KitchenQual','Functional',
                                'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence',
                                'MiscFeature','SaleType','SaleCondition','PavedDrive']])'''
'''train_cov['CentralAir']=train['CentralAir']
test_cov['CentralAir']=train['CentralAir']'''
from sklearn import model_selection
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
#建立模型

'''train_x,test_x,train_y,test_y=model_selection.train_test_split(train_cov,Y_label,random_state=0)
print("train_x shape",train_x.shape)
print("test_x shape",test_x.shape)'''
#train_x.isnull().sum()
#首先对n_estimators进行网格搜索
'''param_test1={'n_estimators':[_ for _ in range(100,300,20)]}
gsearch1=GridSearchCV(estimator=RandomForestRegressor(random_state=10,min_samples_split=4,max_depth=9,min_samples_leaf=1,max_features='sqrt'),
                     param_grid=param_test1,cv=3)
gsearch1.fit(train_x,train_y)
print(gsearch1.grid_scores_)
print('*'*10)
print(gsearch1.best_params_)
print('*'*10)
print(gsearch1.best_score_)'''
#{'n_estimators': 280}
#0.8398942093950141
#上面我们得到最佳的若学习迭代次数，接着我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索
'''param_test2={'max_depth':[_ for _ in range(15,20,1)],'min_samples_split':[_ for _ in range(2,4,1)]}
gsearch2=GridSearchCV(estimator=RandomForestRegressor(n_estimators=280,oob_score=True,max_features='sqrt',min_samples_leaf=1,
                                                    random_state=10),param_grid=param_test2,iid=False,cv=5,n_jobs=-1)
gsearch2.fit(train_x,train_y)
print(gsearch2.grid_scores_)
print('*'*10)
print(gsearch2.best_params_)
print('*'*10)
print(gsearch2.best_score_)'''
#{'max_depth': 17, 'min_samples_split': 2}
#0.8551798499687674
#对min_samples_split和min_samples_leaf进行搜索
'''param_test3={'min_samples_split':[_ for _ in range(2,5,1)],'min_samples_leaf':[_ for _ in range(1,5,1)]}
gsearch3=GridSearchCV(estimator=RandomForestRegressor(n_estimators=280,max_depth=17,max_features='sqrt',oob_score=True,
                                                      random_state=10),param_grid=param_test3,iid=False,cv=5,n_jobs=-1)
gsearch3.fit(train_x,train_y)
print(gsearch3.grid_scores_)
print('*'*10)
print(gsearch3.best_params_)
print('*'*10)
print(gsearch3.best_score_)'''
#{'min_samples_leaf': 1, 'min_samples_split': 2}
#0.8551798499687674
#train_x.columns.values
#对max_features 进行搜索
'''param_test4={'max_features':[_ for _ in range(100,114,2)]}
gsearch4=GridSearchCV(estimator=RandomForestRegressor(n_estimators=280,max_depth=17,min_samples_split=2,min_samples_leaf=1,
                                        oob_score=True,random_state=10),param_grid=param_test4,iid=False,cv=5,n_jobs=-1)
gsearch4.fit(train_x,train_y)
print(gsearch4.grid_scores_)
print('*'*10)
print(gsearch4.best_params_)
print('*'*10)
print(gsearch4.best_score_)'''
#{'max_features': 110}
#0.8683955578058189
'''rf=RandomForestRegressor(n_estimators=320,max_depth=17,min_samples_split=2,min_samples_leaf=1,max_features=110,
                                        oob_score=True,random_state=10)
rf.fit(train_x,train_y)
y_predict=rf.predict(test_x)
print(rf.oob_score_,np.sqrt(metrics.mean_squared_error(test_y, y_predict)))#输出袋外准确率，泛化能力体现，输出均方误差'''
'''param_test1={'n_estimators':[_ for _ in range(280,340,10)]}
gsearch1=GridSearchCV(estimator=RandomForestRegressor(max_depth=17,min_samples_split=2,min_samples_leaf=1,max_features=110,
                                        oob_score=True,random_state=10),
                     param_grid=param_test1,cv=3,n_jobs=-1)
gsearch1.fit(train_x,train_y)
print(gsearch1.grid_scores_)
print('*'*10)
print(gsearch1.best_params_)
print('*'*10)
print(gsearch1.best_score_)'''
'''rf2=RandomForestRegressor(n_estimators=320,max_depth=17,min_samples_split=2,min_samples_leaf=1,max_features=110,
                                        oob_score=True,random_state=10)
rf2.fit(train_x,train_y)'''
#y_predict=rf2.predict(test_x)
#print(rf2.oob_score_,np.sqrt(metrics.mean_squared_error(test_y, y_predict)))#输出袋外准确率，泛化能力体现，输出均方误差
'''print(sorted(zip(map(lambda x: round(x, 4), rf2.feature_importances_),train_x.columns), 
             reverse=True)) '''
#由Lasso得到特征进行筛选
train_cov2=pd.get_dummies(train[['RoofMatl','Neighborhood','KitchenQual','ExterQual','BsmtExposure','Functional','MSZoning',
                                'BsmtQual','Condition1','Exterior1st','SaleType','LotConfig','OverallQual','Condition2','LotShape',
                                 'OverallCond','BsmtFinType1','GarageCars','Fireplaces','FullBath','TotRmsAbvGrd','Electrical','HeatingQC','GarageType',
                                 'SaleCondition','GarageCond','BsmtFullBath','LandContour','LandSlope','HouseStyle','MasVnrType','GarageQual','FireplaceQu',
                                 'Exterior2nd','Fence','YearBuilt','HalfBath','LotFrontage','BldgType','YearRemodAdd','GarageFinish','YrSold','2ndFlrSF',
                                 'PoolArea','1stFlrSF','BsmtFinType2','MasVnrArea','BsmtFinSF1','ScreenPorch','LowQualFinSF','BsmtFinSF2','WoodDeckSF',
                                 '3SsnPorch' ,'GarageArea','OpenPorchSF','BsmtUnfSF','TotalBsmtSF','GrLivArea','EnclosedPorch','KitchenAbvGr','Street',
                                 'BedroomAbvGr','Foundation','MoSold','MSSubClass','GarageYrBlt']])
test_cov2=pd.get_dummies(test[['RoofMatl','Neighborhood','KitchenQual','ExterQual','BsmtExposure','Functional','MSZoning',
                                'BsmtQual','Condition1','Exterior1st','SaleType','LotConfig','OverallQual','Condition2','LotShape',
                                 'OverallCond','BsmtFinType1','GarageCars','Fireplaces','FullBath','TotRmsAbvGrd','Electrical','HeatingQC','GarageType',
                                 'SaleCondition','GarageCond','BsmtFullBath','LandContour','LandSlope','HouseStyle','MasVnrType','GarageQual','FireplaceQu',
                                 'Exterior2nd','Fence','YearBuilt','HalfBath','LotFrontage','BldgType','YearRemodAdd','GarageFinish','YrSold','2ndFlrSF',
                                 'PoolArea','1stFlrSF','BsmtFinType2','MasVnrArea','BsmtFinSF1','ScreenPorch','LowQualFinSF','BsmtFinSF2','WoodDeckSF',
                                 '3SsnPorch' ,'GarageArea','OpenPorchSF','BsmtUnfSF','TotalBsmtSF','GrLivArea','EnclosedPorch','KitchenAbvGr','Street',
                                 'BedroomAbvGr','Foundation','MoSold','MSSubClass','GarageYrBlt']])
#进行特征选择 去掉feature_importances_为0的特征
'''train_cov2=pd.get_dummies(train[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd',
                                'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
                                'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                                'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',
                                'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold',
                                'MSZoning','LotShape','LandContour','LotConfig','LandSlope',
                                'BldgType','HouseStyle','RoofStyle',
                                'Exterior1st','Exterior2nd','ExterQual','BsmtQual','BsmtExposure',
                                'BsmtFinType1','HeatingQC','KitchenQual',
                                'FireplaceQu','GarageType','GarageFinish',
                                'PavedDrive']])
test_cov2=pd.get_dummies(test[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd',
                                'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
                                'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                                'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',
                                'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold',
                                'MSZoning','LotShape','LandContour','LotConfig','LandSlope',
                                'BldgType','HouseStyle','RoofStyle',
                                'Exterior1st','Exterior2nd','ExterQual','BsmtQual','BsmtExposure',
                                'BsmtFinType1','HeatingQC','KitchenQual',
                                'FireplaceQu','GarageType','GarageFinish',
                                'PavedDrive']])'''
train_cov2['CentralAir']=train['CentralAir']
test_cov2['CentralAir']=test['CentralAir']
print(test_cov2.shape)
print(train_cov2.shape)
train_x,test_x,train_y,test_y=model_selection.train_test_split(train_cov2,Y_label,random_state=0)
print("train_x shape",train_x.shape)
print("test_x shape",test_x.shape)
#首先对n_estimators进行网格搜索
'''param_test1={'n_estimators':[_ for _ in range(250,350,10)]}
gsearch1=GridSearchCV(estimator=RandomForestRegressor(random_state=10,min_samples_split=4,max_depth=9,min_samples_leaf=1,max_features='sqrt'),
                     param_grid=param_test1,cv=3,n_jobs=-1)
gsearch1.fit(train_x,train_y)
print(gsearch1.grid_scores_)
print('*'*10)
print(gsearch1.best_params_)
print('*'*10)
print(gsearch1.best_score_)
#{'n_estimators': 260}
#0.8476365944194006'''
#上面我们得到最佳的若学习迭代次数，接着我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索
'''param_test2={'max_depth':[_ for _ in range(15,20,1)],'min_samples_split':[_ for _ in range(2,6,1)]}
gsearch2=GridSearchCV(estimator=RandomForestRegressor(n_estimators=340,oob_score=True,max_features='sqrt',min_samples_leaf=1,
                                                    random_state=10),param_grid=param_test2,iid=False,cv=5,n_jobs=-1)
gsearch2.fit(train_x,train_y)
print(gsearch2.grid_scores_)
print('*'*10)
print(gsearch2.best_params_)
print('*'*10)
print(gsearch2.best_score_)'''
#对min_samples_split和min_samples_leaf进行搜索
'''param_test3={'min_samples_split':[_ for _ in range(2,5,1)],'min_samples_leaf':[_ for _ in range(1,5,1)]}
gsearch3=GridSearchCV(estimator=RandomForestRegressor(n_estimators=340,max_depth=15,max_features='sqrt',oob_score=True,
                                                      random_state=10),param_grid=param_test3,iid=False,cv=5,n_jobs=-1)
gsearch3.fit(train_x,train_y)
print(gsearch3.grid_scores_)
print('*'*10)
print(gsearch3.best_params_)
print('*'*10)
print(gsearch3.best_score_)'''
#对max_features 进行搜索
'''param_test4={'max_features':[_ for _ in range(46,60,2)]}
gsearch4=GridSearchCV(estimator=RandomForestRegressor(n_estimators=340,max_depth=15,min_samples_split=2,min_samples_leaf=1,
                                        oob_score=True,random_state=10),param_grid=param_test4,iid=False,cv=5,n_jobs=-1)
gsearch4.fit(train_x,train_y)
print(gsearch4.grid_scores_)
print('*'*10)
print(gsearch4.best_params_)
print('*'*10)
print(gsearch4.best_score_)'''
'''for i in range(100,200,10):
    rf2=RandomForestRegressor(n_estimators=i,max_depth=15,min_samples_split=2,min_samples_leaf=1,max_features=46,
                                        oob_score=True,random_state=10)
    rf2.fit(train_cov2,Y_label)
    #print(rf2.oob_score_)
    y_predict=rf2.predict(test_x)
    print(rf2.oob_score_,np.sqrt(metrics.mean_squared_error(test_y, y_predict)),i)#输出袋外准确率，泛化能力体现，输出均方误差
    #630'''
rf2=RandomForestRegressor(n_estimators=630,max_depth=15,min_samples_split=2,min_samples_leaf=1,max_features=46,
                                        oob_score=True,random_state=10)
rf2.fit(train_cov2,Y_label)
print(rf2.oob_score_)
#y_predict=rf2.predict(test_x)
#print(rf2.oob_score_,np.sqrt(metrics.mean_squared_error(test_y, y_predict)))#输出袋外准确率，泛化能力体现，输出均方误差
test_cov2=test_cov2.reindex(columns=train_cov2.columns)
test_cov2.fillna(0,inplace=True)
train_cov2.shape
#out.columns.values
submission['SalePrice']=rf2.predict(test_cov2)
submission.head()
#print(sorted(zip(map(lambda x: round(x, 4), rf2.feature_importances_),train_x.columns), 
 #            reverse=True)) 
submission.to_csv("../working/submit.csv", index=False)