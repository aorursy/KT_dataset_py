pd.options.display.max_rows = 40000
pd.options.display.max_columns = 40000
import pandas as pd
import numpy as np
import seaborn as sn
ds=pd.read_csv('train.csv')

sp=ds[['SalePrice']]
ds=ds.drop(columns=['Id','Alley', 'FireplaceQu','PoolQC','PoolArea','GarageCond','Fence','MiscFeature','Street','Utilities','Condition2','RoofMatl','Heating','LowQualFinSF','SalePrice'],axis=1)
clist=[]
for items in ds.isnull().sum().iteritems():
    if items[1]>0:
        clist.append(items[0])
for j in clist:
    count=0
    for i in ds[j].value_counts().iteritems():
        count=count+1
        if count==1:
            ds[j].fillna(value=i[1],inplace=True)
dxcopy=ds.copy()
dxcopy.shape
dstest=pd.read_csv('test.csv')
dstest=dstest.drop(columns=['Id','Alley', 'FireplaceQu','PoolQC','PoolArea','GarageCond','Fence','MiscFeature','Street','Utilities','Condition2','RoofMatl','Heating','LowQualFinSF'],axis=1)
dlist=[]
for items in dstest.isnull().sum().iteritems():
    if items[1]>0:
        dlist.append(items[0])
for j in dlist:
    count=0
    for i in dstest[j].value_counts().iteritems():
        count=count+1
        if count==1:
            dstest[j].fillna(value=i[1],inplace=True)
dstest.shape
dtestcopy=dstest.copy()
combo=pd.concat([dxcopy,dtestcopy],axis=0)
combo.shape
vo=0
objlist=[]
for col in combo.columns:
    if combo.dtypes[col]=="object":
        vo=vo+1
        objlist.append(col)
print(objlist)
print(vo)
combo=pd.get_dummies(combo,columns = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'PavedDrive', 'SaleType', 'SaleCondition'],drop_first = True)
combo.shape
combo1=combo.copy()
combo1=combo1.T.drop_duplicates().T
combo1.shape
combo1['YearBuilt'] = combo1['YrSold'] - combo1['YearBuilt']
combo1['YearRemodAdd'] = combo1['YrSold'] - combo1['YearRemodAdd']
combo1['GarageYrBlt'] = combo1['YrSold'] - combo1['GarageYrBlt']
combo1=combo1.drop(columns=['YrSold'],axis=1)
combo1.shape
combo2=combo1.copy()
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
combo2=sc_X.fit_transform(combo2)
combo3=pd.DataFrame(combo2, columns=['MSSubClass',
 'LotFrontage',
 'LotArea',
 'OverallQual',
 'OverallCond',
 'YearBuilt',
 'YearRemodAdd',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'GrLivArea',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageYrBlt',
 'GarageCars',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'MiscVal',
 'MoSold',
 'MSZoning_C (all)',
 'MSZoning_FV',
 'MSZoning_RH',
 'MSZoning_RL',
 'MSZoning_RM',
 'LotShape_IR2',
 'LotShape_IR3',
 'LotShape_Reg',
 'LandContour_HLS',
 'LandContour_Low',
 'LandContour_Lvl',
 'LotConfig_CulDSac',
 'LotConfig_FR2',
 'LotConfig_FR3',
 'LotConfig_Inside',
 'LandSlope_Mod',
 'LandSlope_Sev',
 'Neighborhood_Blueste',
 'Neighborhood_BrDale',
 'Neighborhood_BrkSide',
 'Neighborhood_ClearCr',
 'Neighborhood_CollgCr',
 'Neighborhood_Crawfor',
 'Neighborhood_Edwards',
 'Neighborhood_Gilbert',
 'Neighborhood_IDOTRR',
 'Neighborhood_MeadowV',
 'Neighborhood_Mitchel',
 'Neighborhood_NAmes',
 'Neighborhood_NPkVill',
 'Neighborhood_NWAmes',
 'Neighborhood_NoRidge',
 'Neighborhood_NridgHt',
 'Neighborhood_OldTown',
 'Neighborhood_SWISU',
 'Neighborhood_Sawyer',
 'Neighborhood_SawyerW',
 'Neighborhood_Somerst',
 'Neighborhood_StoneBr',
 'Neighborhood_Timber',
 'Neighborhood_Veenker',
 'Condition1_Feedr',
 'Condition1_Norm',
 'Condition1_PosA',
 'Condition1_PosN',
 'Condition1_RRAe',
 'Condition1_RRAn',
 'Condition1_RRNe',
 'Condition1_RRNn',
 'BldgType_2fmCon',
 'BldgType_Duplex',
 'BldgType_Twnhs',
 'BldgType_TwnhsE',
 'HouseStyle_1.5Unf',
 'HouseStyle_1Story',
 'HouseStyle_2.5Fin',
 'HouseStyle_2.5Unf',
 'HouseStyle_2Story',
 'HouseStyle_SFoyer',
 'HouseStyle_SLvl',
 'RoofStyle_Gable',
 'RoofStyle_Gambrel',
 'RoofStyle_Hip',
 'RoofStyle_Mansard',
 'RoofStyle_Shed',
 'Exterior1st_AsbShng',
 'Exterior1st_AsphShn',
 'Exterior1st_BrkComm',
 'Exterior1st_BrkFace',
 'Exterior1st_CBlock',
 'Exterior1st_CemntBd',
 'Exterior1st_HdBoard',
 'Exterior1st_ImStucc',
 'Exterior1st_MetalSd',
 'Exterior1st_Plywood',
 'Exterior1st_Stone',
 'Exterior1st_Stucco',
 'Exterior1st_VinylSd',
 'Exterior1st_Wd Sdng',
 'Exterior1st_WdShing',
 'Exterior2nd_AsbShng',
 'Exterior2nd_AsphShn',
 'Exterior2nd_Brk Cmn',
 'Exterior2nd_BrkFace',
 'Exterior2nd_CBlock',
 'Exterior2nd_CmentBd',
 'Exterior2nd_HdBoard',
 'Exterior2nd_ImStucc',
 'Exterior2nd_MetalSd',
 'Exterior2nd_Other',
 'Exterior2nd_Plywood',
 'Exterior2nd_Stone',
 'Exterior2nd_Stucco',
 'Exterior2nd_VinylSd',
 'Exterior2nd_Wd Sdng',
 'Exterior2nd_Wd Shng',
 'MasVnrType_878',
 'MasVnrType_BrkCmn',
 'MasVnrType_BrkFace',
 'MasVnrType_None',
 'MasVnrType_Stone',
 'ExterQual_Fa',
 'ExterQual_Gd',
 'ExterQual_TA',
 'ExterCond_Fa',
 'ExterCond_Gd',
 'ExterCond_Po',
 'ExterCond_TA',
 'Foundation_CBlock',
 'Foundation_PConc',
 'Foundation_Slab',
 'Foundation_Stone',
 'Foundation_Wood',
 'BsmtQual_649',
 'BsmtQual_Ex',
 'BsmtQual_Fa',
 'BsmtQual_Gd',
 'BsmtQual_TA',
 'BsmtCond_Fa',
 'BsmtCond_Gd',
 'BsmtCond_Po',
 'BsmtCond_TA',
 'BsmtExposure_953',
 'BsmtExposure_Av',
 'BsmtExposure_Gd',
 'BsmtExposure_Mn',
 'BsmtExposure_No',
 'BsmtFinType1_431',
 'BsmtFinType1_ALQ',
 'BsmtFinType1_BLQ',
 'BsmtFinType1_GLQ',
 'BsmtFinType1_LwQ',
 'BsmtFinType1_Rec',
 'BsmtFinType1_Unf',
 'BsmtFinType2_1256',
 'BsmtFinType2_ALQ',
 'BsmtFinType2_BLQ',
 'BsmtFinType2_GLQ',
 'BsmtFinType2_LwQ',
 'BsmtFinType2_Rec',
 'BsmtFinType2_Unf',
 'HeatingQC_Fa',
 'HeatingQC_Gd',
 'HeatingQC_Po',
 'HeatingQC_TA',
 'CentralAir_Y',
 'Electrical_FuseA',
 'Electrical_FuseF',
 'Electrical_FuseP',
 'Electrical_Mix',
 'Electrical_SBrkr',
 'KitchenQual_Ex',
 'KitchenQual_Fa',
 'KitchenQual_Gd',
 'KitchenQual_TA',
 'Functional_Maj1',
 'Functional_Maj2',
 'Functional_Min1',
 'Functional_Min2',
 'Functional_Mod',
 'Functional_Sev',
 'Functional_Typ',
 'GarageType_870',
 'GarageType_2Types',
 'GarageType_Attchd',
 'GarageType_Basment',
 'GarageType_BuiltIn',
 'GarageType_CarPort',
 'GarageType_Detchd',
 'GarageFinish_625',
 'GarageFinish_Fin',
 'GarageFinish_RFn',
 'GarageFinish_Unf',
 'GarageQual_Ex',
 'GarageQual_Fa',
 'GarageQual_Gd',
 'GarageQual_Po',
 'GarageQual_TA',
 'PavedDrive_P',
 'PavedDrive_Y',
 'SaleType_COD',
 'SaleType_CWD',
 'SaleType_Con',
 'SaleType_ConLD',
 'SaleType_ConLI',
 'SaleType_ConLw',
 'SaleType_New',
 'SaleType_Oth',
 'SaleType_WD',
 'SaleCondition_AdjLand',
 'SaleCondition_Alloca',
 'SaleCondition_Family',
 'SaleCondition_Normal',
 'SaleCondition_Partial'])
combo3.shape
Xtrain=combo3.iloc[:1460,:]
Xtest=combo3.iloc[1460:,:]
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
sel_model_train=SelectFromModel(Lasso(alpha=0.005,random_state=0,tol=0.007,max_iter=100000))
sel_model_train.fit(Xtrain,sp)
sel_feat=Xtrain.columns[(sel_model_train.get_support())]
sel_feat
Xtrain=Xtrain[sel_feat]
Xtrain=Xtrain.iloc[:,:].values
ytrain=sp.iloc[:,:].values
ytrain
from xgboost import XGBRegressor
regressor=XGBRegressor(alpha=0.001, base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.3, gamma=0.3,
             importance_type='gain', learning_rate=0.05, max_delta_step=0,
             max_depth=5, min_child_weight=5, missing=None, n_estimators=500,
             n_jobs=1, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=None, subsample=1, verbosity=1)
regressor.fit(Xtrain,ytrain)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=regressor,X=Xtrain,y=ytrain,cv=10)
print(accuracies.mean())
from sklearn.model_selection import RandomizedSearchCV


parameters={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
             "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
             "min_child_weight" : [ 1, 3, 5, 7 ],
             "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
             "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
              "booster"         : ['gblinear','gbtree']}
random_search=RandomizedSearchCV(estimator= regressor,
                        param_distributions=parameters,
                        scoring = 'neg_mean_absolute_error',
                        cv = 10,
                        n_iter=50,
                        n_jobs= 4,
                        verbose=5,
                        return_train_score=True,
                        random_state=42)
best_acc=random_search.fit(Xtrain,ytrain)

random_search.best_estimator_
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=1000, random_state=0)
regressor.fit(Xtrain,ytrain)
Xtest=Xtest[sel_feat]
Xtest=Xtest.iloc[:,:].values
ypred=regressor.predict(Xtest)
nfi=pd.read_csv("sample_submission.csv")
nfi=pd.DataFrame(nfi)
fdf = pd.DataFrame(ypred)
xs=pd.concat([nfi['Id'],fdf],axis=1)
xs.columns=['Id','SalePrice']
xs.to_csv('hypertunexgboostsubmit500.csv',index=False)