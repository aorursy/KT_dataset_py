import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt

import lightgbm as lgb

import xgboost as xgb

import os

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV, KFold,GridSearchCV

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from sklearn.preprocessing import StandardScaler,RobustScaler,LabelEncoder,PowerTransformer

from sklearn.ensemble import GradientBoostingRegressor,StackingRegressor, RandomForestRegressor

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.pipeline import make_pipeline



import pandas_profiling as pp

from sklearn.linear_model import Lasso

from sklearn.linear_model import LassoCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder



print(os.listdir("../input/house-prices-advanced-regression-techniques"))

%matplotlib inline
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

sample=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
test.head()
train.shape
test.shape
sample.head()
train.SalePrice.describe()
import matplotlib.pyplot as plt 

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize']=(10,6)



print("Skewness : ", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()


print("Skewness after log: ", np.log(train.SalePrice).skew())

plt.hist(np.log(train.SalePrice), color='blue')

plt.show()

target= np.log(train.SalePrice)
num_features = train.select_dtypes(include=[np.number])

#check data types of these 

num_features.dtypes
train.OverallQual.unique()
qual_pivot = train.pivot_table(index='OverallQual', 

                               values='SalePrice', 

                               aggfunc=np.mean)

display(qual_pivot)
#create pivot for overall quality 

qual_pivot.plot(kind='bar', color='green')

plt.xlabel('Overall Quality')

plt.ylabel('Mean Sale Price')

plt.xticks(rotation=0)

plt.show()
#create pivot for Gr Living area 

plt.scatter(x=train['GrLivArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Above grade(ground) living area square feet')

plt.show()
plt.scatter(x=train['GarageArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
nuls = pd.DataFrame(train.isnull().sum().sort_values(ascending =False)[:25])

nuls.columns = ['Null Count']

nuls.index.name = 'Feature'

nuls
catgr  = train.select_dtypes(exclude=[np.number])

catgr.describe()
# One more variable, Garage car capacity

train.GarageCars.value_counts().plot(kind='bar', color='green')

plt.xlabel('Garage Car Capacity')

plt.ylabel('Counts')

plt.xticks(rotation=0)

plt.show()


condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='skyblue')

plt.xlabel('Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()

#encoding steps

def encode_condition(x) : 

    return 1 if x =='Partial' else 0

train['enc_condition'] = train.SaleCondition.apply(encode_condition)

test['enc_condition'] = test.SaleCondition.apply(encode_condition)
condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='gray')

plt.xlabel('Encoded Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
sb.set(font_scale=1.1)  # big

correlation_train=train.corr()

plt.figure(figsize=(30,20))

sb.heatmap(correlation_train,annot=True,fmt='.1f',cmap='PiYG')
correlation_train.columns
corr_dict=correlation_train['SalePrice'].sort_values(ascending=False).to_dict()

important_columns=[]

for key,value in corr_dict.items():

    if ((value>0.1) & (value<0.8)) | (value<=-0.1):

        important_columns.append(key)

important_columns
plt.figure(figsize=(30,10))

sb.boxplot(x='YearBuilt', y="SalePrice", data=train)

sb.swarmplot(x='YearBuilt', y="SalePrice", data=train, color=".25")

plt.xticks(weight='bold',rotation=90)
train_test=pd.concat([train,test],axis=0,sort=False)

train_test.head()
pd.set_option('display.max_rows', 5000)

train_test_null_info=pd.DataFrame(train_test.isnull().sum(),columns=['Count of NaN'])

train_test_dtype_info=pd.DataFrame(train_test.dtypes,columns=['DataTypes'])

train_tes_info=pd.concat([train_test_null_info,train_test_dtype_info],axis=1)

train_tes_info
train_test.loc[train_test['Fireplaces']==0,'FireplaceQu']='Nothing'

train_test['LotFrontage'] = train_test['LotFrontage'].fillna(train_test.groupby('1stFlrSF')['LotFrontage'].transform('mean'))

train_test['LotFrontage'].interpolate(method='linear',inplace=True)

train_test['LotFrontage']=train_test['LotFrontage'].astype(int)

train_test['MasVnrArea'] = train_test['MasVnrArea'].fillna(train_test.groupby('MasVnrType')['MasVnrArea'].transform('mean'))

train_test['MasVnrArea'].interpolate(method='linear',inplace=True)

train_test['MasVnrArea']=train_test['MasVnrArea'].astype(int)

train_test["Fence"] = train_test["Fence"].fillna("None")

train_test["FireplaceQu"] = train_test["FireplaceQu"].fillna("None")

train_test["Alley"] = train_test["Alley"].fillna("None")

train_test["PoolQC"] = train_test["PoolQC"].fillna("None")

train_test["MiscFeature"] = train_test["MiscFeature"].fillna("None")

train_test.loc[train_test['BsmtFinSF1']==0,'BsmtFinType1']='Unf'

train_test.loc[train_test['BsmtFinSF2']==0,'BsmtQual']='TA'

train_test['Total_SF'] = train_test['TotalBsmtSF'] + train_test['1stFlrSF'] + train_test['2ndFlrSF']      

train_test['YrBltRmd']=train_test['YearBuilt']+train_test['YearRemodAdd']

train_test['Total_Square_Feet'] = (train_test['BsmtFinSF1'] + train_test['BsmtFinSF2'] + train_test['1stFlrSF'] + train_test['2ndFlrSF'])

train_test['Total_Bath'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) + train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))

train_test['Total_Porch_Area'] = (train_test['OpenPorchSF'] + train_test['3SsnPorch'] + train_test['EnclosedPorch'] + train_test['ScreenPorch'] + train_test['WoodDeckSF'])

train_test['existpool'] = train_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

train_test['existfireplace'] = train_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

train_test['old_house'] = train_test['YearBuilt'].apply(lambda x: 1 if x <1995 else 0)



for i in train_test.columns:

    if 'SalePrice' not in i:

        if 'object' in str(train_test[str(i)].dtype):

            train_test[str(i)]=train_test[str(i)].fillna(method='ffill')
columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'YrSold', 'MoSold',

           'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond')



for col in columns:

    lbl_enc = LabelEncoder() 

    lbl_enc.fit(list(train_test[col].values)) 

    train_test[col] = lbl_enc.transform(list(train_test[col].values))
train_test=pd.get_dummies(train_test)
train_test_null_info=pd.DataFrame(train_test.isnull().sum(),columns=['Count of NaN'])

train_test_dtype_info=pd.DataFrame(train_test.dtypes,columns=['DataTypes'])

train_test_info=pd.concat([train_test_null_info,train_test_dtype_info],axis=1)

train_test_info
train=train_test[0:1460]

test=train_test[1460:2919]
train.interpolate(method='linear',inplace=True)

test.interpolate(method='linear',inplace=True)
corr_new_train=train.corr()

plt.figure(figsize=(5,20))

sb.heatmap(corr_new_train[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(60),vmin=-1, cmap='PiYG', annot=True)
plt.figure(figsize=(25,10))

train.boxplot(column=['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',

                'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'LotArea', 'BsmtFullBath', 'BsmtUnfSF', 'BedroomAbvGr',

                'ScreenPorch', 'EnclosedPorch', 'KitchenAbvGr'])

plt.xticks(weight='bold',rotation=90)
corr_dict2=corr_new_train['SalePrice'].sort_values(ascending=False).to_dict()

corr_dict2
important_columns2=[]

for key,value in corr_dict2.items():

    if ((value>=0.40) & (value<0.9)) | (value<=-0.4):

        important_columns2.append(key)

important_columns2
best_columns=['OverallQual', 'Total_SF', 'GrLivArea', 'Total_Square_Feet', 'GarageCars', 'Total_Bath', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',

              'YrBltRmd', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'Foundation_PConc', 'MasVnrArea', 'existfireplace', 'GarageYrBlt',

              'Fireplaces', 'HeatingQC', 'GarageFinish', 'old_house', 'KitchenQual', 'ExterQual']



colors=[plt.cm.Set3(each) for each in np.linspace(0, 1, len(best_columns))]

for i,c in zip(best_columns,colors):

    plt.style.use('dark_background')

    #plt.figure(figsize=(20,16))

    #plt.scatter(x=train[i], y=train["SalePrice"],c=c,alpha=0.7)

    sb.jointplot(x=i, y="SalePrice", data=train, kind="reg",color=c,height=10)

    #sb.regplot(x=train[i], y=train["SalePrice"], color=c,fit_reg=True,marker='o',scatter_kws={'s':50})

    plt.xlabel(str(i),fontsize=20)

    plt.yticks(np.arange(0,800001,50000))

    plt.ylabel('SalePrice',fontsize=20)
train = train.drop(train[(train.OverallQual==4) & (train.SalePrice>200000)].index)

train = train.drop(train[(train.OverallQual==10) & (train.SalePrice<200000)].index)

train = train.drop(train[(train.Total_SF>=10000) & (train.SalePrice<200000)].index)

train = train.drop(train[(train.GrLivArea>5000) & (train.SalePrice<200000)].index)

train = train.drop(train[(train.GrLivArea<3000) & (train.SalePrice>550000)].index)

train = train.drop(train[(train.Total_Bath>4) & (train.SalePrice<150000)].index)

train = train.drop(train[(train.Total_Bath<5) & (train.SalePrice>550000)].index)

train = train.drop(train[(train.GarageArea>1200) & (train.SalePrice<100000)].index)

train = train.drop(train[(train.GarageArea<1000) & (train.SalePrice>500000)].index)

train = train.drop(train[(train.TotalBsmtSF>6000) & (train.SalePrice<200000)].index)

train = train.drop(train[(train.FullBath<1) & (train.SalePrice>250000)].index)

train = train.drop(train[(train.TotRmsAbvGrd==14) & (train.SalePrice<300000)].index)

train = train.drop(train[(train.TotRmsAbvGrd==10) & (train.SalePrice>700000)].index)

train = train.drop(train[(train.YearBuilt<1950) & (train.SalePrice>350000)].index)

train = train.drop(train[(train.YearRemodAdd<1970) & (train.SalePrice>350000)].index)

train = train.drop(train[(train.Foundation_PConc==0) & (train.SalePrice>450000)].index)

train = train.drop(train[(train.MasVnrArea>=1500) & (train.SalePrice<250000)].index)

train = train.drop(train[(train.MasVnrArea<=750) & (train.SalePrice>550000)].index)

train = train.drop(train[(train.GarageYrBlt>1990) & (train.SalePrice>600000)].index)

train = train.drop(train[(train.GarageYrBlt<1980) & (train.SalePrice>330000)].index)

train = train.drop(train[(train.HeatingQC.isin([2,4])) & (train.SalePrice>350000)].index)

train = train.drop(train[(train.GarageFinish.isin([1,2])) & (train.SalePrice>470000)].index)

train = train.drop(train[(train.old_house==0) & (train.SalePrice<100000)].index)

train = train.drop(train[(train.old_house==1) & (train.SalePrice>450000)].index)

train = train.drop(train[(train.KitchenQual==2) & (train.SalePrice>600000)].index)

train = train.drop(train[(train.KitchenQual==3) & (train.SalePrice>350000)].index)

train = train.drop(train[(train.ExterQual==2) & (train.SalePrice>500000)].index)

train = train.drop(train[(train.BsmtFullBath==0) & (train.SalePrice>650000)].index)



train = train[train.GarageArea * train.GarageCars < 3700]

train = train[(train.FullBath + (train.HalfBath*0.5) + train.BsmtFullBath + (train.BsmtHalfBath*0.5))<5]
corr1_new_train=train.corr()

plt.figure(figsize=(5,20))

sb.heatmap(corr1_new_train[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(60),vmin=-1, cmap='PiYG', annot=True)
train.SalePrice = np.log1p(train.SalePrice)
train.isnull().sum()
test.isnull().sum()
del test['SalePrice']
train.head()
X=train.drop(['SalePrice'],axis=1)

y=train.SalePrice
std_scaler=StandardScaler()

rbst_scaler=RobustScaler()

power_transformer=PowerTransformer()

X_std=std_scaler.fit_transform(X)

X_rbst=rbst_scaler.fit_transform(X)

X_pwr=power_transformer.fit_transform(X)



test_std=std_scaler.fit_transform(test)

test_rbst=rbst_scaler.fit_transform(test)

test_pwr=power_transformer.fit_transform(test)
X_train,X_test,y_train,y_test=train_test_split(X_pwr,y,test_size=0.4)

print('X_train Shape :',X_train.shape)

print('X_test Shape :',X_test.shape)

print('y_train Shape :',y_train.shape)

print('y_test Shape :',y_test.shape)
lgb_regressor=lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.03, n_estimators=1938, max_bin=50, bagging_fraction=0.65,bagging_freq=5, bagging_seed=7, 

                                feature_fraction=0.201, feature_fraction_seed=7,n_jobs=-1)

lgb_regressor.fit(X_pwr, y)

y_head=lgb_regressor.predict(X_test)

print('-'*10+'LGBM'+'-'*10)

print('R square Accuracy: ',r2_score(y_test,y_head))

print('Mean Absolute Error Accuracy: ',mean_absolute_error(y_test,y_head))

print('Mean Squared Error Accuracy: ',mean_squared_error(y_test,y_head))
gb_reg = GradientBoostingRegressor(n_estimators=1792, learning_rate=0.01005, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=14, loss='huber', random_state =42)

gb_reg.fit(X_pwr, y)

y_head=gb_reg.predict(X_test)

print('-'*10+'GBR'+'-'*10)

print('R square Accuracy: ',r2_score(y_test,y_head))

print('Mean Absolute Error Accuracy: ',mean_absolute_error(y_test,y_head))

print('Mean Squared Error Accuracy: ',mean_squared_error(y_test,y_head))
kfolds = KFold(n_splits=8, shuffle=True, random_state=42)



alphas=[1e-9,1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1, 1, 10]



ridgecv_reg= make_pipeline(RidgeCV(alphas=alphas, cv=kfolds))

ridgecv_reg.fit(X_pwr, y)

y_head=ridgecv_reg.predict(X_test)

print('-'*10+'RidgeCV'+'-'*10)

print('R square Accuracy: ',r2_score(y_test,y_head))

print('Mean Absolute Error Accuracy: ',mean_absolute_error(y_test,y_head))

print('Mean Squared Error Accuracy: ',mean_squared_error(y_test,y_head))
kfolds = KFold(n_splits=8, shuffle=True, random_state=42)



lassocv_reg= make_pipeline(LassoCV(alphas=alphas, cv=kfolds))

lassocv_reg.fit(X_pwr, y)

y_head=lassocv_reg.predict(X_test)

print('-'*10+'LassoCV'+'-'*10)

print('R square Accuracy: ',r2_score(y_test,y_head))

print('Mean Absolute Error Accuracy: ',mean_absolute_error(y_test,y_head))

print('Mean Squared Error Accuracy: ',mean_squared_error(y_test,y_head))
kfolds = KFold(n_splits=8, shuffle=True, random_state=42)



alphas=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,0.0008,0.0009,0.001]

l1ratio=[0.8, 0.83, 0.85, 0.87, 0.9,0.92, 0.95,0.97, 0.99, 1]



elasticv_reg= make_pipeline(ElasticNetCV(alphas=alphas, cv=kfolds, l1_ratio=l1ratio))

elasticv_reg.fit(X_pwr, y)

y_head=elasticv_reg.predict(X_test)

print('-'*10+'ElasticNetCV'+'-'*10)

print('R square Accuracy: ',r2_score(y_test,y_head))

print('Mean Absolute Error Accuracy: ',mean_absolute_error(y_test,y_head))

print('Mean Squared Error Accuracy: ',mean_squared_error(y_test,y_head))
estimators = [('lgbm', lgb_regressor),

              ('gbr', gb_reg),   

              ('lasso', lassocv_reg),   

              ('ridge', ridgecv_reg),   

              ('elasticnet', elasticv_reg)]



stack_reg=StackingRegressor(estimators=estimators,final_estimator=ridgecv_reg,n_jobs=-1)

stack_reg.fit(X_pwr, y)

y_head=stack_reg.predict(X_test)

print('-'*10+'StackingRegressor'+'-'*10)

print('R square Accuracy: ',r2_score(y_test,y_head))

print('Mean Absolute Error Accuracy: ',mean_absolute_error(y_test,y_head))

print('Mean Squared Error Accuracy: ',mean_squared_error(y_test,y_head))
y_head=pd.DataFrame(y_head,columns=['Predict'])

y_test.reset_index(drop=True,inplace=True)

y_test_y_head=pd.concat([y_test,y_head],axis=1)

y_test_y_head.head()
test_pred_lgb=lgb_regressor.predict(test_pwr)

test_pred_gb=gb_reg.predict(test_pwr)

test_pred_elastic=elasticv_reg.predict(test_pwr)

test_pred_ridge=ridgecv_reg.predict(test_pwr)

test_pred_lasso=lassocv_reg.predict(test_pwr)

test_pred_stack=stack_reg.predict(test_pwr)
test_pred_lgb=pd.DataFrame(test_pred_lgb,columns=['SalePrice'])

test_pred_gb=pd.DataFrame(test_pred_gb,columns=['SalePrice'])

test_pred_elastic=pd.DataFrame(test_pred_elastic,columns=['SalePrice'])

test_pred_ridge=pd.DataFrame(test_pred_ridge,columns=['SalePrice'])

test_pred_lasso=pd.DataFrame(test_pred_lasso,columns=['SalePrice'])

test_pred_stack=pd.DataFrame(test_pred_stack,columns=['SalePrice'])
test_pred_lgb.SalePrice =np.floor(np.expm1(test_pred_lgb.SalePrice))

test_pred_gb.SalePrice =np.floor(np.expm1(test_pred_gb.SalePrice))

test_pred_elastic.SalePrice =np.floor(np.expm1(test_pred_elastic.SalePrice))

test_pred_ridge.SalePrice =np.floor(np.expm1(test_pred_ridge.SalePrice))

test_pred_lasso.SalePrice =np.floor(np.expm1(test_pred_lasso.SalePrice))

test_pred_stack.SalePrice =np.floor(np.expm1(test_pred_stack.SalePrice))
old_prediction=sample
old_prediction.head()
sample.iloc[:,1]=(0.5 * test_pred_stack.iloc[:,0])+(0.5 * old_prediction.iloc[:,1])
sample.head()
sample.to_csv('Submition.csv',index=False)