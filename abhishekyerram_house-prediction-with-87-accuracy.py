#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.shape

test.shape
train.describe()
test.describe()
train.info()
test.info()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.isnull().sum().sort_values(ascending=False)[0:20]
test.isnull().sum().sort_values(ascending=False)[0:34]
train.columns.tolist
test.columns.tolist
lst=['PoolQC','MiscFeature','Alley','Fence'] #droping columns which have more null values
train.drop(lst,axis=1,inplace=True)
test.drop(lst,axis=1,inplace=True)
list1 = ['BsmtQual', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'MasVnrType',
         'BsmtExposure','BsmtFinType2']
for col in list1:
    train[col]=train[col].fillna(train[col].mode()[0])
    test[col]=test[col].fillna(test[col].mode()[0])
train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())
test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())
train.drop('GarageYrBlt',axis=1,inplace=True)
test.drop('GarageYrBlt',axis=1,inplace=True)
train['BsmtCond']=train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
test['BsmtCond']=test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])
train['BsmtFinType1']=train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0])
test['BsmtFinType1']=test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])
train['MasVnrArea']=train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])
test['MasVnrArea']=test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
train['Electrical']=train['Electrical'].fillna(train['Electrical'].mode()[0])
train.isnull().sum().sort_values(ascending=False)[0:5] #all train null values have been cleaned
columns = [ 'Utilities','BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 
           'Exterior1st', 'KitchenQual']
columns1 = ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',  'TotalBsmtSF', 'GarageArea']

for item in columns:
    test[item] = test[item].fillna(test[item].mode()[0])
for item in columns1:
    test[item] = test[item].fillna(test[item].mean())
test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test.isnull().sum().sort_values(ascending=False)[0:5] #test data also been cleaned
sns.distplot(train['SalePrice'])
plt.figure(figsize=(40,30))
sns.heatmap(train.corr(),annot=True)
sns.jointplot(train['OverallQual'],train['SalePrice'])
sns.lmplot(x='PoolArea',y='SalePrice',data=train)
sns.lmplot(x='GrLivArea',y='SalePrice',data=train,palette='viridis')
train.select_dtypes(include=object).columns.tolist
test.select_dtypes(include=object).columns.tolist
columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
def One_hot_encoding(columns):
   
    df_final=train
    i=0
    for fields in columns:
        df1=pd.get_dummies(train[fields],drop_first=True)
        
        train.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:           
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([train,df_final],axis=1)
        
    return df_final
train= One_hot_encoding(columns)
train.shape
train =train.loc[:,~train.columns.duplicated()]
train.shape
def One_hot_encoding(columns):
   
    df_final=test
    i=0
    for fields in columns:
        df1=pd.get_dummies(test[fields],drop_first=True)
        
        test.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:           
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([test,df_final],axis=1)
        
    return df_final
test= One_hot_encoding(columns)

test=test.loc[:,~test.columns.duplicated()]
test.shape
train.head(3)
test.head(3)
common=(train.columns)&(test.columns)
lst1=['NoSeWa', '2.5Fin', 'CompShg', 'Membran', 'Metal', 'Roll',
       'Other', 'GasA', 'OthW', 'Mix']
train.drop(lst1,axis=1,inplace=True)
X=train.drop('SalePrice',axis=1)
Y=train['SalePrice']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,Y_train)
predictions=lm.predict(X_test)
from sklearn import metrics
metrics.r2_score(Y_test,predictions)
import xgboost as xgb
my_model = xgb.XGBRegressor(n_estimators=1000,learning_rate = 0.1)
my_model.fit(X_train,Y_train)
predictions1 = my_model.predict(X_test)
from sklearn import metrics
metrics.r2_score(Y_test,predictions1)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=1000)
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [100, 500, 900]
criterion = ['mse', 'mae']
depth = [3,5,10,15]
min_split=[2,3,4]
min_leaf=[2,3,4]
bootstrap = ['True', 'False']
verbose = [5]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':depth,
    'criterion':criterion,
    'bootstrap':bootstrap,
    'verbose':verbose,
    'min_samples_split':min_split,
    'min_samples_leaf':min_leaf
    }

random_cv = RandomizedSearchCV(estimator=regressor,
                               param_distributions=hyperparameter_grid,
                               cv=5, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = 4, 
                               return_train_score = True,
                               random_state=42)
regressor.fit(X_train,Y_train)
y=regressor.predict(X_test)
from sklearn import metrics
metrics.r2_score(Y_test,y)
import lightgbm as lgbm
my = lgbm.LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=12000, 
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.4, 
                                       )
my.fit(X_train,Y_train)
res=my.predict(X_test)
from sklearn import metrics
metrics.r2_score(Y_test,res)
final=my.predict(test)
final
final1=regressor.predict(test)
final1
#choosing best predictions in submission file
df_submission=pd.DataFrame(test['Id'])
df_submission['SalePrice']=final
df_submission.to_csv('Submission_file.csv',index=False)
