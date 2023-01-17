# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# pandas
import pandas as pd
pd.DataFrame.fillna 
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

from IPython.display import display

# remove warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv',index_col='Id')
test  = pd.read_csv('../input/test.csv',index_col='Id')
print(train.shape)
display(train.head(1))

print(test.shape)
display(test.head(1))
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
train.head(5)
train.SalePrice.describe()
print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()
target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes
corr = numeric_features.corr()

print (corr['SalePrice'].sort_values(ascending=False)[1:11], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-10:])
# How many unique features are there?

train.OverallQual.unique()
def pivotandplot(data,variable,onVariable,aggfunc):
    pivot_var = data.pivot_table(index=variable,
                                  values=onVariable, aggfunc=aggfunc)
    pivot_var.plot(kind='bar', color='blue')
    plt.xlabel(variable)
    plt.ylabel(onVariable)
    plt.xticks(rotation=0)
    plt.show()
    
pivotandplot(train,'OverallQual','SalePrice',np.median)
# It is a continous variable and hence lets look at the relationship of GrLivArea with SalePrice using a Regression plot

_ = sns.regplot(train['GrLivArea'], train['SalePrice'])
train=train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
_ = sns.regplot(train['GrLivArea'], train['SalePrice'])
_ = sns.regplot(train['GarageArea'], train['SalePrice'])
train = train[train['GarageArea'] < 1200]
_ = sns.regplot(train['GarageArea'], train['SalePrice'])
# Let us first create a DF for log transformation of SalePrice
train['log_SalePrice']=np.log(train['SalePrice']+1)
saleprices=train[['SalePrice','log_SalePrice']]

saleprices.head(5)

train=train.drop(columns=['SalePrice','log_SalePrice'])
print(train.shape)
print(test.shape)
all_data = pd.concat((train, test))
print(all_data.shape)
all_data.head(5)
null_data = pd.DataFrame(all_data.isnull().sum().sort_values(ascending=False))[:50]

null_data.columns = ['Null Count']
null_data.index.name = 'Feature'
null_data
(null_data/len(all_data)) * 100
print ("Unique values are:", train.MiscFeature.unique())
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    all_data[col] = all_data[col].fillna(0)
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
_=sns.regplot(train['LotFrontage'],saleprices['SalePrice'])
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(14,10)
_ = sns.regplot(train['TotalBsmtSF'], saleprices['SalePrice'], ax=ax1)
_ =sns.regplot(train['1stFlrSF'], saleprices['SalePrice'], ax=ax2)
_ = sns.regplot(train['2ndFlrSF'], saleprices['SalePrice'], ax=ax3)
_ = sns.regplot(train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF'], saleprices['SalePrice'], ax=ax4)
#Impute the entire data set
all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

#Let's add two new variables for No nd floor and no basement
all_data['No2ndFlr']=(all_data['2ndFlrSF']==0)
all_data['NoBsmt']=(all_data['TotalBsmtSF']==0)
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(14,10)
_ = sns.barplot(train['BsmtFullBath'], saleprices['SalePrice'], ax=ax1)
_ = sns.barplot(train['FullBath'], saleprices['SalePrice'], ax=ax2)
_ = sns.barplot(train['BsmtHalfBath'], saleprices['SalePrice'], ax=ax3)
_ = sns.barplot(train['BsmtFullBath'] + train['FullBath'] + train['BsmtHalfBath'] + train['HalfBath'], saleprices['SalePrice'], ax=ax4)
all_data['TotalBath']=all_data['BsmtFullBath'] + all_data['FullBath'] + all_data['BsmtHalfBath'] + all_data['HalfBath']
figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18,8)
_ = sns.regplot(train['YearBuilt'], saleprices['SalePrice'], ax=ax1)
_ = sns.regplot(train['YearRemodAdd'], saleprices['SalePrice'], ax=ax2)
_ = sns.regplot((train['YearBuilt']+train['YearRemodAdd'])/2, saleprices['SalePrice'], ax=ax3)
all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']

# Deleting dominating features over 97%
all_data=all_data.drop(columns=['Street','Utilities','Condition2','RoofMatl','Heating'])
# treat some numeric values as str which is actually a categorical data
all_data['MSSubClass']=all_data['MSSubClass'].astype(str)
all_data['MoSold']=all_data['MoSold'].astype(str)
all_data['YrSold']=all_data['YrSold'].astype(str)
# I found these features might look better without 0 data. (just like the column '2ndFlrSF' above.)
all_data['NoLowQual']=(all_data['LowQualFinSF']==0)
all_data['NoOpenPorch']=(all_data['OpenPorchSF']==0)
all_data['NoWoodDeck']=(all_data['WoodDeckSF']==0)
all_data['NoGarage']=(all_data['GarageArea']==0)
all_data=all_data.drop(columns=['PoolArea','PoolQC']) # most of the houses has no pools. 
all_data=all_data.drop(columns=['MiscVal','MiscFeature']) # most of the houses has no misc feature
Basement = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']
Bsmt=all_data[Basement]
Bsmt.head()
Bsmt['BsmtCond'].unique()
Bsmt=Bsmt.replace(to_replace='Po', value=1)
Bsmt=Bsmt.replace(to_replace='Fa', value=2)
Bsmt=Bsmt.replace(to_replace='TA', value=3)
Bsmt=Bsmt.replace(to_replace='Gd', value=4)
Bsmt=Bsmt.replace(to_replace='Ex', value=5)
Bsmt=Bsmt.replace(to_replace='None', value=0)

Bsmt=Bsmt.replace(to_replace='No', value=1)
Bsmt=Bsmt.replace(to_replace='Mn', value=2)
Bsmt=Bsmt.replace(to_replace='Av', value=3)
Bsmt=Bsmt.replace(to_replace='Gd', value=4)

Bsmt=Bsmt.replace(to_replace='Unf', value=1)
Bsmt=Bsmt.replace(to_replace='LwQ', value=2)
Bsmt=Bsmt.replace(to_replace='Rec', value=3)
Bsmt=Bsmt.replace(to_replace='BLQ', value=4)
Bsmt=Bsmt.replace(to_replace='ALQ', value=5)
Bsmt=Bsmt.replace(to_replace='GLQ', value=6)

# Replacing Categorical values to numbers(just like a score!) so we can do a math with them
Bsmt.head()
Bsmt['BsmtScore']= Bsmt['BsmtQual']  * Bsmt['BsmtCond'] * Bsmt['TotalBsmtSF']
all_data['BsmtScore']=Bsmt['BsmtScore']
Bsmt['BsmtFin'] = (Bsmt['BsmtFinSF1'] * Bsmt['BsmtFinType1']) + (Bsmt['BsmtFinSF2'] * Bsmt['BsmtFinType2'])
all_data['BsmtFinScore']=Bsmt['BsmtFin']
all_data['BsmtDNF']=(all_data['BsmtFinScore']==0)
lot=['LotFrontage', 'LotArea','LotConfig','LotShape']
Lot=all_data[lot]
Lot.head()
garage=['GarageArea','GarageCars','GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt']
Garage=all_data[garage]
Garage=Garage.replace(to_replace='Po', value=1)
Garage=Garage.replace(to_replace='Fa', value=2)
Garage=Garage.replace(to_replace='TA', value=3)
Garage=Garage.replace(to_replace='Gd', value=4)
Garage=Garage.replace(to_replace='Ex', value=5)
Garage=Garage.replace(to_replace='None', value=0)

Garage=Garage.replace(to_replace='Unf', value=1)
Garage=Garage.replace(to_replace='RFn', value=2)
Garage=Garage.replace(to_replace='Fin', value=3)

Garage=Garage.replace(to_replace='CarPort', value=1)
Garage=Garage.replace(to_replace='Basment', value=4)
Garage=Garage.replace(to_replace='Detchd', value=2)
Garage=Garage.replace(to_replace='2Types', value=3)
Garage=Garage.replace(to_replace='Basement', value=5)
Garage=Garage.replace(to_replace='Attchd', value=6)
Garage=Garage.replace(to_replace='BuiltIn', value=7)
Garage.head()
Garage['GarageScore']=(Garage['GarageArea']) * (Garage['GarageCars']) * (Garage['GarageFinish'])*(Garage['GarageQual']) *(Garage['GarageType'])
all_data['GarageScore']=Garage['GarageScore']
all_data.head()
non_numeric=all_data.select_dtypes(exclude=[np.number, bool])
non_numeric.head()
def onehot(col_list):
    global all_data
    while len(col_list) !=0:
        col=col_list.pop(0)
        data_encoded=pd.get_dummies(all_data[col], prefix=col)
        all_data=pd.merge(all_data, data_encoded, on='Id')
        all_data=all_data.drop(columns=col)
    print(all_data.shape)
onehot(list(non_numeric))
def log_transform(col_list):
    transformed_col=[]
    while len(col_list)!=0:
        col=col_list.pop(0)
        if all_data[col].skew() > 0.5:
            all_data[col]=np.log(all_data[col]+1)
            transformed_col.append(col)
        else:
            pass
    print(f"{len(transformed_col)} features had been tranformed")
    print(all_data.shape)
numeric=all_data.select_dtypes(include=np.number)
log_transform(list(numeric))
print(train.shape)
print(test.shape)

train=all_data[:len(train)]
test=all_data[len(train):]

# re-Set the train & test data for ML
print(train.shape)
print(test.shape)

# OK. I'm ready
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from sklearn import linear_model, model_selection, ensemble, preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
def rmse(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score
rmse_score = make_scorer(rmse)
rmse_score
feature_names=list(all_data)
Xtrain=train[feature_names]
Xtest=test[feature_names]
Ytrain=saleprices['log_SalePrice']
def score(model):
    score = cross_val_score(model, Xtrain, Ytrain, cv=5, scoring=rmse_score).mean()
    return score
model_Lasso= make_pipeline(RobustScaler(), Lasso(alpha =0.000327, random_state=18))

model_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.00052, l1_ratio=0.70654, random_state=18))

model_KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.029963, kernel='polynomial', degree=1.103746, coef0=5.442672))

model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =18)

model_XGB=xgb.XGBRegressor(n_jobs=-1, n_estimators=849, learning_rate=0.015876, 
                           max_depth=58, colsample_bytree=0.599653, colsample_bylevel=0.287441, subsample=0.154134, seed=18)
scores={}
scores.update({'Lasso':score(model_Lasso)})
scores.update({"ElasticNet":score(model_ENet)})
scores.update({"KNN":score(model_KRR)})
scores.update({"XGB":score(model_XGB)})
scores.update({"XGBoost":score(model_GBoost)})
model_Lasso.fit(Xtrain, Ytrain)
Lasso_Predictions=np.exp(model_Lasso.predict(Xtest))-1

model_ENet.fit(Xtrain, Ytrain)
ENet_Predictions=np.exp(model_ENet.predict(Xtest))-1

model_XGB.fit(Xtrain, Ytrain)
XGB_Predictions=np.exp(model_XGB.predict(Xtest))-1

model_GBoost.fit(Xtrain, Ytrain)
GBoost_Predictions=np.exp(model_GBoost.predict(Xtest))-1

model_KRR.fit(Xtrain, Ytrain)
KRR_Predictions=np.exp(model_KRR.predict(Xtest))-1
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(Xtrain, Ytrain)
score(forest_reg)
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [5,10, 30,50,70,74], 'max_features': [2, 4, 6, 8,12,15,18]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [5,10, 30,50,70], 'max_features': [2, 4, 6, 8,12,15,18]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(Xtrain, Ytrain)
grid_search.best_params_
grid_search.best_estimator_
pd.DataFrame(grid_search.cv_results_)
scores.update({'GridSearchRandomForest':-grid_search.best_score_})
scores
scores_df =pd.DataFrame(list(scores.items()),columns=['Model','Score'])
scores_df.sort_values(['Score'])
_ =sns.scatterplot(x='Model',y='Score',data=scores_df,style='Model')
submission=pd.read_csv('../input/sample_submission.csv')
submission['SalePrice']=np.exp(grid_search.best_estimator_.predict(Xtest))-1
submission.head()
submission.to_csv('results.csv',index=False)