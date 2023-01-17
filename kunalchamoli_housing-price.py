# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# remove warnings

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv',index_col='Id')

test  = pd.read_csv('../input/test.csv',index_col='Id')
print(train.shape)

display(train.head(1))



print(test.shape)

display(test.head(1))
plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)
print ("Skew is:", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()
target = np.log(train.SalePrice)

print ("Skew is:", target.skew())

plt.hist(target, color='blue')

plt.show()
numeric_features = train.select_dtypes(include=[np.number])#getting numeric columns
corr = numeric_features.corr()



print (corr['SalePrice'].sort_values(ascending=False)[1:11], '\n')

print (corr['SalePrice'].sort_values(ascending=False)[-10:])
train.OverallQual.unique()#it means it is rating of some sort
def pivotandplot(data, variable, onVariable, aggfunc):

    pivot_var = data.pivot_table(index = variable,

                                values = onVariable,

                                aggfunc= aggfunc)

    pivot_var.plot(kind='bar', color='blue')

    plt.xlabel(variable)

    plt.ylabel(onVariable)

    plt.xticks(rotation=0)

    plt.show()
pivotandplot(train, 'OverallQual', 'SalePrice', np.median)

#it shows a increasing trend 
# It is a continous variable and hence lets look at the relationship of GrLivArea with SalePrice using a Regression plot



_ = sns.regplot(train['GrLivArea'], train['SalePrice'])
train=train.drop(train[(train['GrLivArea']>4000)&(train['SalePrice']<300000)].index)

sns.regplot(train['GrLivArea'], train['SalePrice'])
sns.regplot(train['GarageArea'], train['SalePrice'])
train = train[train['GarageArea'] < 1200]

sns.regplot(train['GarageArea'], train['SalePrice'])
train['log_SalePrice']=np.log(train['SalePrice']+1)

saleprices=train[['SalePrice','log_SalePrice']]#making of dataframe 



saleprices.head(5)
train=train.drop(columns=['SalePrice','log_SalePrice'])
all_data = pd.concat((train, test))

print(all_data.shape)
type(all_data.isnull().sum().sort_values(ascending = False))
null_data = pd.DataFrame(all_data.isnull().sum().sort_values(ascending=False))[:34]#to use only first 34 rows as a dataframe



null_data.columns = ['Null Count']

null_data.index.name = 'Feature'

null_data
len(all_data)
(null_data/len(all_data)) * 100
train.MiscFeature.unique()
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 

            'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):

    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):

    all_data[col] = all_data[col].fillna(0)
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

figure.set_size_inches(14,10)

_ = sns.regplot(train['TotalBsmtSF'], saleprices['SalePrice'], ax=ax1)

_ =sns.regplot(train['1stFlrSF'], saleprices['SalePrice'], ax=ax2)

_ = sns.regplot(train['2ndFlrSF'], saleprices['SalePrice'], ax=ax3)

_ = sns.regplot(train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF'], saleprices['SalePrice'], ax=ax4)
#Impute the entire data set

all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



#Let's add two new variables for No 2nd floor and no basement

all_data['No2ndFlr']=(all_data['2ndFlrSF']==0)

all_data['NoBsmt']=(all_data['TotalBsmtSF']==0)
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

figure.set_size_inches(14,10)

_ = sns.barplot(train['BsmtFullBath'], saleprices['SalePrice'], ax=ax1)

_ = sns.barplot(train['FullBath'], saleprices['SalePrice'], ax=ax2)

_ = sns.barplot(train['BsmtHalfBath'], saleprices['SalePrice'], ax=ax3)

_ = sns.barplot(train['BsmtFullBath'] + train['FullBath'] + train['BsmtHalfBath'] + train['HalfBath'], saleprices['SalePrice'], ax=ax4)
all_data['TotalBath']=all_data['BsmtFullBath'] + all_data['FullBath'] + all_data['BsmtHalfBath'] + all_data['HalfBath']
all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']
# treat some numeric values as str which is actually a categorical data

all_data['MSSubClass']=all_data['MSSubClass'].astype(str)

all_data['MoSold']=all_data['MoSold'].astype(str)

all_data['YrSold']=all_data['YrSold'].astype(str)
all_data['NoLowQual']=(all_data['LowQualFinSF']==0)

all_data['NoOpenPorch']=(all_data['OpenPorchSF']==0)

all_data['NoWoodDeck']=(all_data['WoodDeckSF']==0)

all_data['NoGarage']=(all_data['GarageArea']==0)

all_data=all_data.drop(columns=['PoolArea','PoolQC']) # most of the houses has no pools. 

all_data=all_data.drop(columns=['MiscVal','MiscFeature']) # most of the houses has no misc feature.
all_data.shape
Basement = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']

Bsmt=all_data[Basement]

Bsmt.head()
from sklearn.preprocessing import LabelEncoder

cond_encoder = LabelEncoder()

Bsmt['BsmtCond']=cond_encoder.fit_transform(Bsmt['BsmtCond'])



exposure_encoder = LabelEncoder()

Bsmt['BsmtExposure'] = exposure_encoder.fit_transform(Bsmt['BsmtExposure'])



finTyp1_encoder = LabelEncoder()

Bsmt['BsmtFinType1'] = finTyp1_encoder.fit_transform(Bsmt['BsmtFinType1'])



finTyp2_encoder = LabelEncoder()

Bsmt['BsmtFinType2'] = finTyp2_encoder.fit_transform(Bsmt['BsmtFinType2'])



qual_encoder = LabelEncoder()

Bsmt['BsmtQual'] = qual_encoder.fit_transform(Bsmt['BsmtQual'])
Bsmt.head(10)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
Bsmt['BsmtScore']= Bsmt['BsmtQual']  * Bsmt['BsmtCond'] * Bsmt['TotalBsmtSF']

all_data['BsmtScore']=Bsmt['BsmtScore']
Bsmt['BsmtFin'] = (Bsmt['BsmtFinSF1'] * Bsmt['BsmtFinType1']) + (Bsmt['BsmtFinSF2'] * Bsmt['BsmtFinType2'])

all_data['BsmtFinScore']=Bsmt['BsmtFin']

all_data['BsmtDNF']=(all_data['BsmtFinScore']==0)
garage=['GarageArea','GarageCars','GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt']

Garage=all_data[garage]



garcond_encoder = LabelEncoder()

Garage['GarageCond'] = garcond_encoder.fit_transform(Garage['GarageCond'])



garfin_encoder = LabelEncoder()

Garage['GarageFinish'] = garfin_encoder.fit_transform(Garage['GarageFinish'])



garqual_encoder = LabelEncoder()

Garage['GarageQual'] = garqual_encoder.fit_transform(Garage['GarageQual'])



gartyp_encoder = LabelEncoder()

Garage['GarageType'] = gartyp_encoder.fit_transform(Garage['GarageType'])
Garage['GarageScore']=(Garage['GarageArea']) * (Garage['GarageCars']) * (Garage['GarageFinish'])*(Garage['GarageQual']) *(Garage['GarageType'])

all_data['GarageScore']=Garage['GarageScore']
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
train=all_data[:len(train)]

test=all_data[len(train):]
# loading pakages for model. 

from sklearn.linear_model import ElasticNet, Lasso

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



#Evaluation Metrics

from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score,mean_absolute_error
def rmse(predict, actual):

    score =mean_squared_error(Ytrain, y_pred)**0.5

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
scores = {}
forest_reg = RandomForestRegressor(random_state=42)

forest_reg.fit(Xtrain, Ytrain)



forest_reg.fit(Xtrain,Ytrain)

y_pred = forest_reg.predict(Xtrain)



print('')

print('####### RandomForest Regression #######')

meanCV = score(forest_reg)

print('Mean CV Score : %.4f' % meanCV)





mse = mean_squared_error(Ytrain,y_pred)

mae = mean_absolute_error(Ytrain, y_pred)

rmse = mean_squared_error(Ytrain, y_pred)**0.5

r2 = r2_score(Ytrain, y_pred)

scores.update({'RandomForest':[meanCV,mse,mae,rmse,r2]})



print('')

print('MSE(RSS)    : %0.4f ' % mse)

print('MAE         : %0.4f ' % mae)

print('RMSE        : %0.4f ' % rmse)

print('R2          : %0.4f ' % r2)
from sklearn.model_selection import GridSearchCV



param_grid = [

  

    {'n_estimators': [70,100], 'max_features': [150]},

   

    {'bootstrap': [True], 'n_estimators': [70,100], 'max_features': [150]},

  ]



forest_reg = RandomForestRegressor(random_state=42)

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(Xtrain, Ytrain)



print('')

print('####### GridSearch RF Regression #######')

meanCV = score(grid_search)

print('Mean CV Score : %.4f' % meanCV)





mse = mean_squared_error(Ytrain,y_pred)

mae = mean_absolute_error(Ytrain, y_pred)

rmse = mean_squared_error(Ytrain, y_pred)**0.5

r2 = r2_score(Ytrain, y_pred)

scores.update({'GridSearchRF':[meanCV,mse,mae,rmse,r2]})



print('')

print('MSE(RSS)    : %0.4f ' % mse)

print('MAE         : %0.4f ' % mae)

print('RMSE        : %0.4f ' % rmse)

print('R2          : %0.4f ' % r2)
model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =42)



model_GBoost.fit(Xtrain,Ytrain)

y_pred = model_GBoost.predict(Xtrain)



print('')

print('####### GradientBoosting Regression #######')

meanCV = score(model_GBoost)

print('Mean CV Score : %.4f' % meanCV)





mse = mean_squared_error(Ytrain,y_pred)

mae = mean_absolute_error(Ytrain, y_pred)

rmse = mean_squared_error(Ytrain, y_pred)**0.5

r2 = r2_score(Ytrain, y_pred)

scores.update({'GradientBoosting':[meanCV,mse,mae,rmse,r2]})



print('')

print('MSE(RSS)    : %0.4f ' % mse)

print('MAE         : %0.4f ' % mae)

print('RMSE        : %0.4f ' % rmse)

print('R2          : %0.4f ' % r2)
GBoost_Predictions=np.exp(model_GBoost.predict(Xtest))-1

output = pd.DataFrame({'Id': Xtest.index,

                       'SalePrice': GBoost_Predictions})

output.to_csv('submission.csv', index=False)