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
train=pd.read_csv("../input/train.csv")

test=pd.read_csv('../input/test.csv')
train.info()
missing_1=train.isnull().sum(axis=0)

missing_2=test.isnull().sum(axis=0)

print('missing data in trainning data')

print(missing_1[missing_1>0])

print('\n missing data in testing data')

print(missing_2[missing_2>0])
train.columns.values

train_val=train.drop('SalePrice',axis=1)

test_val=test

val_total=pd.concat([train_val,test_val])
miss=val_total.isnull().sum(axis=0).sort_values(ascending=False)

miss=miss[miss>0]

miss
#for catergorical variables, we replece missing data with None

Miss_cat=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 

          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 

          'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']

for col in Miss_cat:

    val_total[col].fillna('None',inplace=True)

# for numerical variables, we replace missing value with 0

Miss_num=['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 

          'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'] 

for col in Miss_num:

    val_total[col].fillna(0, inplace=True)
miss=val_total.isnull().sum(axis=0).sort_values(ascending=False)

miss=miss[miss>0]

miss
rest_val=['MSZoning','Functional','Utilities','Exterior1st', 'SaleType','Electrical', 'Exterior2nd','KitchenQual']

for col in rest_val:

    val_total[col].fillna(val_total[col].mode()[0],inplace=True)
miss=val_total.isnull().sum(axis=0).sort_values(ascending=False)

miss=miss[miss>0]

miss
val_total['LotFrontage']=val_total.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in ['MSSubClass', 'YearBuilt', 'YearRemodAdd']:

    val_total[col]=val_total[col].astype(str)
new_train_data=pd.concat([val_total[:len(train)],train['SalePrice']],axis=1)
Cat=[]

Num=[]

for col in new_train_data.columns:

    if new_train_data.dtypes[col]=='object':

        Cat.append(col)

    else:

        Num.append(col)
Num.remove('Id')

Num.remove('SalePrice')
##Check correlation between Num variables and SalePrice

import matplotlib.pyplot as plt

import seaborn as sns

Num_data=pd.concat([new_train_data[Num],new_train_data['SalePrice']],axis=1)

plt.figure(figsize=(20,10))

sns.heatmap(Num_data.corr())

##show the relation between SalesPrice and Each numerical Variables:

i=0

sns.set_style('whitegrid')

plt.figure()

fig,ax=plt.subplots(6,6,figsize=(20,20))

for feature in Num:

    i+=1

    plt.subplot(6,6,i)

    sns.scatterplot(new_train_data['SalePrice'],new_train_data[feature])

    plt.xlabel('SalePrice')

    plt.ylabel(feature)

plt.tight_layout()

plt.show()
##show relation between Sales price and catergorical variables

i=0

sns.set_style('whitegrid')

plt.figure()

fig,ax=plt.subplots(8,6,figsize=(20,40))

for feature in Cat:

    i+=1

    plt.subplot(8,6,i)

    sns.boxplot(new_train_data[feature],new_train_data['SalePrice'])

    plt.ylabel('SalePrice')

    plt.xlabel(feature)

plt.tight_layout()

plt.show()
i=0

sns.set_style('whitegrid')

plt.figure()

fig,ax=plt.subplots(6,6,figsize=(20,20))

for feature in Num:

    i+=1

    plt.subplot(6,6,i)

    sns.distplot(val_total[feature],kde=False)

    plt.xlabel(feature)

plt.tight_layout()

plt.show()
#check skewness of variables

from scipy.stats import skew

skewness = val_total[Num].apply(lambda x: skew(x)).sort_values(ascending=False)

skewness=pd.DataFrame({'skewness':skewness})

from scipy.special import boxcox1p

skew_var=skewness[abs(skewness['skewness'])>0.75].index

for var in skew_var:

    val_total[var]=boxcox1p(val_total[var], 0.15)

i=0

sns.set_style('whitegrid')

plt.figure()

fig,ax=plt.subplots(6,6,figsize=(20,20))

for feature in Num:

    i+=1

    plt.subplot(6,6,i)

    sns.distplot(val_total[feature],kde=False)

    plt.xlabel(feature)

plt.tight_layout()

plt.show()
from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

min_max_scaler=preprocessing.MinMaxScaler()

min_max_val=min_max_scaler.fit_transform(val_total[Num])
min_max_val.shape
dummy_cat = pd.get_dummies(val_total[Cat])

##convert dummy_cat to numpy array

dummy_cat=np.array(dummy_cat)
dummy_cat.shape
total_feature=np.concatenate([min_max_val,dummy_cat],axis=1)
train_feature=total_feature[: len(train),:]

test_feature=total_feature[len(train):,:]
Y=train['SalePrice']
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(train_feature,Y,test_size=0.2)
##XGBOOST 

from xgboost.sklearn import XGBRegressor

from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

random_grid={'learning_rate':[0.001,0.01],

            'max_depth':[10,30],

            'n_estimators':[200,300],

            'subsample':[0.5,0.7]

}

xgb = XGBRegressor(objective='reg:linear')

grid_search=GridSearchCV(estimator=xgb,param_grid = random_grid,cv = 3, n_jobs = -1, verbose = 2,scoring='neg_mean_squared_error')

                         
grid_search.fit(X_train,Y_train)

print("\nGrid Search Best parameters set :")

print(grid_search.best_params_)
predict=grid_search.predict(X_test)

mse = np.mean((Y_test - predict)**2)

print('MSE:', mse)
sns.scatterplot(Y_test,predict)

plt.xlabel('True Price')

plt.ylabel('Predicted Price')

plt.show()
Test_predict=grid_search.predict(test_feature)
prediction = pd.DataFrame(Test_predict, columns=['SalePrice'])

result = pd.concat([ test['Id'], prediction], axis=1)
result.to_csv('./submission.csv', index=False)