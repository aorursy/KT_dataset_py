# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest,f_classif

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.linear_model import Lasso,LinearRegression,Ridge

from xgboost import XGBRegressor
train_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

# test_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

train_full = train_full.drop('Id',axis=1)

# test_full = test_full.drop('Id',axis=1)

train_full.shape
num_features = train_full.select_dtypes(exclude='object').columns

num_features
cat_features = train_full.select_dtypes(include='object').columns

cat_features
num_data = train_full[num_features].drop('SalePrice',axis=1)



fig = plt.figure(figsize=(16,20))



for i in range(len(num_data.columns)):

    fig.add_subplot(9,4,i+1)

    sns.boxplot(y=num_data.iloc[:,i])

plt.tight_layout()   

plt.show()
target = train_full.SalePrice



fig = plt.figure(figsize=(16,20))



for i in range(len(num_data.columns)):

    fig.add_subplot(9,4,i+1)

    

    sns.scatterplot(num_data.iloc[:,i],target)

    plt.xlabel(num_data.columns[i])

plt.tight_layout()

plt.show()
num_data_insale = train_full.select_dtypes(exclude='object')



correlation = num_data_insale.corr()

plt.figure(figsize=(20,20))

sns.heatmap(correlation,square=True,linewidth=2.2,linecolor='black',annot_kws={'size':12})
correlation['SalePrice'].sort_values(ascending=True).head(15)
correlation['SalePrice'].sort_values(ascending=False).head(15)
num_columns = train_full.select_dtypes(exclude='object').columns

corr_to_price = correlation['SalePrice']

n_rows = 8

n_cols = 5

fig,ax = plt.subplots(n_rows,n_cols,sharey=True,figsize=(16,20))



plt.subplots_adjust(bottom=-0.8)



for i in range(n_rows):

    for j in range(n_cols):

        plt.sca(ax[i,j])

        index = n_cols*i+j

        

        if index<len(num_columns):

            plt.scatter(train_full[num_columns[index]],train_full.SalePrice)

            plt.xlabel(num_columns[index])

            plt.title('Corr to SalePrice:{:.3f}'.format(corr_to_price[index]))

plt.show()



high_corr = ['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF'

             ,'GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']

mid_corr = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtUnfSF','2ndFlrSF'

            ,'BsmtFullBath','HalfBath','Fireplaces','GarageYrBlt','WoodDeckSF','OpenPorchSF',]

corr_ = train_full[high_corr+mid_corr].corr()

plt.figure(figsize=(18,14))

sns.heatmap(corr_,annot=True,square=True,linewidth=2.2,linecolor='black',annot_kws={'size':12})
fig,ax = plt.subplots(2,2,sharex=False,sharey=False,figsize=(12,8))



ax[0][0].scatter(train_full['1stFlrSF'],train_full['TotalBsmtSF'])

ax[0][0].set_title('1stFlrSF - TotalBsmtSF')



ax[0][1].scatter(train_full['GrLivArea'],train_full['TotRmsAbvGrd'])

ax[0][1].set_title('GrLivArea - TotRmsAbvGrd')



ax[1][0].scatter(train_full['GarageCars'],train_full['GarageArea'])

ax[1][0].set_title('GarageCars - GarageArea')



plt.show()

fig,ax = plt.subplots(2,2,sharex=False,sharey=False,figsize=(12,8))



ax[0][0].scatter(train_full['1stFlrSF']+train_full['TotalBsmtSF'],train_full['SalePrice'])

ax[0][0].set_title('1stFlrSF + TotalBsmtSF')



ax[0][1].scatter(train_full['TotRmsAbvGrd']+train_full['GrLivArea'],train_full['SalePrice'])

ax[0][1].set_title('GrLivArea + TotRmsAbvGrd')



ax[1][0].scatter(train_full['GarageCars']+train_full['GarageArea'],train_full['SalePrice'])

ax[1][0].set_title('GarageCars+GarageArea')



plt.show()
fig,ax = plt.subplots(2,2,sharex=False,sharey=False,figsize=(12,8))



ax[0][0].scatter(train_full['GarageCars'],train_full['SalePrice'])

ax[0][0].set_title('GarageCars')



ax[0][1].scatter(train_full['GarageArea'],train_full['SalePrice'])

ax[0][1].set_title('GarageArea')



ax[1][0].scatter(train_full['GarageCars']+train_full['GarageArea'],train_full['SalePrice'])

ax[1][0].set_title('GarageCars+GarageArea')



plt.show()
num_data.isnull().sum().sort_values(ascending=False).head(4)
cat_data = train_full.select_dtypes(include='object')

cat_data.describe()
cat_data.isnull().sum().sort_values(ascending=False).head(17)
train_full['Electrical'].value_counts()
train_full.head()
train_full['LotFrontage'].fillna(train_full['LotFrontage'].median(),inplace=True)

train_full['LotFrontage'].isnull().sum()
train_full['GarageYrBlt'].fillna(0,inplace=True)

train_full['MasVnrArea'].fillna(0,inplace=True)
None_cols=['PoolQC','Alley','Fence','FireplaceQu','GarageQual'

           ,'GarageFinish','GarageType','GarageCond','BsmtFinType2','BsmtExposure'

           ,'BsmtFinType1','BsmtQual','BsmtCond','MasVnrType']



for col in None_cols:

    train_full[col].fillna(value='None',inplace=True)



train_full['Electrical'].fillna('SBrkr',inplace=True)

train_data = train_full.copy()



train_data.isnull().sum().sort_values(ascending=False)



train_data.shape
train_data['LotFrontage'][train_data['LotFrontage']>200].index
train_data = train_data.drop(train_data['LotFrontage'][train_data['LotFrontage']>200].index)

train_data = train_data.drop(train_data['LotArea'][train_data['LotArea']>10000].index)

train_data = train_data.drop(train_data['MasVnrArea'][train_data['MasVnrArea']>1000].index)

# train_data['GarageArea'] = train_data.drop(train_data['GarageArea'][train_data['GarageArea']>1250].index)

train_data = train_data.drop(train_data['GrLivArea'][train_data['GrLivArea']>4500].index)

train_data = train_data.drop(train_data['WoodDeckSF'][train_data['WoodDeckSF']>800].index)

train_data = train_data.drop(train_data['OpenPorchSF'][train_data['OpenPorchSF']>400].index)
train_data.shape
drop_features = ['MiscFeature','MSSubClass','OverallCond','BsmtFinSF2'

                 ,'LowQualFinSF','BsmtHalfBath','BedroomAbvGr','KitchenAbvGr'

                 ,'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal',

                 'MoSold','YrSold','GarageArea','TotalBsmtSF','TotRmsAbvGrd','GrLivArea','1stFlrSF']



train_data['GrLivTGrd'] = train_full['GrLivArea'] + train_full['TotRmsAbvGrd']

train_data['1stTBSF'] = train_full['1stFlrSF'] + train_full['TotalBsmtSF']

train_data = train_data.drop(drop_features,axis=1)

train_data.shape
train_data.isnull().sum().sort_values(ascending=False).head()
train_data.head()
cat_features = list(train_data.select_dtypes(include='object').columns)

len(cat_features)
X = pd.get_dummies(train_data.drop('SalePrice',axis=1))

y = np.log(train_data.SalePrice)

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,shuffle=True)

my_imputer = SimpleImputer()

imputed_X_train = my_imputer.fit_transform(X_train)

imputed_X_val = my_imputer.transform(X_val)

print('Train_set size:',X_train.shape)

print('Valid_set size:',X_val.shape)
def inv(transformed_y):

    return np.exp(transformed_y)



mae = {}



#1.LR

LR = LinearRegression()

LR.fit(X_train,y_train)

LR_pred = LR.predict(X_val)

LR_mae = mean_absolute_error(inv(LR_pred),inv(y_val))

mae['LR_MAE'] = LR_mae



#2.Lasso

lasso = Lasso(alpha=0.0005,random_state=5)

lasso.fit(X_train,y_train)

Lasso_pred = lasso.predict(X_val)

Lasso_mae = mean_absolute_error(inv(Lasso_pred),inv(y_val))

mae['Lasso_MAE'] = Lasso_mae



#3.RF

RF = RandomForestRegressor(n_estimators=100)

RF.fit(X_train,y_train)

RF_pred = RF.predict(X_val)

RF_mae = mean_absolute_error(inv(RF_pred),(y_val))

mae['RF_MAE'] = RF_mae



#4.Ridge

Rg = Ridge(alpha=0.002,random_state=5)

Rg.fit(X_train,y_train)

Rg_pred = Rg.predict(X_val)

Rg_mae = mean_absolute_error(inv(Rg_pred),inv(y_val))

mae['Rg_MAE'] = Rg_mae



#5.XGboost

XG = XGBRegressor(n_estimator=500,learning_rate=0.01)

XG.fit(X_train,y_train,early_stopping_rounds=5,eval_set=[(X_val,y_val)])

XG_pred = XG.predict(X_val)

XG_mae = mean_absolute_error(inv(XG_pred),inv(y_val))

mae['XG_MAE'] = XG_mae



print(mae)
from sklearn.model_selection import cross_val_score



score = cross_val_score(lasso,X,y,scoring='neg_mean_squared_error',cv=10,)



RF_score = np.sqrt(-score)



print('For RF model:')



print('Mean RMSE:{:.3f}'.format(RF_score.mean()))



print('Error std deviation:{:.3f}'.format(RF_score.std()))
from sklearn.model_selection import GridSearchCV



param = [{'alpha':[0.0001,0.001,0.005,0.01]}]



top_reg = Lasso()

grid_search = GridSearchCV(top_reg,param,cv=5,scoring='neg_mean_squared_error')

grid_search.fit(X,y)

grid_search.best_params_
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

test_data = test.drop('Id',axis=1)

Id = test.Id
test_data['LotFrontage'].fillna(train_full['LotFrontage'].median(),inplace=True)

test_data['LotFrontage'].isnull().sum()
test_data['GarageYrBlt'].fillna(0,inplace=True)

test_data['MasVnrArea'].fillna(0,inplace=True)
None_cols=['PoolQC','Alley','Fence','FireplaceQu','GarageQual'

           ,'GarageFinish','GarageType','GarageCond','BsmtFinType2','BsmtExposure'

           ,'BsmtFinType1','BsmtQual','BsmtCond','MasVnrType']



for col in None_cols:

    test_data[col].fillna(value='None',inplace=True)



test_data['Electrical'].fillna('SBrkr',inplace=True)

test_data_ = test_data.drop(drop_features,axis=1).copy()



test_data_.isnull().sum().sort_values(ascending=False)
test_data_['GrLivTGrd'] = test_data['GrLivArea'] + test_data['TotRmsAbvGrd']

test_data_['1stTBSF'] = test_data['1stFlrSF'] + test_data['TotalBsmtSF']

test_data_.shape
X_test = pd.get_dummies(test_data_)

final_train, final_test = X.align(X_test, join='left', axis=1)
imputed_test = my_imputer.transform(final_test)

imputed_test.shape
final_model = Lasso(alpha=0.001,random_state=5)

final_X_train = my_imputer.fit_transform(final_train)

final_model.fit(final_X_train,y)
final_preds = final_model.predict(imputed_test)
outcome = pd.DataFrame({'Id':Id,'SalePrice':inv(final_preds)})

outcome.shape
outcome.to_csv('outcome3.csv',index=False)