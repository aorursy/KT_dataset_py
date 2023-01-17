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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from sklearn.linear_model import LassoCV,ElasticNet,Ridge
train = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')
train.head()
num_features = train.select_dtypes(exclude=['object'],).drop(['SalePrice'],axis=1).copy()

fig = plt.figure(figsize=(18,16))

for index in range(len(num_features.columns)):

    plt.subplot(8,5,index+1)

    sns.distplot(num_features.iloc[:,index],rug=True,hist=False,label='UW',kde_kws={'bw':0.1})

plt.tight_layout()
fig = plt.figure(figsize=(18,16))

for index in range(len(num_features.columns)):

    plt.subplot(8,5,index+1)

    sns.boxplot(y=num_features.iloc[:,index].dropna())

plt.tight_layout()
fig = plt.figure(figsize=(18,16))

for index in range(len(num_features.columns)):

    plt.subplot(8,5,index+1)

    sns.scatterplot(x=num_features.iloc[:,index].dropna(),y=train['SalePrice'])

plt.tight_layout()
cat_features = train.select_dtypes(include=['object']).copy()

fig = plt.figure(figsize=(25,30))

for index in range(len(cat_features.columns)):

    plt.subplot(11,4,index+1)

    plt.xticks(rotation=90)

    sns.barplot(cat_features.iloc[:,index].dropna(),train['SalePrice'])

plt.tight_layout()
fig, axs = plt.subplots(ncols=2,nrows=5,figsize=(20,20))

for ax in fig.axes:

    plt.sca(ax)

sns.regplot(train.LotFrontage,train.SalePrice,ax=axs[0][0])

sns.regplot(train.LotArea,train.SalePrice,ax=axs[0][1])

sns.regplot(train.MasVnrArea,train.SalePrice,ax=axs[1][0])

sns.regplot(train.BsmtFinSF1,train.SalePrice,ax=axs[1][1])

sns.regplot(train.TotalBsmtSF,train.SalePrice,ax=axs[2][0])

sns.regplot(train.GrLivArea,train.SalePrice,ax=axs[2][1])

sns.regplot(train['1stFlrSF'],train.SalePrice,ax=axs[3][0])

sns.regplot(train.EnclosedPorch,train.SalePrice,ax=axs[3][1])

sns.regplot(train.MiscVal,train.SalePrice,ax=axs[4][0])

sns.regplot(train.LowQualFinSF,train.SalePrice,ax=axs[4][1])
train = train.drop(train[train['MiscVal']>5000].index)

train = train.drop(train[(train['LowQualFinSF']>600)&(train.SalePrice>400000)].index)

train = train.drop(train[train['EnclosedPorch']>500].index)

train = train.drop(train[train['1stFlrSF']>4000].index)

train = train.drop(train[train['TotalBsmtSF']>4000].index)

train = train.drop(train[(train['GrLivArea']>4000)&(train.SalePrice>300000)].index)

train = train.drop(train[train['BsmtFinSF1']>4000].index)

train = train.drop(train[train['MasVnrArea']>1200].index)

train = train.drop(train[train['LotFrontage']>200].index)

train = train.drop(train[train['LotArea']>100000].index)
corr_num = train.select_dtypes(exclude=['object']).corr()

fig = plt.figure(figsize=(18,16))

plt.title('numerical features correlation')

sns.heatmap(corr_num > 0.8 , annot=True, square=True)
train = train.drop(columns=['GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageArea'])

test = test.drop(columns=['GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageArea'])
train.drop(columns=['Street', 'Utilities'],inplace=True)

test.drop(columns=['Street', 'Utilities'],inplace=True)
null_val = train.isnull().mean().sort_values(ascending=False)

null_val
train.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'PoolArea'],inplace=True)

test.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'PoolArea'],inplace=True)
test.isnull().mean().sort_values(ascending=False)
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))

test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))

train.MasVnrType.fillna('None',inplace=True)

test.MasVnrType.fillna('None',inplace=True)

train.MasVnrArea.fillna(0,inplace=True)

test.MasVnrArea.fillna(0,inplace=True)

train.BsmtQual.fillna('No Basement',inplace=True)

train.BsmtCond.fillna('No Basement',inplace=True)

train.BsmtFinType2.fillna('No Basement',inplace=True)

train.BsmtFinType1.fillna('No Basement',inplace=True)

train.BsmtExposure.fillna('No Basement',inplace=True)

test.BsmtQual.fillna('No Basement',inplace=True)

test.BsmtCond.fillna('No Basement',inplace=True)

test.BsmtFinType2.fillna('No Basement',inplace=True)

test.BsmtFinType1.fillna('No Basement',inplace=True)

test.BsmtExposure.fillna('No Basement',inplace=True)

train.Electrical.fillna('SBrkr',inplace=True)

train.FireplaceQu.fillna('No Fireplace',inplace=True)

test.FireplaceQu.fillna('No Fireplace',inplace=True)

train.GarageType.fillna('No Garage',inplace=True)

test.GarageType.fillna('No Garage',inplace=True)

train.GarageCond.fillna('No Garage',inplace=True)

test.GarageCond.fillna('No Garage',inplace=True)

train.GarageFinish.fillna('No Garage',inplace=True)

test.GarageFinish.fillna('No Garage',inplace=True)

train.GarageQual.fillna('No Garage',inplace=True)

test.GarageQual.fillna('No Garage',inplace=True)

train.Fence.fillna('No Fence',inplace=True)

test.Fence.fillna('No Fence',inplace=True)

test.Functional.fillna('Typ',inplace=True)

test.BsmtHalfBath.fillna(0,inplace=True)

test.TotalBsmtSF.fillna(0,inplace=True)

test.BsmtFinSF2.fillna(0,inplace=True)

test.BsmtUnfSF.fillna(0,inplace=True)

test.BsmtFullBath.fillna(0,inplace=True)

test.KitchenQual.fillna('TA',inplace=True)

test.SaleType.fillna('WD',inplace=True)

test.BsmtFinSF1.fillna(0,inplace=True)

test.Exterior1st.fillna('VinylSd',inplace=True)

test.GarageCars.fillna(0,inplace=True)

test.Exterior2nd.fillna('VinylSd',inplace=True)

train['MSSubClass'] = train.MSSubClass.apply(str)

test['MSSubClass'] = test.MSSubClass.apply(str)

train['MoSold'] = train.MoSold.apply(str)

test['MoSold'] = test.MoSold.apply(str)

train['YrSold'] = train.YrSold.apply(str)

test['YrSold'] = test.YrSold.apply(str)

train['MSZoning'] = train.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

test['MSZoning'] = test.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
test.isnull().sum().sort_values(ascending=False)
list(train.select_dtypes(exclude = ['object']))
fig, (ax1,ax2,ax3)= plt.subplots(nrows=1,ncols=3)

fig.set_size_inches(20,10)

_=sns.regplot(train.TotalBsmtSF,train.SalePrice,ax=ax1)

_=sns.regplot(train['2ndFlrSF'],train.SalePrice,ax=ax2)

_=sns.regplot(train.TotalBsmtSF+train['2ndFlrSF'],train.SalePrice,ax=ax3)

train['TotalSF']= train.TotalBsmtSF + train['2ndFlrSF']

test['TotalSF']= test.TotalBsmtSF + test['2ndFlrSF']
fig ,axs = plt.subplots(3,2, figsize=(20,10))



sns.barplot(train.BsmtFullBath,train.SalePrice,ax=axs[0,0])

sns.barplot(train.BsmtHalfBath,train.SalePrice,ax=axs[0,1])

sns.barplot(train.FullBath,train.SalePrice,ax=axs[1,0])

sns.barplot(train.FullBath,train.SalePrice,ax=axs[1,1])

sns.barplot(train.BsmtFullBath + train.FullBath + (0.5*train.BsmtHalfBath)+(0.5*train.HalfBath),train.SalePrice,ax=axs[2,0])

train['TotalBath'] = train.BsmtFullBath + train.FullBath + (0.5*train.BsmtHalfBath)+(0.5*train.HalfBath)

test['TotalBath'] = test.BsmtFullBath + test.FullBath + (0.5*test.BsmtHalfBath)+(0.5*test.HalfBath)
fig , axs = plt.subplots(2,2,figsize=(20,10))

sns.regplot(train.YearBuilt,train.SalePrice,ax=axs[0,0])

sns.regplot(train.YearRemodAdd,train.SalePrice,ax=axs[0,1])

sns.regplot((train.YearBuilt + train.YearRemodAdd)/2,train.SalePrice,ax=axs[1,0])

train['YrbuiltRemod'] = (train.YearBuilt + train.YearRemodAdd)/2

test['YrbuiltRemod'] = (test.YearBuilt + test.YearRemodAdd)/2
fig , axes =plt.subplots(3,2,figsize=(18,16))

sns.regplot(train.WoodDeckSF,train.SalePrice,ax=axes[0,0])

sns.regplot(train.OpenPorchSF,train.SalePrice,ax=axes[0,1])

sns.regplot(train.EnclosedPorch,train.SalePrice,ax=axes[1,0])

sns.regplot(train['3SsnPorch'],train.SalePrice,ax=axes[1,1])

sns.regplot(train.ScreenPorch,train.SalePrice,ax=axes[2,0])

sns.regplot((train.WoodDeckSF + train.OpenPorchSF + train.EnclosedPorch + train['3SsnPorch'] + train.ScreenPorch),train.SalePrice,ax=axes[2,1])

train['has2ndFlr'] = train['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)

train['hasBsmt'] = train['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)

train['hasFireplace'] = train['Fireplaces'].apply(lambda x: 1 if x>0 else 0)

test['has2ndFlr'] = test['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)

test['hasBsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)

test['hasFireplace'] = test['Fireplaces'].apply(lambda x: 1 if x>0 else 0)

train['LotArea'] = train.LotArea.astype(np.int64)

test['LotArea'] = test.LotArea.astype(np.int64)

train['MasVnrArea'] = train.MasVnrArea.astype(np.int64)

test['MasVnrArea'] = test.MasVnrArea.astype(np.int64)
fig = plt.figure(figsize=(18,16))

print('Skew of SalePrice:', train.SalePrice.skew())

plt.hist(train.SalePrice)

plt.show()
fig = plt.figure(figsize=(18,16))

print('Skew of the log-transformed SalePrice:', np.log1p(train.SalePrice).skew())

plt.hist(np.log1p(train.SalePrice))

plt.show()
X = train.drop(['SalePrice'],axis=1)

y=np.log1p(train['SalePrice'])

X_train,X_valid,y_train,y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=2)
numerical_cols = [cname for cname in X.columns if

                 X[cname].dtypes in ['int64', 'float']]

categorical_cols = [cname for cname in X.columns if

                   X[cname].nunique() <=30 and

                   X[cname].dtype in ['object']]

my_cols=numerical_cols + categorical_cols

X_train = X_train[my_cols].copy()

X_valid = X_valid[my_cols].copy()

X_test = test[my_cols].copy()
numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')),

                                         ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[

    ('num',numerical_transformer, numerical_cols),

    ('cat',categorical_transformer,categorical_cols)

])
def inv_y(transformed_y):

    return np.exp(transformed_y)

n_folds = 10

my_model = XGBRegressor(learning_rate=0.03,n_estimators = 3488,max_depth=3,min_child_weight=0,gamma= 0,subsample=0.6,colsample_bytree=0.6,

                        objective='reg:squarederror',nthread=-1,scale_pos_weight=1,seed=27,reg_alpha=0.00006)

pipe = Pipeline(steps=[('preprocessor',preprocessor),('model',my_model)])

pipe.fit(X_train,y_train)

preds=pipe.predict(X_valid)

print('XGB :',mean_absolute_error(inv_y(y_valid),inv_y(preds)))



model = LassoCV(max_iter=1e7,  random_state=14, cv=n_folds)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('Lasso: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))



model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('Gradient: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))







model= Ridge(alpha=17)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('Ridge: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))



model =RandomForestRegressor(random_state=1)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)])

clf.fit(X_train, y_train)

predict = clf.predict(X_valid)

print('RandomForest: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))

#XGB : 13569.310466056035

X_test.info()
model  = XGBRegressor(learning_rate=0.03,n_estimators = 3488,max_depth=3,min_child_weight=0,gamma= 0,subsample=0.6,colsample_bytree=0.6,

                        objective='reg:squarederror',nthread=-1,scale_pos_weight=1,seed=27,reg_alpha=0.00006)

final_model = Pipeline(steps=[('preprocessor',preprocessor),('model',model)])

final_model.fit(X_train,y_train)

final_preds=final_model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': inv_y(final_preds)})



output.to_csv('submission.csv', index=False)