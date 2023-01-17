#common

import numpy as np

import pandas as pd 

import IPython

from IPython.display import display

import warnings

warnings.simplefilter('ignore')

#visualisation

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objects as go

import plotly.express as px

import matplotlib.style as style

from matplotlib.colors import ListedColormap
from sklearn.metrics import SCORERS

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder

from sklearn.preprocessing import PolynomialFeatures

from sklearn.utils import shuffle, resample

from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA, IncrementalPCA
#regressors

from sklearn.dummy import DummyRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import LinearSVR, SVR

import xgboost as xgb
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

subm = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train_df.info()
train_df.columns
train_df['SalePrice'].describe()
sns.set_style('darkgrid')



fig,ax = plt.subplots(1,1,figsize=(8,6))

sns.distplot(train_df['SalePrice'], ax=ax)



ax.set_xlabel('House price, USD')

plt.suptitle('Price distribution', size=15)

plt.show()
len(train_df.query('SalePrice > 500000'))
len(train_df), len(test_df)
train_df.isna().sum().sort_values(ascending=False).head(10)
test_df.isna().sum().sort_values(ascending=False).head(10)
train_df = train_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)

test_df = test_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
temp = train_df.isna().sum().sort_values()

temp[temp>=1]
temp = test_df.isna().sum().sort_values()

temp[temp>=1]
full_df = pd.concat([train_df] + [test_df]).reset_index(drop=True)
full_df
train_ind = train_df['Id']

test_ind = test_df['Id']
test_ind
full_df.head()
temp = full_df.isna().sum().sort_values()

temp[temp>=1]
full_df['FireplaceQu'] = full_df['FireplaceQu'].fillna('None')

full_df['GarageQual'] = full_df['GarageQual'].fillna('None')

full_df['GarageFinish'] = full_df['GarageFinish'].fillna('None')

full_df['GarageCond'] = full_df['GarageCond'].fillna('None')

full_df['GarageType'] = full_df['GarageType'].fillna('None')

full_df['BsmtExposure'] = full_df['BsmtExposure'].fillna('None')

full_df['BsmtQual'] = full_df['BsmtQual'].fillna('None')

full_df['BsmtCond'] = full_df['BsmtCond'].fillna('None')

full_df['BsmtFinType2'] = full_df['BsmtFinType2'].fillna('None')

full_df['BsmtFinType1'] = full_df['BsmtFinType1'].fillna('None')

full_df['MasVnrType'] = full_df['MasVnrType'].fillna('None')

full_df['BsmtFinType2'] = full_df['BsmtFinType2'].fillna('None')
full_df.isna().sum().sort_values(ascending=False).head(20)
temp = full_df[['LotFrontage','LotArea']]



plt.figure(figsize=(10,6))

sns.scatterplot(x=temp['LotFrontage'], y=temp['LotArea'])

plt.title('Correlations between Lot Area and Lot Frontage', size=15);



print(temp.corr())
full_df['LotFrontage'] = full_df['LotFrontage'].fillna(np.sqrt(full_df['LotArea']))
temp = full_df[['LotFrontage','LotArea']]



plt.figure(figsize=(10,6))

sns.scatterplot(x=temp['LotFrontage'], y=temp['LotArea'])

plt.title('Correlations between Lot Area and Lot Frontage with filled missing values', size=15);



print(temp.corr())
temp_year = full_df[['GarageYrBlt', 'YearBuilt']]



temp_year
plt.figure(figsize=(10,7))

sns.scatterplot(temp_year['YearBuilt'], temp_year['GarageYrBlt'])

plt.title('Were houses and garages built at the same time?', size=15);
full_df.query('GarageYrBlt>2100')['GarageYrBlt']
full_df.loc[full_df['GarageYrBlt'] == 2207,'GarageYrBlt'] = 2007
full_df['GarageYrBlt'] = full_df['GarageYrBlt'].fillna(full_df['YearBuilt'])
full_df.isna().sum().sort_values(ascending=False).head(10)
full_df['GarageArea'] = full_df.groupby('GarageType')['GarageArea'].transform(lambda x: x.fillna(value=x.median()))
full_df['GarageCars'].corr(full_df['GarageArea'])
full_df.loc[full_df['GarageCars'].isna()]['GarageArea']
full_df.loc[full_df['GarageArea'] == 400]['GarageCars'].value_counts()
full_df['GarageCars'] = full_df['GarageCars'].fillna(2)
full_df.loc[full_df['MasVnrArea'].isna()][['MasVnrArea', 'MasVnrType']]
full_df['MasVnrArea'] = full_df['MasVnrArea'].fillna(0)
full_df.loc[full_df['MSZoning'].isna()]
full_df['MSZoning'].value_counts()
full_df['MSZoning'] = full_df['MSZoning'].fillna(value='RL')
full_df.loc[full_df['Utilities'].isna()]['YearBuilt'] 
print(full_df.loc[full_df['YearBuilt'] == 1910]['Utilities'].value_counts())

print(full_df.loc[full_df['YearBuilt'] == 1952]['Utilities'].value_counts())
full_df['Utilities'] = full_df['Utilities'].fillna(value='AllPub')
full_df['BsmtHalfBath'].value_counts()
full_df['BsmtFullBath'].value_counts()
full_df.query('BsmtHalfBath=="nan" or BsmtFullBath=="nan"')[['BsmtHalfBath', 'BsmtFullBath', 'YearBuilt']]
full_df.query('YearBuilt == 1959')['BsmtHalfBath'].value_counts()

#full_df.query('YearBuilt == 1946')['BsmtHalfBath']
full_df[['BsmtHalfBath', 'BsmtFullBath']] = full_df[['BsmtHalfBath', 'BsmtFullBath']].fillna(value=0)
full_df.Functional.value_counts()
full_df['Functional'] = full_df['Functional'].fillna('Typ')
full_df.isna().sum().sort_values(ascending=False).head(10)
full_df['BsmtFinSF2'].value_counts()
full_df['BsmtFinSF2'] = full_df['BsmtFinSF2'].fillna(0)
full_df.loc[full_df['BsmtFinSF1'].isna()]['BsmtFinType1']
full_df['BsmtFinSF1'] = full_df['BsmtFinSF1'].fillna(0)
full_df.loc[full_df['TotalBsmtSF'].isna(), 'BsmtFinSF1']
full_df[['TotalBsmtSF', 'BsmtFinSF1']]
full_df['TotalBsmtSF'].corr(full_df['SalePrice'])
full_df.isna().sum().sort_values(ascending=False).head(10)
full_df.loc[full_df['TotalBsmtSF'].isna()]['OverallQual']
full_df.loc[full_df['OverallQual']==4]['BsmtUnfSF'].value_counts()
full_df[['TotalBsmtSF','BsmtUnfSF']] = full_df[['TotalBsmtSF','BsmtUnfSF']].fillna(0)
full_df['SaleType'].value_counts()
full_df['SaleType'] = full_df['SaleType'].fillna('WD')
full_df.loc[full_df['Exterior2nd'].isna()][['Exterior2nd','Exterior1st','YearBuilt']]
full_df.loc[full_df['YearBuilt'] == 1940][['Exterior1st', 'Exterior2nd', 'MSZoning']]
full_df.loc[full_df['YearBuilt'] == 1940]['Exterior1st'].value_counts()
full_df.loc[full_df['YearBuilt'] == 1940]['Exterior2nd'].value_counts()
full_df[['Exterior1st','Exterior2nd']] = full_df[['Exterior1st','Exterior2nd']].fillna('MetalSd')
full_df.loc[full_df['Electrical'].isna()]['YearBuilt']
full_df.loc[full_df['YearBuilt'] == 2006]['Electrical'].value_counts()
full_df['Electrical'] = full_df['Electrical'].fillna(value='SBrkr')
full_df.loc[full_df['KitchenQual'].isna()]['YearBuilt']
full_df.loc[full_df['YearBuilt']==1917][['KitchenQual', 'OverallCond']]
full_df.loc[full_df['OverallCond']==3]['KitchenQual'].value_counts()
full_df['KitchenQual'] = full_df['KitchenQual'].fillna(value='TA')
full_df.isna().sum().sort_values()
full_df['SalePrice'].corr(full_df['YearBuilt'])
full_df.columns
full_df['Heating'].value_counts()
full_df.groupby(['Heating'])['SalePrice'].median().sort_values(ascending=False)
full_df.columns
full_df_ref_man = full_df[[

                           'Street',

                           'Exterior1st',

                           'KitchenQual',

                           'Heating',

    

                           'MSZoning',

                           'YearBuilt',

                           'Neighborhood',

                           'Condition1',

                           'BldgType',

                           'HouseStyle',

                           'OverallQual',

                           'OverallCond',

                           'ExterQual',

                           'ExterCond', 

                           'BsmtQual',

                           'BsmtCond',

                           'CentralAir',

                           'HeatingQC',

                           'Electrical',

                           '1stFlrSF',

                           '2ndFlrSF',

                           'GrLivArea',

                           'FullBath',

                           'BedroomAbvGr',

                           'KitchenAbvGr',

                           'Functional',

                           'GarageType',

                           'GarageQual',

                           'OpenPorchSF',

                           'PoolArea',

                           'SaleType',

                           'SaleCondition',

                           'SalePrice'

                          ]]
full_df_ref_man_1 = full_df[[

                           'MSZoning',

                           'Utilities',

                           'Condition1',                           

                           'OverallQual',

                           'OverallCond',

                           'ExterQual',

                           'ExterCond', 

                           'BsmtQual',

                           'BsmtCond',

                           'HeatingQC',

                           '1stFlrSF',

                           '2ndFlrSF',

                           'GrLivArea',

                           'GarageQual',

                           'OpenPorchSF',

                           'PoolArea',

                           'SalePrice'

                          ]]
full_df_ref_man.index = full_df["Id"]

full_df_ref_man_1.index = full_df['Id']
full_df_ref_man.head()
full_df_ref_man_1.head()
full_df_upd_0 = pd.get_dummies(full_df_ref_man, drop_first=True)

full_df_upd_1 = pd.get_dummies(full_df_ref_man_1, drop_first=True)
X_train_0 = full_df_upd_0.query('index in @train_ind').drop(['SalePrice'], axis=1).reset_index(drop=True)

X_test_0 = full_df_upd_0.query('index in @test_ind').drop(['SalePrice'], axis=1).reset_index(drop=True)



X_train_1 = full_df_upd_1.query('index in @train_ind').drop(['SalePrice'], axis=1).reset_index(drop=True)

X_test_1 = full_df_upd_1.query('index in @test_ind').drop(['SalePrice'], axis=1).reset_index(drop=True)



y_train = full_df_upd_0.query('index in @train_ind')['SalePrice'].reset_index(drop=True)
def mae(model, X_train, X_test, y_train, y_test):

    

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)

    

    print('MAE train = ', mean_absolute_error(y_train, y_train_pred))

    print('MAE test = ', mean_absolute_error(y_test, y_test_pred))
RND_ST = 42
rfr = RandomForestRegressor(n_jobs=-1, random_state=RND_ST)



params_rfr = dict(n_estimators=range(10,500,10),

                  max_features=range(5, 30),

                  max_leaf_nodes = [1,5,10,20])



gbr = GradientBoostingRegressor(random_state=RND_ST)



params_gbr = dict(n_estimators=range(300,1000,5),

                  max_features=range(5, 40),

                  max_depth=[0,2,3, 4],

                  learning_rate = [0.01, 0.1, 0.5, 1],

                  )



params_gbr_nest = dict(n_estimators=range(200,800,5))



params_gbr_other = dict(max_features=range(10, 35),

                        max_depth=[2,3,4],

                        learning_rate = [0.1,0.5,1])

                       
dtrain = xgb.DMatrix(data=X_train_0, label=y_train)

dtrain_data_only = xgb.DMatrix(data=X_train_0)

dtest = xgb.DMatrix(data=X_test_0)



param = {'max_depth': 3, 

         'eta':1,

         }
num_round = 30

bst = xgb.train(param, dtrain, num_round)



#mae(bst, dtrain_data_only, y_train)
pred_bst = pd.DataFrame(bst.predict(dtest), columns=['SalePrice'])
pred_bst = pd.DataFrame(bst.predict(dtest), columns=['SalePrice'])



submission = pd.DataFrame(subm['Id'])



submission = submission.join(pred_bst)



submission.to_csv('/kaggle/working/xgb.csv', index=False)
def random_search(model, params, feat, targ):

    

    

    search = RandomizedSearchCV(model, params, cv=9, scoring='neg_mean_absolute_error', n_jobs=-1)

    search.fit(feat, targ)

    

    print(search.best_score_)

    print(search.best_params_)
def grid_search(model, params, feat, targ):

    

    

    search = GridSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)

    search.fit(feat, targ)

    

    print(search.best_score_)

    print(search.best_params_)
random_search(gbr, params_gbr, X_train_0, y_train)
grid_search(gbr, params_gbr_nest, X_train_0, y_train)
gbr_est = GradientBoostingRegressor(n_estimators=265, random_state=RND_ST)
grid_search(gbr_est, params_gbr_other, X_train_0, y_train)
gbr_new = GradientBoostingRegressor(n_estimators=265, max_depth=4, max_features=28, random_state=RND_ST)
def valid_split(feat, targ):

    

    X_tr_sub, X_test_sub, y_tr_sub, t_test_sub = train_test_split(feat, targ, test_size=0.2, random_state=RND_ST)

    return(X_tr_sub, X_test_sub, y_tr_sub, t_test_sub)
X_train_sub, X_test_sub, y_train_sub, y_test_sub = valid_split(X_train_0, y_train)
random_search(gbr, params_gbr, X_train_0, y_train)
random_search(gbr, params_gbr, X_train_0, y_train)
model_rfr = RandomForestRegressor(n_estimators=270, max_leaf_nodes=20, max_features=23, n_jobs=-1, random_state=RND_ST)
model_gbr = GradientBoostingRegressor(n_estimators=590, max_features=24, max_depth=6, learning_rate=0.08, random_state=RND_ST)
model_gbr_ = GradientBoostingRegressor(n_estimators=810, max_features=24, max_depth=3, learning_rate=0.1, random_state=RND_ST)
model_gbr_more_feat = GradientBoostingRegressor(n_estimators=2000, max_features=26, max_depth=2, learning_rate=0.1, random_state=RND_ST)
gbr_new.fit(X_train_0, y_train)

pred = gbr_new.predict(X_train_0)

mean_absolute_error(y_train, pred)
mae(gbr_new, X_train_sub, X_test_sub, y_train_sub, y_test_sub)
def prediction(model, feat_tr, feat_test, targ_tr):

    

    model.fit(feat_tr, targ_tr)

    pred_final = pd.DataFrame((model.predict(feat_test)), columns=['SalePrice'])

    

    return(pred_final)
pred = np.around(prediction(gbr_new, X_train_0, X_test_0, y_train))



submission = pd.DataFrame(subm['Id'])



submission = submission.join(pred)



submission.to_csv('/kaggle/working/grad_boost_new.csv', index=False)
submission.head()