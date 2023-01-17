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
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool, cv
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
full_df_ver2 = full_df[[
                            ### This features were added during the last attempt ###
                           'LotFrontage',
                           'LotArea',
                           'Condition2',
                           'YearRemodAdd',
                           'MasVnrArea',
                           'BsmtFinType1',
                           'TotalBsmtSF',
                           'TotRmsAbvGrd',
                           'Fireplaces',
                           'GarageYrBlt',
                           'GarageCars',
    
                            ### Current best result was performed with these features ### 
                           'Street',
                           'Exterior1st',
                           'KitchenQual',
                           'Heating',
                            
                            ### I also removed some features from the first list ###
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
                           #'FullBath',
                           #'BedroomAbvGr',
                           #'KitchenAbvGr',
                           'Functional',
                           'GarageType',
                           #'GarageQual',
                           #'OpenPorchSF',
                           #'PoolArea',
                           'SaleType',
                           'SaleCondition',
                           'SalePrice'
                          ]]
full_df_ver5 = full_df[[
                            ### This features were added during the last attempt ###
                           'LotFrontage',
                           'LotArea',
                           'Condition2',
                           'YearRemodAdd',
                           'MasVnrArea',
                           'BsmtFinType1',
                           'TotalBsmtSF',
                           'TotRmsAbvGrd',
                           'Fireplaces',
                           'GarageYrBlt',
                           'GarageCars',
    
                            ### Current best result was performed with these features ### 
                           'Street',
                           'Exterior1st',
                           'KitchenQual',
                           'Heating',
                            
                            ### I also removed some features from the first list ###
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
full_df_ref_man.index = full_df["Id"]
full_df_ver2.index = full_df['Id']
full_df_ver5.index = full_df['Id']
full_df_ver3 = full_df_ver2.copy()
full_df_ver3['YearBuilt'].corr(full_df_ver3['SalePrice'])
temp = full_df_ver3[['YearBuilt','SalePrice']].groupby('YearBuilt', as_index=False).median()

sns.set_style('whitegrid')
fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,12))

sns.distplot(full_df_ver3['YearBuilt'], kde=False, ax=axes[0], color='black')
sns.lineplot(x=temp['YearBuilt'], y=temp['SalePrice'], ax=axes[1], color='dimgray')

axes[0].set_xlabel('')
axes[1].set_xlabel('Construction date', size=12)
axes[1].set_ylabel('Median price', size=12)
axes[0].set_ylabel('Saturation', size=12)

plt.suptitle('Year of construction and Price distributions', size=18, y=(0.91));
def yearblt_bin(row):
    
    row = row['YearBuilt']
    
    if row <=1900 :
        return 'very old'
    if 1900 < row <= 1930:
        return 'old'
    if 1930 < row <= 1980:
        return 'moderate'
    else:
        return 'new'
    

full_df_ver3['YearBins'] = full_df_ver3.apply(yearblt_bin, axis=1)
full_df_ver3['YearBins']
plt.figure(figsize=(12,4))
sns.distplot(full_df_ver3['GrLivArea'], bins=50, color='black', kde=False);
def area_bin(row):
    
    row = row['GrLivArea']
    
    if row <= 800 :
        return 'small'
    if 800 < row <= 1700:
        return 'medium'
    if 1700 < row <= 2900:
        return 'large'
    else:
        return 'extra_large'
    

full_df_ver3['AreaBins'] = full_df_ver3.apply(area_bin, axis=1)
full_df_ver3['AreaBins'].value_counts()
full_df_ver3 = full_df_ver3.drop(['GrLivArea', 'YearBuilt'], axis=1)
full_df_pol = full_df_ver2.copy()
#full_df_pol = full_df_pol.drop(['Condition2','BsmtFinType1','SaleType'], axis=1)

full_df_pol['OverallQual*2'] = full_df_pol['OverallQual']*2
#full_df_pol['GrLivArea*2'] = full_df_pol['GrLivArea']*2
#full_df_pol['RoomArea'] = full_df_pol['GrLivArea'] / full_df_pol['TotRmsAbvGrd'] 

full_df_upd_0 = pd.get_dummies(full_df_ref_man, drop_first=True)
full_df_enc_2 = pd.get_dummies(full_df_ver2, drop_first=True)
full_df_pol_2 = pd.get_dummies(full_df_pol, drop_first=True)
full_df_upd_3 = pd.get_dummies(full_df_ver3, drop_first=True)
full_df_ver5 = pd.get_dummies(full_df_ver5, drop_first=True)
enc = OrdinalEncoder()
full_df_ver2.columns
full_df_ver3.columns
cat_features = ['LotFrontage', 'Condition2',
       'BsmtFinType1', 'Fireplaces', 'SaleType', 'SaleCondition', 'Street',
       'Exterior1st', 'KitchenQual', 'Heating', 'MSZoning', 
       'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'OverallQual',
       'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
       'CentralAir', 'HeatingQC', 'Electrical', 'Functional', 'GarageType', 'SaleCondition']

cat_features_3 = ['LotFrontage', 'Condition2',
       'BsmtFinType1', 'Fireplaces', 'SaleType', 'SaleCondition', 'Street',
       'Exterior1st', 'KitchenQual', 'Heating', 'MSZoning', 
       'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'OverallQual',
       'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
       'CentralAir', 'HeatingQC', 'Electrical', 'Functional', 'GarageType', 'SaleCondition', 'YearBins',
       'AreaBins']
full_df_ver2_cat = full_df_ver2.copy()
full_df_ver2_cat[cat_features] = enc.fit_transform(full_df_ver2_cat[cat_features]).astype('int')

full_df_ver3_cat = full_df_ver3.copy()
full_df_ver3_cat[cat_features_3] = enc.fit_transform(full_df_ver3_cat[cat_features_3]).astype('int')
RND_ST = 42
X_train_2 = full_df_enc_2.query('index in @train_ind').drop(['SalePrice'], axis=1).reset_index(drop=True)
X_test_2 = full_df_enc_2.query('index in @test_ind').drop(['SalePrice'], axis=1).reset_index(drop=True)

X_train_cat = full_df_ver2_cat.query('index in @train_ind').drop(['SalePrice'], axis=1).reset_index(drop=True).astype('int')
X_test_cat = full_df_ver2_cat.query('index in @test_ind').drop(['SalePrice'], axis=1).reset_index(drop=True).astype('int')

X_train_3 = full_df_upd_3.query('index in @train_ind').drop(['SalePrice'], axis=1).reset_index(drop=True).astype('int')
X_test_3 = full_df_upd_3.query('index in @test_ind').drop(['SalePrice'], axis=1).reset_index(drop=True).astype('int')

X_train_3_cat = full_df_ver3_cat.query('index in @train_ind').drop(['SalePrice'], axis=1).reset_index(drop=True).astype('int')
X_test_3_cat = full_df_ver3_cat.query('index in @test_ind').drop(['SalePrice'], axis=1).reset_index(drop=True).astype('int')

y_train = full_df_upd_0.query('index in @train_ind')['SalePrice'].reset_index(drop=True)


### Validation subsets

#X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_train_0, y_train, test_size=0.2, random_state=RND_ST) 

X_train_sub_2, X_valid_sub_2, y_train_sub_2, y_valid_sub_2 = train_test_split(X_train_2, y_train, test_size=0.2, random_state=RND_ST) 


X_train_sub_c, X_valid_sub_c, y_train_sub_c, y_valid_sub_c = train_test_split(X_train_cat, y_train, test_size=0.2, random_state=RND_ST) 
X_train_sub_3, X_valid_sub_3, y_train_sub_3, y_valid_sub_3 = train_test_split(X_train_3, y_train, test_size=0.2, random_state=RND_ST) 

#X_train_sub_3, X_valid_sub_3, y_train_sub_3, y_valid_sub_3 = train_test_split(X_train_3, y_train, test_size=0.2, random_state=RND_ST) 
X_train_sub_3c, X_valid_sub_3c, y_train_sub_3c, y_valid_sub_3c = train_test_split(X_train_3_cat, y_train, test_size=0.2, random_state=RND_ST) 
X_train_5 = full_df_ver5.query('index in @train_ind').drop(['SalePrice'], axis=1).reset_index(drop=True)
X_test_5 = full_df_ver5.query('index in @test_ind').drop(['SalePrice'], axis=1).reset_index(drop=True)

X_train_sub_5, X_valid_sub_5, y_train_sub_5, y_valid_sub_5 = train_test_split(X_train_5, y_train, test_size=0.2, random_state=RND_ST) 
def mae(model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print('MAE train = ', mean_absolute_error(y_train, y_train_pred))
    print('MAE test = ', mean_absolute_error(y_test, y_test_pred))
RND_ST = 42
### Random Forest Regressor ###

rfr = RandomForestRegressor(n_jobs=-1, random_state=RND_ST)

params_rfr = dict(n_estimators=range(10,500,10),
                  max_features=range(5, 30),
                  max_leaf_nodes = [1,5,10,20])


### Gradient Boosting Regressor ###

gbr = GradientBoostingRegressor(random_state=RND_ST)

params_gbr = dict(n_estimators=range(200,1000,5),
                  max_features=range(5, 40),
                  max_depth=[0,2,3,4],
                  learning_rate = [0.01, 0.1, 0.5, 1],
                  )

params_gbr_nest = dict(n_estimators=range(200,900,5))

params_gbr_other = dict(max_features=range(10, 40),
                        max_depth=[2,3,4],
                        learning_rate = [0.1, 0.3, 1]
                        #max_features = ['auto', 'sqrt', 'log2']
                       )


### CatBoost ###

catboost_train = Pool(X_train_sub_c, y_train_sub_c, cat_features=cat_features)
catboost_train_full = Pool(X_train_cat, y_train, cat_features=cat_features)

catboost_train_3 = Pool(X_train_sub_3c, y_train_sub_3c, cat_features=cat_features_3)
catboost_train_full_3 = Pool(X_train_3_cat, y_train, cat_features=cat_features_3)
catboost_1 = CatBoostRegressor(
                          iterations=720, 
                          depth=4, 
                          learning_rate=0.09, 
                          loss_function='MAE', 
                          subsample=0.8,
                          grow_policy='Depthwise',
                          l2_leaf_reg=2,
                          rsm=0.9,
                          verbose=0, 
                          random_seed=RND_ST
    )
catboost_1.fit(X_train_sub_c, y_train_sub_c)

cat_y_tr = catboost_1.predict(X_train_sub_c)
cat_y_val = catboost_1.predict(X_valid_sub_c)

print('Train mae = ', mean_absolute_error(y_train_sub_c, cat_y_tr))
print('Valid mae = ', mean_absolute_error(y_valid_sub_c, cat_y_val))
### CatBoost best
### Train mae =  6562.590378143246
### Valid mae =  16061.543780248663

X_train_stack_1, X_train_stack_2, y_train_stack_1, y_train_stack_2 = train_test_split(
                                                                        X_train_cat, y_train, test_size=0.5, random_state=RND_ST)
lr = LinearRegression(n_jobs=-1) 

rfr_1 = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_split=3, n_jobs=-1, random_state=RND_ST)

rfr_2 = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_split=4, n_jobs=-1, random_state=RND_ST)

rfr_3 = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_split=5, n_jobs=-1, random_state=RND_ST)

gbr_1 = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.9, random_state=RND_ST)

gbr_2 = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.09, subsample=0.8, random_state=RND_ST)
models = [lr, rfr_1, rfr_2, rfr_3, gbr_1, gbr_2]
names = ['lr', 'rfr_1', 'rfr_2', 'rfr_3', 'gbr_1', 'gbr_2']
for model in models:
    model.fit(X_train_stack_1, y_train_stack_1)
X_train_stack_2_upd = X_train_stack_2.copy()
def pred_stack(model, feat, df_upd, name):
    
    pred = pd.Series(model.predict(feat).astype('int'), name=name, index=feat.index)
    
    df_upd = df_upd.join(pred)
    
    return df_upd
for model, name in zip(models, names):
    
    X_train_stack_2_upd = pred_stack(model, X_train_stack_2, X_train_stack_2_upd, name)
X_train_stack_2_upd
X_test_upd = X_test_cat.copy()

for model, name in zip(models, names):
    
    X_test_upd = pred_stack(model, X_test_cat, X_test_upd, name)
X_test_upd
catboost_stack = CatBoostRegressor(iterations=700, 
                          depth=4, 
                          learning_rate=0.09, 
                          loss_function='MAE', 
                          subsample=0.8,
                          grow_policy='Depthwise',
                          l2_leaf_reg=2,
                          rsm=0.9,
                          verbose=0, 
                          random_seed=RND_ST)
X_train_stack_2_upd
catboost_stack.fit(X_train_stack_2_upd, y_train_stack_2)
pred = catboost_stack.predict(X_train_stack_2_upd)

mean_absolute_error(y_train_stack_2, pred)
pool = Pool(X_train_stack_2_upd, y_train_stack_2)
params = dict(iterations=500, 
                          depth=7, 
                          learning_rate=0.09, 
                          loss_function='MAE', 
                          subsample=0.8,
                          grow_policy='Depthwise',
                          l2_leaf_reg=2,
                          rsm=0.9,
                          verbose=0, 
                          #early_stopping_rounds=20,
                          random_seed=RND_ST)

scores = cv(pool,
            params,
            fold_count=2, 
            plot="True")
catboost_stack = CatBoostRegressor(iterations=700, 
                          depth=4, 
                          learning_rate=0.09, 
                          loss_function='MAE', 
                          subsample=0.8,
                          grow_policy='Depthwise',
                          l2_leaf_reg=2,
                          rsm=0.9,
                          verbose=0, 
                          random_seed=RND_ST)
imp = catboost_stack.feature_importances_
names = X_train_stack_2_upd.columns.tolist()

important = pd.DataFrame(columns=['imp', 'names'])

important['imp'] = imp
important['names'] = names

important = important.sort_values(by='imp', ascending=False).reset_index(drop=True)

important
upd_columns = important['names'][:25]
X_train_stack_2_upd_cols = X_train_stack_2_upd[upd_columns]
X_test_upd_cols = X_test_upd[upd_columns]
catboost_stack.fit(X_train_stack_2_upd_cols, y_train_stack_2)
pred = catboost_stack.predict(X_train_stack_2_upd)

mean_absolute_error(y_train_stack_2, pred)


def prediction(model, feat_tr, feat_test, targ_tr):
    
    model.fit(feat_tr, targ_tr)
    pred_final = pd.DataFrame((model.predict(feat_test)), columns=['SalePrice'])
    
    return(pred_final)
pred = np.around(prediction(catboost_stack, X_train_stack_2_upd_cols, X_test_upd_cols, y_train_stack_2))

submission = pd.DataFrame(subm['Id'])

submission = submission.join(pred)

submission.to_csv('/kaggle/working/cb_new_08.csv', index=False)
submission.head()
submission.head()
