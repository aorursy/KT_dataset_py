import numpy as np 

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt  

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

from scipy import stats

from scipy.stats import norm, skew 



def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(train.head())

print(train.describe())

print(train.shape)

print(test.head())

print(test.describe())

print(test.shape)
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train['Log_SalePrice'] = np.log1p(train['SalePrice'])



#Check the new distribution 

sns.distplot(train['Log_SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['Log_SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Log_SalePrice distribution')



#Get the QQ-plot

fig = plt.figure()

res = stats.probplot(train['Log_SalePrice'], plot=plt)

plt.show()
y_train = train['Log_SalePrice']



all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop('SalePrice', axis=1, inplace=True)

all_data.drop('Log_SalePrice', axis=1, inplace=True)

all_data.shape
corrmat = all_data.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True, center = 0, cmap = 'viridis')
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]



f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na, color='black')

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data, by feature', fontsize=15)



all_data['PoolQC'].value_counts()
print(len(all_data['PoolArea'].value_counts()))

all_data['PoolArea'].value_counts()
all_data.drop('PoolQC', axis=1, inplace=True)

train.drop('PoolQC', axis=1, inplace=True)

test.drop('PoolQC', axis=1, inplace=True)
all_data['MiscFeature'].value_counts()
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['Alley'].value_counts()

all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'].value_counts()

all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'].value_counts()

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
all_data['LotFrontage'].value_counts()

all_data['LotFrontage'] = all_data['LotFrontage'].fillna(0)
print("YearBuilt: " + str(all_data['GarageYrBlt'].count()))

print("Condition: "+ str(all_data['GarageCond'].count()))

print("Quality: "+ str(all_data['GarageQual'].count()))

print("Finish: " + str(all_data['GarageFinish'].count()))

print("Type: " + str(all_data['GarageType'].count()))
all_data.loc[all_data['GarageCond'].notna() & all_data['GarageYrBlt'].notna() & all_data['GarageQual'].notna()

            & all_data['GarageFinish'].notna() & all_data['GarageType'].notna(),

            ['GarageCond', 'GarageYrBlt','GarageQual','GarageFinish','GarageType']]
all_data.loc[all_data['GarageCond'].isna() & all_data['GarageType'].notna(), 

             ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageYrBlt']]



all_data.loc[all_data['GarageCond'].isna() & all_data['GarageType'].notna(), ['GarageType']] = 'None'
all_data['GarageQual'].value_counts()

all_data['GarageQual'] = all_data['GarageQual'].fillna('None')
all_data['GarageCond'].value_counts()

all_data['GarageCond'] = all_data['GarageCond'].fillna('None')
print(all_data['GarageFinish'].value_counts())

print(all_data['GarageYrBlt'].value_counts())

print(all_data['GarageType'].value_counts())
all_data['GarageFinish'] = all_data['GarageFinish'].fillna('None')

all_data['GarageType'] = all_data['GarageType'].fillna('None')

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)
gb_conds = [

    all_data['GarageYrBlt'] == all_data['YearBuilt'],

    all_data['GarageYrBlt'] == 0,

    all_data['GarageYrBlt'] > all_data['YearBuilt'],

    all_data['GarageYrBlt'] < all_data['YearBuilt']

    ]



gb_outputs = ['With_House', 'None', 'Renovated_Garage', 'Old_Garage']



all_data['GarageYrBlt_Cat'] = np.select(gb_conds, gb_outputs, 'Other')

all_data['GarageYrBlt_Cat'].value_counts()
print("Exposure: " + str(all_data['BsmtExposure'].count()))

print("Condition: "+ str(all_data['BsmtCond'].count()))

print("Quality: "+ str(all_data['BsmtQual'].count()))

print("Fin1: "+ str(all_data['BsmtFinType1'].count()))

print("Fin2: "+ str(all_data['BsmtFinType2'].count()))
len(all_data.loc[all_data['BsmtCond'].notna() & all_data['BsmtExposure'].notna() & all_data['BsmtQual'].notna() & 

             all_data['BsmtFinType1'].notna() & all_data['BsmtFinType2'].notna()])
all_data.loc[(all_data['BsmtCond'].isna() | all_data['BsmtExposure'].isna() | all_data['BsmtQual'].isna() | 

             all_data['BsmtFinType1'].isna() | all_data['BsmtFinType2'].isna()) & 

             (all_data['BsmtCond'].notna() | all_data['BsmtExposure'].notna() | all_data['BsmtQual'].notna() | 

             all_data['BsmtFinType1'].notna() | all_data['BsmtFinType2'].notna()), 

             ['BsmtCond','BsmtExposure','BsmtQual', 'BsmtFinType1','BsmtFinType2' ]]
all_data['BsmtCond'].value_counts().plot(kind='bar')
all_data.loc[(all_data['BsmtCond'].isna() & all_data['BsmtExposure'].notna()), 'BsmtCond'] = 'TA'
all_data['BsmtExposure'].value_counts().plot(kind='bar')
all_data.loc[(all_data['BsmtCond'].notna() & all_data['BsmtExposure'].isna()), 'BsmtExposure'] = 'No'
all_data['BsmtQual'].value_counts().plot(kind='bar')
all_data.loc[(all_data['BsmtQual'].isna() & all_data['BsmtExposure'].notna()), 'BsmtQual'] = 'Gd'
all_data.loc[(all_data['BsmtFinType2'].isna() & all_data['BsmtFinType1'].notna()),

            ['BsmtFinType1', 'BsmtFinType2','TotalBsmtSf', 'BsmtFinSF1', 'BsmtFinSF2']]
all_data['BsmtFinType2'].value_counts().plot(kind='bar')
all_data.loc[(all_data['BsmtFinType2'].isna() & all_data['BsmtFinType1'].notna()), 'BsmtFinType2'] = 'Unf'

all_data.loc[(all_data['BsmtCond'].isna() & all_data['BsmtExposure'].isna() & all_data['BsmtQual'].isna() & 

             all_data['BsmtFinType1'].isna() & all_data['BsmtFinType2'].isna()), 

             ['BsmtCond','BsmtExposure','BsmtQual', 'BsmtFinType1','BsmtFinType2']] = 'None'
print("Type: " + str(all_data['MasVnrType'].count()))

print("Area: "+ str(all_data['MasVnrArea'].count()))
all_data.loc[(all_data['MasVnrType'].isna() | all_data['MasVnrArea'].isna()), 

             ['MasVnrArea', 'MasVnrType', 'Exterior1st', 'Exterior2nd']]
all_data['MasVnrType'].value_counts().plot(kind='bar')
all_data['MasVnrArea'].hist()
all_data.loc[all_data['MasVnrType'].isna(), 'MasVnrType'] = 'None'

all_data.loc[all_data['MasVnrArea'].isna(), 'MasVnrArea'] = 0
print(all_data.loc[(all_data['MasVnrType'] == 'None') & (all_data['MasVnrArea'] > 0),

            ['MasVnrArea', 'MasVnrType', 'Exterior1st', 'Exterior2nd']])



all_data.loc[(all_data['MasVnrType'] == 'None') & (all_data['MasVnrArea'] > 0), 'MasVnrArea'] = 0
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]



f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na, color='black')

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data, by feature', fontsize=15)

print(all_data['MSZoning'].count())

all_data['MSZoning'].value_counts().plot(kind='bar')
all_data.loc[all_data['MSZoning'].isna(), 'MSZoning'] = 'no_zone'
print(all_data['Alley'].count() - all_data['Utilities'].count())

all_data['Utilities'].value_counts().plot(kind='bar')
all_data.loc[all_data['Utilities'].isna(), 'Utilities'] = 'AllPub'
print(all_data['Alley'].count() - all_data['Functional'].count())

all_data['Functional'].value_counts().plot(kind='bar')
all_data.loc[all_data['Functional'].isna(), 'Functional'] = 'Typ'
all_data['BsmtFullBath'].hist()

all_data['BsmtHalfBath'].hist()
all_data.loc[all_data['BsmtHalfBath'].isna(), 'BsmtHalfBath'] = 0

all_data.loc[all_data['BsmtFullBath'].isna(), 'BsmtFullBath'] = 0
print(all_data['Alley'].count() - all_data['TotalBsmtSF'].count())

all_data['TotalBsmtSF'].hist()
all_data.loc[all_data['TotalBsmtSF'].isna(), 

             ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]
all_data.loc[all_data['TotalBsmtSF'].isna(), 

             ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']] = 0
print(all_data['Alley'].count() - all_data['SaleType'].count())

all_data['SaleType'].value_counts().plot(kind='bar')
all_data.loc[all_data['SaleType'].isna(), 'SaleType'] = 'WD'
print(all_data['Alley'].count() - all_data['KitchenQual'].count())

all_data['KitchenQual'].value_counts().plot(kind='bar')
all_data.loc[all_data['KitchenQual'].isna(), 'KitchenQual'] = 'TA'
print(all_data.loc[all_data['GarageCars'].isna(), ['GarageCars', 'GarageArea']])

all_data.loc[all_data['GarageCars'].isna(), 'GarageCars'] = 0
all_data.drop('GarageArea', axis=1, inplace=True)



print(all_data.shape)
all_data['Exterior1st'].value_counts().plot(kind='bar')

all_data['Exterior2nd'].value_counts().plot(kind='bar')
print(all_data.loc[(all_data['Exterior1st'].isna() | all_data['Exterior2nd'].isna()), ['Exterior1st','Exterior2nd','MsVnrType']])
all_data.loc[all_data['MasVnrType'].isna(), 'MasVnrType'] = 'None'
all_data.loc[all_data['Exterior1st'].isna(), ['Exterior1st', 'Exterior2nd']] = 'VinylSd'
print(all_data['Alley'].count() - all_data['Electrical'].count())

all_data['Electrical'].value_counts().plot(kind='bar')
all_data.loc[all_data['Electrical'].isna(), 'Electrical'] = 'SBrkr'
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na.describe()
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)
corrmat = all_data.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True, center = 0, cmap = 'viridis')
from sklearn.preprocessing import OrdinalEncoder





ord_cols = ['MSSubClass', 'OverallCond', 'OverallQual',

        'YrSold', 'Street', 'LotShape', 'LandContour',

           'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',

           'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional',

           'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond']



enc = OrdinalEncoder()

enc.fit(all_data[ord_cols])

all_data[ord_cols] = enc.transform(all_data[ord_cols])



print('Shape all_data: {}'.format(all_data.shape))

print(all_data[ord_cols].head())
all_data = pd.get_dummies(all_data)

print(all_data.shape)
train = all_data[:len(train)]

test = all_data[len(train):]



print(train.head())

print(train.shape)

print(test.head())

print(test.shape)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, StackingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

lasso_pipeline = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

lasso.fit(train.values, y_train)



y_pred_test = lasso.predict(test)

rf = RandomForestRegressor(random_state = 42)



params_rf = {'n_estimators': [400,500,600],

             'max_depth': [10,20,30],

    'max_features':['log2', 'auto', 'sqrt'],

    'min_samples_leaf':[2,5,10]

}



grid_rf = GridSearchCV(estimator=rf,

                       param_grid=params_rf,

                       scoring='neg_mean_squared_error',

                       cv=5,

                       verbose=1,

                       n_jobs=-1)



grid_rf.fit(train.values, y_train)



best_rf = grid_rf.best_estimator_



rf_score = rmsle_cv(best_rf)

print("\nRFR score: {:.4f} ({:.4f})\n".format(rf_score.mean(), rf_score.std()))



best_rf.get_params()
y_rf_pred = best_rf.predict(test)
estimators = [('lasso', lasso_pipeline),

    ('rfr', RandomForestRegressor(random_state=42))]



stacked_reg = StackingRegressor(estimators=estimators)

stacked_reg.fit(train.values, y_train)
y_stacked_test = stacked_reg.predict(test)
print(y_pred_test[1:5])

lasso_output_pred = np.exp(y_pred_test)

rf_output_pred = np.exp(y_rf_pred)

stacked_output_pred = np.exp(y_stacked_test)


id_col = test['Id']

output = list(zip(id_col,stacked_output_pred))

output_df = pd.DataFrame(output, columns = ['Id', 'SalePrice'])

output_df.head()
output_df.to_csv('submission.csv', index=False) 