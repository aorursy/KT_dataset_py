# Import libraries

import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn import model_selection

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, r2_score

from math import sqrt
data_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# checking if files have been loaded correctly

data_train.head()
data_train.shape, data_test.shape
data_train['SalePrice'].describe()
# columns which have null values

columns_having_nulls_train = data_train.columns[data_train.isna().any()].sort_values(ascending=False).to_series()



# count of null values in those column

values_train = data_train[columns_having_nulls_train].isna().sum().sort_values(ascending=False)



# percent of null values in those columns

percent_train = (data_train[columns_having_nulls_train].isna().sum()/data_train.shape[0]).sort_values(ascending=False)



# concat series and create a datframe to present the information cleanly

pd.concat([columns_having_nulls_train, values_train, percent_train], axis=1, sort=False, join='inner'

          , keys=['Feature', 'Null Count', 'Percent Null']).sort_values(by=['Null Count'], ascending=False).reset_index(drop=True)
columns_having_nulls_test = data_test.columns[data_test.isna().any()].sort_values(ascending=False).to_series()

values_test = data_test[columns_having_nulls_test].isna().sum().sort_values(ascending=False)

percent_test = (data_test[columns_having_nulls_test].isna().sum()/data_test.shape[0]).sort_values(ascending=False)

pd.concat([columns_having_nulls_test, values_test, percent_test], axis=1, sort=False, join='inner'

          , keys=['Feature', 'Null Count', 'Percent Null']).sort_values(by=['Null Count'], ascending=False).reset_index(drop=True)
#data_train.describe().T
plt.figure(figsize=(10,5))

g = sns.scatterplot(x="LotArea", y="SalePrice", hue="GrLivArea", data=data_train, palette="Set2")
# Let's create a new variable Total Floor Surface Area (1st + 2nd) and see how total surface are is related with SalePrice

data_train['TotalSF'] = data_train['1stFlrSF'] + data_train['2ndFlrSF']

plt.figure(figsize=(10,5))

g = sns.scatterplot(x="TotalSF", y="SalePrice", hue="OverallQual", data=data_train, palette="Set2")
g = sns.catplot(x="TotRmsAbvGrd", y="SalePrice", kind="box", data=data_train, aspect=2, height=4, palette="Set2")
g = sns.catplot(x="OverallQual", y="SalePrice", kind="box", data=data_train, aspect=1.6, height=4, palette="Set2")
g = sns.catplot(x="OverallCond", y="SalePrice", kind="box", data=data_train, aspect=1.6, height=4, palette="Set2")
data_train['TotalBaths'] = data_train['FullBath'] + data_train['HalfBath']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize = (15, 5))

sns.boxplot(x="HalfBath", y="SalePrice", data=data_train, ax=ax1)

sns.boxplot(x="FullBath", y="SalePrice", data=data_train, ax=ax2)

sns.boxplot(x="TotalBaths", y="SalePrice", data=data_train, ax=ax3)
sns.catplot(x="Utilities", y="SalePrice", kind="violin", data=data_train, aspect=1, height=4)
data_train[data_train['Utilities']=='NoSeWa'].shape[0]
data_test['Utilities'].value_counts()
g = sns.catplot(x="YearBuilt", y="SalePrice", kind="box", data=data_train, aspect=4, height=4)
# Age before selling

data_train['AgeAtSale'] = data_train['YrSold'] - data_train['YearBuilt'] + data_train['MoSold']/12

data_train['AgeAtSale'] = round(data_train['AgeAtSale'])

data_train['AgeAtSale'] = pd.qcut(data_train['AgeAtSale'], 4)

g = sns.catplot(x="AgeAtSale", y="SalePrice", kind="violin", data=data_train, aspect=4, height=4)
sns.catplot(x="MSZoning", y="SalePrice", kind="box", data=data_train, aspect=2, height=4, palette="Set2")
sns.catplot(x="MSZoning", y="TotalSF", kind="box", data=data_train, aspect=2, height=4, palette="Set2")
data_test['MSZoning'].unique()
sns.catplot(x="LandContour", y="SalePrice", kind="box", data=data_train, aspect=2, height=4, palette="Set2")
sns.catplot(x="ExterQual", y="SalePrice", kind="violin", data=data_train, aspect=2, height=4, palette="Set2")
sns.catplot(x="BsmtQual", y="SalePrice", kind="violin", data=data_train, aspect=2, height=4, palette="Set2")
plt.figure(figsize=(10,5))

g = sns.regplot(x="SalePrice", y="GarageArea", data=data_train)
data_train['SaleType_updated'] = data_train['SaleType']

data_train.loc[((data_train['SaleType'] == 'WD') | (data_train['SaleType'] == 'CWD') | (data_train['SaleType'] == 'VWD')), 'SaleType_updated'] = 'WD'

data_train.loc[((data_train['SaleType'] == 'Con') | (data_train['SaleType'] == 'ConLD') | (data_train['SaleType'] == 'ConLI') | (data_train['SaleType'] == 'ConLw')), 'SaleType_updated'] = 'Con'

sns.catplot(x="SaleType_updated", y="SalePrice", kind="box", data=data_train, aspect=1.5, height=4, palette="Set2")
data_train.loc[(data_train['SaleType'] != 'New'), 'SaleType_updated'] = 'Other'

sns.catplot(x="SaleType_updated", y="SalePrice", kind="violin", data=data_train, aspect=1.5, height=4, palette="Set2")
sns.catplot(x="CentralAir", y="SalePrice", kind="violin", data=data_train, aspect=1, height=4, palette="Set2")
corrmat = data_train.loc[:,~data_train.columns.isin(['TotalSF','TotalBaths'])].corr()

f, ax = plt.subplots(figsize=(15, 12))

sns.heatmap(corrmat, vmax=.8, square=True);
corrmat = data_train[['TotalSF','TotalBaths','TotalBsmtSF','1stFlrSF','FullBath','GarageCars','GarageArea','TotRmsAbvGrd','GrLivArea','SalePrice']].corr()

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(corrmat, square=True, annot=True);
cols_to_remove1 = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage']  #cols with lot of nulls

cols_to_remove2 = ['GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea', 'TotalBsmtSF'] #correlated with other feature

cols_to_remove3 = ['Utilities'] #not contributing to prediction

cols_to_remove4 = ['OverallCond', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 

                   'HalfBath', 'BedroomAbvGrd', 'KitchenAbvGrd', 'WoodDeckSF', 'OpenPorchSF', 'EncliosedPorch', '3SsnPorch', 

                   'ScreenPorch', 'PoolArea', 'MiscVal']

# 'MSSubclass', 'MoSold', 'YrSold' -- on hold
# from our understanding of data

categorical_cols = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',

                   'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','RoofStyle',

                   'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',

                    'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical',

                   'KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

                   'PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition',

                    'MoSold']
# so what remains are the numerical columns

numerical_cols = list(set(data_train.columns.tolist()) - set(categorical_cols))
data_train = data_train[~((data_train['TotalSF'] > 4000) & (data_train['SalePrice'] < 300000))]
data_train.shape
data_train['GarageType'].fillna(value=data_train['GarageType'].mode()[0], inplace=True)

data_train['BsmtQual'].fillna(value=data_train['BsmtQual'].mode()[0], inplace=True)

data_train['Electrical'].fillna(value=data_train['Electrical'].mode()[0], inplace=True)

data_train['GarageQual'].fillna(value=data_train['GarageQual'].mode()[0], inplace=True)

data_train['MasVnrArea'].fillna(value=data_train['MasVnrArea'].mean(), inplace=True)
data_test['KitchenQual'].fillna(data_test['KitchenQual'].mode()[0], inplace=True)

data_test['GarageType'].fillna(data_test['GarageType'].mode()[0], inplace=True)

data_test['MSZoning'].fillna(data_test['MSZoning'].mode()[0], inplace=True)

data_test['BsmtQual'].fillna(data_test['BsmtQual'].mode()[0], inplace=True)

data_test['SaleType'].fillna(data_test['SaleType'].mode()[0], inplace=True)

data_test['GarageCars'].fillna(data_test['GarageCars'].mode()[0], inplace=True)

data_test['GarageQual'].fillna(data_test['GarageQual'].mode()[0], inplace=True)

data_test['MasVnrArea'].fillna(data_test['MasVnrArea'].mean(), inplace=True)

data_test['BsmtFinSF1'].fillna(data_test['BsmtFinSF1'].mean(), inplace=True)
data_train[['GarageType', 'MasVnrArea', 'BsmtQual', 'Electrical', 'GarageQual']].isna().any().sum()
data_test[['KitchenQual', 'GarageType', 'MSZoning', 'MasVnrArea', 'BsmtQual', 'SaleType', 'BsmtFinSF1', 'GarageCars', 'GarageQual']].isna().any().sum()
# data_train['AgeAtSale'] - Already created

data_train['RemodAgeAtSale'] = data_train['YrSold'] - data_train['YearRemodAdd']

data_train['RemodAgeAtSale'] = round(data_train['RemodAgeAtSale'])

data_train['RemodAgeAtSale'] = pd.cut(data_train['RemodAgeAtSale'], 3)
data_train['IsMultiStory'] = data_train['2ndFlrSF'].apply(lambda x: 0 if x <= 0 else 1)
data_test['AgeAtSale'] = data_test['YrSold'] - data_test['YearBuilt'] + data_test['MoSold']/12

data_test['AgeAtSale'] = round(data_test['AgeAtSale'])

data_test['AgeAtSale'] = pd.qcut(data_test['AgeAtSale'], 4)



data_test['TotalBaths'] = data_test['FullBath'] + data_test['HalfBath']



data_test['RemodAgeAtSale'] = data_test['YrSold'] - data_test['YearRemodAdd']

data_test['RemodAgeAtSale'] = round(data_test['RemodAgeAtSale'])

data_test['RemodAgeAtSale'] = pd.cut(data_test['RemodAgeAtSale'], 3)



data_test['IsMultiStory'] = data_test['2ndFlrSF'].apply(lambda x: 0 if x <= 0 else 1)
# 'Id' is kept so that we can create submission files at the end

num_cols_to_keep = ['Id','1stFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'EnclosedPorch', 'Fireplaces', 'TotalBaths', 'FullBath', 'GarageCars', 'GrLivArea', 'KitchenAbvGr', 'LotArea', 'MasVnrArea', 'SalePrice']

cat_cols_to_keep = ['BsmtQual', 'CentralAir', 'Electrical', 'ExterQual', 'GarageQual', 'GarageType', 'Heating', 'HouseStyle', 'KitchenQual', 'MSZoning', 'OverallQual', 'SaleType']

new_cols_we_created = ['IsMultiStory','AgeAtSale','RemodAgeAtSale']

cols_to_keep = list(set(num_cols_to_keep)|set(cat_cols_to_keep)|set(new_cols_we_created))
data_train = data_train[cols_to_keep]

data_train.shape
data_test = data_test[list(set(cols_to_keep) - {'SalePrice'})]

data_test.shape
data_train.loc[(data_train['Electrical'] != 'SBrkr'), 'Electrical'] = 0

data_train.loc[(data_train['Electrical'] == 'SBrkr'), 'Electrical'] = 1



data_train.loc[((data_train['KitchenQual'] == 'Fa') | (data_train['KitchenQual'] == 'TA')), 'KitchenQual'] = 'Fa'



data_train.loc[(data_train['SaleType'] != 'New'), 'SaleType'] = 0

data_train.loc[(data_train['SaleType'] == 'New'), 'SaleType'] = 1



data_train.loc[((data_train['GarageQual'] == 'Fa') | (data_train['GarageQual'] == 'Po')), 'GarageQual'] = 0

data_train.loc[(data_train['GarageQual'] != 'Poor'), 'GarageQual'] = 1



data_train.loc[((data_train['GarageType'] == 'Attchd') | (data_train['GarageType'] == 'BuiltIn') | (data_train['GarageType'] == 'Basment') | (data_train['GarageType'] == '2Types') ), 'GarageType'] = 1

data_train.loc[((data_train['GarageType'] == 'Detchd') | (data_train['GarageType'] == 'CarPort') ), 'GarageType'] = 0



data_train.loc[((data_train['Heating'] == 'Floor') | (data_train['Heating'] == 'Wall') | (data_train['Heating'] == 'Grav') | (data_train['Heating'] == 'OthW') ), 'Heating'] = 0

data_train.loc[((data_train['Heating'] == 'GasA') | (data_train['Heating'] == 'GasW') ), 'Heating'] = 1



data_train.loc[((data_train['HouseStyle'] == '1.5Unf') | (data_train['HouseStyle'] == '2.5Unf') ), 'HouseStyle'] = 0

data_train.loc[(data_train['HouseStyle'] != 'Unfinished'), 'HouseStyle'] = 1
data_test.loc[(data_test['Electrical'] != 'SBrkr'), 'Electrical'] = 0

data_test.loc[(data_test['Electrical'] == 'SBrkr'), 'Electrical'] = 1



data_test.loc[((data_test['KitchenQual'] == 'Fa') | (data_test['KitchenQual'] == 'TA')), 'KitchenQual'] = 'Fa'



data_test.loc[(data_test['SaleType'] != 'New'), 'SaleType'] = 0

data_test.loc[(data_test['SaleType'] == 'New'), 'SaleType'] = 1



data_test.loc[((data_test['GarageQual'] == 'Fa') | (data_test['GarageQual'] == 'Po')), 'GarageQual'] = 0

data_test.loc[(data_test['GarageQual'] != 'Poor'), 'GarageQual'] = 1



data_test.loc[((data_test['GarageType'] == 'Attchd') | (data_test['GarageType'] == 'BuiltIn') | (data_test['GarageType'] == 'Basment') | (data_test['GarageType'] == '2Types') ), 'GarageType'] = 1

data_test.loc[((data_test['GarageType'] == 'Detchd') | (data_test['GarageType'] == 'CarPort') ), 'GarageType'] = 0



data_test.loc[((data_test['Heating'] == 'Floor') | (data_test['Heating'] == 'Wall') | (data_test['Heating'] == 'Grav') | (data_test['Heating'] == 'OthW') ), 'Heating'] = 0

data_test.loc[((data_test['Heating'] == 'GasA') | (data_test['Heating'] == 'GasW') ), 'Heating'] = 1



data_test.loc[((data_test['HouseStyle'] == '1.5Unf') | (data_test['HouseStyle'] == '2.5Unf') ), 'HouseStyle'] = 0

data_test.loc[(data_test['HouseStyle'] != 'Unfinished'), 'HouseStyle'] = 1
data_train['BsmtQual'].replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)

data_train['ExterQual'].replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)

data_train['KitchenQual'].replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)

data_train['CentralAir'].replace({'Y': 1, 'N': 2}, inplace=True)

#---

data_test['BsmtQual'].replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)

data_test['ExterQual'].replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)

data_test['KitchenQual'].replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)

data_test['CentralAir'].replace({'Y': 1, 'N': 2}, inplace=True)
data_train = pd.get_dummies(data_train, columns=['MSZoning','AgeAtSale','RemodAgeAtSale'])

data_test = pd.get_dummies(data_test, columns=['MSZoning','AgeAtSale','RemodAgeAtSale'])
data_train.columns
data_train[num_cols_to_keep].describe()
cols_to_standardize = ['1stFlrSF', 'BsmtFinSF1', 'EnclosedPorch', 'GrLivArea', 'LotArea', 'MasVnrArea']

col_scalar = MinMaxScaler()

for col in cols_to_standardize:

    data_train[col] = col_scalar.fit_transform(data_train[col].values.reshape(-1,1))

    data_test[col] = col_scalar.transform(data_test[col].values.reshape(-1,1))
data_train[cols_to_standardize].head()
data_test.shape
cv_splits = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
kwargs = {

    'LinearRegression':{'fit_intercept': True},

    'Ridge': {'alpha': 1,

              'max_iter': 1000,

              'solver': 'auto',

              'random_state': 42},

    'Lasso': {'alpha': 1,

              'max_iter': 1000,

              #'precompute': 'auto',

              'warm_start': False,

              'selection':'cyclic',

              'random_state': 42},

    'KNeighborsRegressor': {'n_neighbors': 10,

                            'algorithm': 'auto',

                           'leaf_size': 10,

                           'p': 2},

    'SVR': {'kernel': 'poly',

            'degree':5,

            'C':10.0, 

            'epsilon':0.01, 

            'shrinking':True},

    'GradientBoostingRegressor': {'loss':'huber',

                                  'learning_rate':0.1,

                                  'n_estimators':100,

                                  'subsample': 1, 

                                  'max_depth':3, 

                                  'random_state':42, 

                                  'max_features':None, 

                                  'max_leaf_nodes':None, 

                                  'validation_fraction':0.1},

    'MLPRegressor': {'hidden_layer_sizes':(64,32), 

                     'activation':'relu', 

                     'solver':'lbfgs', 

                     'alpha':0.025, 

                     'batch_size':'auto', 

                     'learning_rate':'adaptive', 

                     'learning_rate_init':0.001, 

                     'max_iter':500, 

                     'shuffle':True, 

                     'random_state':42, 

                     'momentum':0.8, 

                     'beta_1':0.9, 'beta_2':0.999, 

                     'epsilon':1e-08}

}
algos = {

    'LinearRegression':LinearRegression(**kwargs['LinearRegression']),

    'Ridge':Ridge(**kwargs['Ridge']),

    'Lasso':Lasso(**kwargs['Lasso']),

    'KNeighborsRegressor':KNeighborsRegressor(**kwargs['KNeighborsRegressor']),

    'SVR':SVR(**kwargs['SVR']),

    'GradientBoostingRegressor':GradientBoostingRegressor(**kwargs['GradientBoostingRegressor']),

    'MLPRegressor':MLPRegressor(**kwargs['MLPRegressor'])

}
cv_results = {'Algorithm':[],                     # algorithm name

              'Mean Train MSE':[],                # Mean of training accuracy on all splits

              'Mean Test MSE':[],                 # Mean of test accuracy on all splits

              'Mean Train R2':[],

              'Mean Test R2':[],

              'Fit Time': []}                     # how fast the algorithm converges
for alg_name,alg in algos.items():

    cv_results['Algorithm'].append(alg_name)

    

    cross_val = model_selection.cross_validate(alg, 

                                               data_train.loc[:, ~data_train.columns.isin(['Id','SalePrice'])], 

                                               data_train['SalePrice'],

                                               scoring = ['neg_mean_squared_error','r2'],

                                               cv  = cv_splits,

                                               return_train_score=True,

                                               return_estimator=False

                                              )

    

    cv_results['Mean Train MSE'].append(cross_val['train_neg_mean_squared_error'].mean())

    cv_results['Mean Test MSE'].append(cross_val['test_neg_mean_squared_error'].mean())

    cv_results['Mean Train R2'].append(cross_val['train_r2'].mean())

    cv_results['Mean Test R2'].append(cross_val['test_r2'].mean())

    cv_results['Fit Time'].append(cross_val['fit_time'].mean())

    
cv_results_df = pd.DataFrame.from_dict(cv_results)

cv_results_df.sort_values(by=['Mean Test R2'], inplace=True, ascending=False)

cv_results_df
# store the predictions in a dictionary

y_predicted = {}
for alg_name,alg in algos.items():

    

    alg.fit(data_train.loc[:, ~data_train.columns.isin(['Id','SalePrice'])], data_train['SalePrice'])

    y_predicted[alg_name] = alg.predict(data_test.loc[:, ~data_test.columns.isin(['Id'])])

    
# create a dataframe and write to a csv file

for alg_name in algos.keys():

    results_dict = {'Id':data_test['Id'].values.tolist(), 'SalePrice':list(y_predicted[alg_name])}

    results_df = pd.DataFrame.from_dict(results_dict)

    #results_df.to_csv(alg_name+'.csv', index=False)