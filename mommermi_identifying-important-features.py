import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/train.csv')



# replace prices with logarithmic prices

data['logSalePrice'] = np.log10(data.SalePrice.values)

data.drop('SalePrice', axis=1, inplace=True)
# automatically identify continuous features

continuous_features = [col for col in data.columns if data[col].dtype != 'O']



continuous_features.remove('Id')

continuous_features.remove('logSalePrice') # remove the target feature



# manually select ordinal features that can be ranked in some way

ordinal_features = ['Street', 'Alley', 'LotShape', 'Utilities', 'LandSlope', 'ExterQual', 

                    'LandContour', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 

                    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 

                    'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',

                    'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 

                    'Fence'] 



# manually select categorical features that will be ranked based on median `logSalePrice`

categorical_features = ['LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 

                        'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 

                        'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'MiscFeature',

                        'SaleType', 'SaleCondition', 'MSZoning', 'BldgType']
for col in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 

            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 

            'PoolQC', 'Fence', 'MiscFeature']:

    data.loc[:, col].fillna('NA', inplace=True) 
data.fillna({col: data.loc[:, col].median() for col in continuous_features}, inplace=True)

data.fillna({col: data.loc[:, col].value_counts().index[0] for 

             col in categorical_features + ordinal_features}, inplace=True)
ordinal_transform = {'Street':  {'Grvl': 1, 'Pave': 2}, 

                     'Alley': {'NA': 0, 'Grvl': 1, 'Pave': 2}, 

                     'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}, 

                     'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}, 

                     'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3}, 

                     'LandContour': {'Low': 1, 'HLS': 1, 'Bnk': 2, 'Lvl': 3}, 

                     'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                     'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, 

                     'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 

                                      'GLQ': 6}, 

                     'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 

                                      'GLQ': 6}, 

                     'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                     'CentralAir': {'N': 1, 'Y': 2}, 

                     'Electrical': {'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}, 

                     'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 

                                    'Min1': 7, 'Typ': 8}, 

                     'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'GarageType': {'NA': 0, 'Detchd': 1, 'CarPort': 2, 'BuiltIn': 3, 

                                    'Basment': 4, 'Attchd': 5, '2Types': 6},

                     'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, 

                     'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'PavedDrive': {'N': 1, 'P': 2, 'Y': 3}, 

                     'PoolQC': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'Fence': {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}, 

                    }



# apply transformations

for col in ordinal_features:

    data.loc[:, col] = data.loc[:, col].map(ordinal_transform[col], na_action='ignore')

    

# move some features from continuous to ordinal feature list

for col in ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

            'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']:

    continuous_features.remove(col)

    ordinal_features.append(col)

    

# move one feature from continuous to categorial feature list

continuous_features.remove('MSSubClass')

categorical_features.append('MSSubClass')
ordinalized = []

for col in categorical_features:

    sorted_labels = data[[col, 'logSalePrice']].groupby(col).logSalePrice.median().sort_values()

    data.loc[:, col] = data.loc[:, col].map({sorted_labels.index.values[i]:i for i in range(len(sorted_labels))})

    ordinalized.append(col)



for col in ordinalized:

    categorical_features.remove(col)

    ordinal_features.append(col)
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharey=True)

plt.subplots_adjust(hspace=0.5)



# continuous features

data[continuous_features].corrwith(data.logSalePrice).agg('square').plot.bar(ax=ax1, alpha=0.5)

ax1.set_title('Coefficient of Determination: Continuous Features')

ax1.grid()



# ordinal features

data[ordinal_features].corrwith(data.logSalePrice).agg('square').plot.bar(ax=ax2, alpha=0.5)

ax2.set_title('Coefficient of Determination: Ordinal Features')

ax2.grid()

results = pd.DataFrame(data.drop('logSalePrice', axis=1).corrwith(data.logSalePrice).agg('square'), 

                       columns=['det_weight'])



ranks = np.zeros(len(results), dtype=np.int)

for i, j in enumerate(np.argsort(results.det_weight)[::-1]):

    ranks[j] = i

results['det_rank'] = ranks



results.sort_values('det_rank').loc[:, ['det_rank', 'det_weight']].iloc[0:10]
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



pipe = Pipeline([('model', RandomForestRegressor(n_jobs=-1, random_state=42))])



param_space = {'model__max_depth': [10, 15, 20],

               'model__max_features': [10, 15, 20],

               'model__n_estimators': [250, 300, 350]}



grid = GridSearchCV(pipe, param_grid=param_space, cv=10, scoring='neg_mean_squared_error')



grid.fit(data.drop('logSalePrice', axis=1), data.logSalePrice)
print('best-fit parameters:', grid.best_params_)

print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))
weights = grid.best_estimator_.steps[0][1].feature_importances_



results['rf_weight'] = weights



ranks = np.zeros(len(results), dtype=np.int)

for i, j in enumerate(np.argsort(weights)[::-1]):

    ranks[j] = i

results['rf_rank'] = ranks



results.sort_values('rf_rank').loc[:, ['rf_rank', 'rf_weight']].iloc[0:10]
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import Ridge



pipe = Pipeline([('scaler', RobustScaler()), 

                 ('model', Ridge(random_state=42))])



param_space = {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]}



grid = GridSearchCV(pipe, param_grid=param_space, cv=10, scoring='neg_mean_squared_error')



grid.fit(data.drop('logSalePrice', axis=1), data.logSalePrice)
print('best-fit parameters:', grid.best_params_)

print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))
weights = grid.best_estimator_.steps[1][1].coef_



results['ridge_weight'] = weights



ranks = np.zeros(len(results), dtype=np.int)

for i, j in enumerate(np.argsort(weights)[::-1]):

    ranks[j] = i

results['ridge_rank'] = ranks



results.sort_values('ridge_rank').loc[:, ['ridge_rank', 'ridge_weight']].iloc[0:10]
from sklearn.linear_model import Lasso



pipe = Pipeline([('scaler', RobustScaler()), 

                 ('model', Lasso(random_state=42))])



param_space = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid = GridSearchCV(pipe, param_grid=param_space, cv=10, scoring='neg_mean_squared_error')



grid.fit(data.drop('logSalePrice', axis=1), data.logSalePrice)
print('best-fit parameters:', grid.best_params_)

print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))
weights = grid.best_estimator_.steps[1][1].coef_



results['lasso_weight'] = weights



ranks = np.zeros(len(results), dtype=np.int)

for i, j in enumerate(np.argsort(weights)[::-1]):

    ranks[j] = i

results['lasso_rank'] = ranks



results.sort_values('lasso_rank').loc[:, ['lasso_rank', 'lasso_weight']].iloc[0:10]
f, ax = plt.subplots(figsize=(17, 5))



results['mean_rank'] = results.loc[:, ['det_rank', 'rf_rank', 'ridge_rank', 'lasso_rank']].agg('abs').mean(axis=1)

results['mean_rank_std'] = results.loc[:, ['det_rank', 'rf_rank', 'ridge_rank', 'lasso_rank']].std(axis=1)



results.mean_rank.sort_values().plot.bar(width=0.5, color='orange', alpha=0.7, ax=ax)

results.sort_values(by='mean_rank').mean_rank_std.plot.bar(width=0.5, color='black', alpha=0.3, ax=ax)



ax.set_xlim([0,79.5])

ax.plot(ax.get_xlim(), [0, 80], color='red')

ax.set_ylabel('Average Rank')
from matplotlib import ticker



f, ax = plt.subplots(figsize=(5, 17))



for idx, feature in enumerate(results.sort_values(by='mean_rank').index):

    ax.plot(range(4), results.loc[feature, ['det_rank', 'rf_rank', 'ridge_rank', 'lasso_rank']]+0.5, marker='o',

           alpha=np.clip(1/(results.loc[feature, 'mean_rank_std']+0.001), 0.01, 1),

           linewidth=np.clip(1/(results.loc[feature, 'mean_rank_std']+0.1), 0.5, 1)*3)

ax.set_ylim([80, 0])

ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=80))

ax.set_yticklabels(results.sort_values(by='mean_rank').index)

ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=4))

ax.set_xticklabels(['det_rank', 'rf_rank', 'ridge_rank', 'lasso_rank'])

for col in ['det_weight', 'rf_weight', 'ridge_weight', 'lasso_weight']:

    weightsum = results.loc[:, col].abs().sum(axis=0)

    results.loc[:, col] = results.loc[:, col].apply(lambda x: np.abs(x)/weightsum)
f, ax = plt.subplots(figsize=(17, 5))

ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=80))

results.sort_values('mean_rank').loc[:, ['det_weight', 'rf_weight', 'ridge_weight', 'lasso_weight']].cumsum().plot.line(

    rot='vertical', ax=ax)

ax.grid()
pipe = Pipeline([('scaler', RobustScaler()), 

                 ('model', Ridge(random_state=42))])



param_space = {'model__alpha': [1, 5, 10, 50, 100]}



grid = GridSearchCV(pipe, param_grid=param_space, cv=10, scoring='neg_mean_squared_error')



grid.fit(data.drop('logSalePrice', axis=1).loc[:, ['OverallQual', 'Neighborhood', 'GrLivArea']], data.logSalePrice)
print('best-fit parameters:', grid.best_params_)

print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))
pipe = Pipeline([('model', RandomForestRegressor(n_jobs=-1, random_state=42))])



param_space = {'model__max_depth': [5, 7, 10],

               'model__max_features': [1, 2],

               'model__n_estimators': [70, 100, 120]}



grid = GridSearchCV(pipe, param_grid=param_space, cv=5, scoring='neg_mean_squared_error')



grid.fit(data.drop('logSalePrice', axis=1).loc[:, ['OverallQual', 'Neighborhood', 'GrLivArea']], data.logSalePrice)
print('best-fit parameters:', grid.best_params_)

print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))
pipe = Pipeline([('model', RandomForestRegressor(n_jobs=-1, random_state=42))])



param_space = {'model__max_depth': [10, 15, 20, 25],

               'model__max_features': [2, 4, 6],

               'model__n_estimators': [100, 150, 200, 250, 300]}



grid = GridSearchCV(pipe, param_grid=param_space, cv=5, scoring='neg_mean_squared_error')



grid.fit(data.drop('logSalePrice', axis=1).loc[:, results.sort_values('mean_rank').index[:15]], data.logSalePrice)
print('best-fit parameters:', grid.best_params_)

print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))