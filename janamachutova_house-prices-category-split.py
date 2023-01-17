import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder



%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.SalePrice.min()
test['SalePrice'] = 0

full = train.append(test)
ident_cols = ['Id', 'SalePrice']

sale_prop_cols = ['Id', 'YrSold', 'MoSold', 'SaleType', 'SaleCondition']

dates_cols = ['Id', 'YrSold', 'MoSold', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd']

construction_cols = ['Id', 'MSSubClass', 'Utilities', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'BsmtExposure', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'MiscFeature', 'MiscVal']

disposition_cols = ['Id', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

measure_cols = ['Id', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']

landscape_cols = ['Id', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2']

qualities_cols = ['Id', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
ident = full[ident_cols].copy()

ident.to_csv('ident.csv', index=False)
qualities = full[qualities_cols].copy()

qualities.head()
def combine_features(data, name1, name2, new_name = None):

    col_name = new_name

    if col_name == None:

        col_name = name1 + '_' + name2

    data[col_name] = data[name1].map(str) + "_" + data[name2].map(str)

    data.drop([name1, name2], axis=1, inplace=True)

    return data
for c in qualities.columns[qualities.isna().any()].tolist():

    print(c)
tmp = pd.merge(qualities[['BsmtQual', 'BsmtCond', 'Id']], full[['TotalBsmtSF', 'Id']], on=['Id'])

ta_rows = tmp[(tmp.isnull().any(axis=1)) & (tmp.TotalBsmtSF > 0.0)].Id

print(ta_rows.tolist())

qualities['BsmtQual'] = qualities.apply(lambda x : 'TA' if ta_rows.isin([x['Id']]).any() and (pd.isnull(x['BsmtQual'])) else x['BsmtQual'], axis=1)

qualities['BsmtCond'] = qualities.apply(lambda x : 'TA' if ta_rows.isin([x['Id']]).any() and (pd.isnull(x['BsmtCond'])) else x['BsmtCond'], axis=1)



tmp = pd.merge(qualities[['BsmtFinType1', 'BsmtFinType2', 'Id']], full[['TotalBsmtSF', 'Id']], on=['Id'])

ta_rows = tmp[(tmp.isnull().any(axis=1)) & (tmp.TotalBsmtSF > 0.0)].Id

print(ta_rows.tolist())

qualities['BsmtFinType1'] = qualities.apply(lambda x : 'Rec' if ta_rows.isin([x['Id']]).any() and (pd.isnull(x['BsmtFinType1'])) else x['BsmtFinType1'], axis=1)

qualities['BsmtFinType2'] = qualities.apply(lambda x : 'Rec' if ta_rows.isin([x['Id']]).any() and (pd.isnull(x['BsmtFinType2'])) else x['BsmtFinType2'], axis=1)
tmp = pd.merge(qualities[['GarageFinish', 'GarageQual', 'GarageCond', 'Id']], full[['GarageArea', 'GarageCars', 'Id']], on=['Id'])

miss_rows = tmp[(tmp.isnull().any(axis=1)) & (tmp.GarageArea > 0)].Id

print(tmp[(tmp.isnull().any(axis=1)) & (tmp.GarageArea > 0)])
qualities.GarageFinish = qualities.apply(lambda x : 'RFn' if x['Id'] == 2127 else x['GarageFinish'], axis=1)

qualities.GarageQual = qualities.apply(lambda x : 'TA' if x['Id'] == 2127 else x['GarageQual'], axis=1)

qualities.GarageCond = qualities.apply(lambda x : 'TA' if x['Id'] == 2127 else x['GarageCond'], axis=1)
tmp = pd.merge(qualities[['PoolQC', 'Id']], full[['PoolArea', 'Id']], on=['Id'])

miss_rows = tmp[(tmp.isnull().any(axis=1)) & (tmp.PoolArea > 0)].Id

print(miss_rows.tolist())

qualities['PoolQC'] = qualities.apply(lambda x : 'TA' if miss_rows.isin([x['Id']]).any() and (pd.isnull(x['PoolQC'])) else x['PoolQC'], axis=1)
na_grades_cols = ['Fence','PoolQC','BsmtFinType1','BsmtFinType2','GarageFinish',

                    'ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',

                    'KitchenQual','FireplaceQu','GarageQual','GarageCond']

for c in na_grades_cols:

    qualities[c] = qualities[c].map(lambda x : 'NA' if pd.isnull(x) else x)



qualities.Functional = qualities.Functional.map(lambda x : 'Typ' if pd.isnull(x) else x)
qual_grades_cols = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond']

qual_grades = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']

for c in qual_grades_cols:

    qualities[c] = qualities[c].map(lambda x : 0 if pd.isnull(x) else qual_grades.index(x))
#PoolQC

pool_grades = ['NA', 'Fa', 'TA', 'Gd', 'Ex']

#BsmtFinType1 BsmtFinType2

fin_types = ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'] 

#Functional

func_types = ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']

#Garage Finish

gar_fin = ['NA', 'Unf', 'RFn', 'Fin']

#Fence

fence_qual = ['NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
qualities.PoolQC = qualities.PoolQC.map(lambda x : 0 if pd.isnull(x) else pool_grades.index(x))

qualities.BsmtFinType1 = qualities.BsmtFinType1.map(lambda x : 0 if pd.isnull(x) else fin_types.index(x))

qualities.BsmtFinType2 = qualities.BsmtFinType2.map(lambda x : 0 if pd.isnull(x) else fin_types.index(x))

qualities.Functional = qualities.Functional.map(lambda x : 0 if pd.isnull(x) else func_types.index(x))

qualities.GarageFinish = qualities.GarageFinish.map(lambda x : 0 if pd.isnull(x) else gar_fin.index(x))

qualities.Fence = qualities.Fence.map(lambda x : 0 if pd.isnull(x) else fence_qual.index(x))
qualities['OverallCond_Qual'] = qualities.OverallQual + qualities.OverallCond

qualities['BsmtCond_Qual'] = qualities.BsmtQual + qualities.BsmtCond

qualities['ExterCond_Qual'] = qualities.ExterQual + qualities.ExterCond

qualities.OverallCond_Qual = qualities.OverallCond_Qual + qualities.ExterCond_Qual

qualities['GarageCond_Qual'] = qualities.GarageQual + qualities.GarageCond

qualities['GarageCond_Qual'] = qualities.GarageCond_Qual + qualities.GarageFinish

qualities.drop(['ExterCond_Qual','GarageFinish','OverallQual', 'OverallCond', 'ExterQual','ExterCond','BsmtQual','BsmtCond','GarageQual','GarageCond'], axis=1, inplace=True)
qualities
fin_types = qualities['BsmtFinType1'].value_counts().index.tolist()

qualities.BsmtFinType1 = qualities.BsmtFinType1.map(lambda x : fin_types.index(x) + 1)

qualities.BsmtFinType2 = qualities.BsmtFinType2.map(lambda x : fin_types.index(x) + 1)
qualities
fig, axes = plt.subplots(ncols=2, nrows=6, figsize=(15,40))

for i, idx in zip(qualities.drop('Id', axis=1).columns, range(0,len(qualities.columns)-1)):

    sns.countplot(qualities[i], label=i, ax=axes.flat[idx], )
qualities.describe()
qualities.info()
qualities.to_csv('qualities.csv', index=False)
sale_prop = full[sale_prop_cols].copy()

sale_prop.head()
print('first sale {}, last sale {}'.format(sale_prop.YrSold.min(),sale_prop.YrSold.max()))
label_encoder = LabelEncoder()

sale_prop.SaleType = sale_prop.SaleType.map(lambda x : str(x)) 

sale_prop.SaleType = label_encoder.fit_transform(sale_prop.SaleType)

sale_prop.SaleCondition = sale_prop.SaleCondition.map(lambda x : str(x)) 

sale_prop.SaleCondition = label_encoder.fit_transform(sale_prop.SaleCondition)
sale_prop['SoldAge'] = 2010 - sale_prop.YrSold

sale_prop.drop('YrSold', axis=1, inplace=True)

sale_prop.head()
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,10))

for i, idx in zip(sale_prop.drop('Id', axis=1).columns, range(0,len(sale_prop.columns)-1)):

    sns.countplot(sale_prop[i], label=i, ax=axes.flat[idx])
sale_prop.describe()
sale_prop.info()
sale_prop.to_csv('sale_prop.csv', index=False)
dates = full[dates_cols].copy()

dates.head()
print('built {}-{}, garage {}-{}'.format(dates.YearBuilt.min(), dates.YearBuilt.max(), dates.GarageYrBlt.min(), dates.GarageYrBlt.max()))

sns.heatmap(dates.isnull())
dates[dates.GarageYrBlt < dates.YearBuilt]
dates[dates.GarageYrBlt > 2010]
dates.GarageYrBlt = dates.GarageYrBlt.map(lambda x : 2007 if x > 2010 else x)

dates[dates.GarageYrBlt > 2010]
full[full.YearRemodAdd == 1950][['YearRemodAdd', 'YearBuilt']]
years_cols = ['YrSold', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd']

for c in years_cols:

    col_name = c + 'Age'

    dates[col_name] = 2010 - dates[c]

    dates[col_name] = dates[col_name].map(lambda x : 200 if pd.isnull(x) else int(x))

    dates.drop(c, axis=1, inplace=True)
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(15,12))

for i, idx in zip(dates.drop('Id', axis=1).columns, range(0,len(dates.columns)-1)):

    sns.countplot(dates[i], label=i, ax=axes.flat[idx])
dates.describe()
dates.info()
dates.to_csv('dates.csv', index=False)
constructions = full[construction_cols].copy()

constructions.head()
constructions['HasDoubleExt'] = constructions.apply(lambda x : 0 if x['Exterior1st'] == x['Exterior2nd'] else 1, axis=1)
constructions['IsUnfinished'] = constructions.HouseStyle.map(

                lambda x : 1 if (x == '1.5Unf') | (x == '2.5Unf') else 0)
for c in constructions.columns:

    if constructions[c].dtype == 'O':

        print('{}'.format(constructions[c].value_counts()))
constructions.drop('Utilities', axis=1, inplace=True)
constructions.CentralAir = pd.get_dummies(constructions.CentralAir, drop_first=True)
constructions = pd.concat([constructions, pd.get_dummies(constructions.PavedDrive, prefix='PavedDrive')], axis=1)

constructions.drop(['PavedDrive', 'PavedDrive_P'], axis=1, inplace=True)

constructions.head()
null_cols = constructions.columns[constructions.isna().any()].tolist()

sns.heatmap(constructions[null_cols].isnull())
constructions.drop('MiscFeature', axis=1, inplace=True)
constructions.Exterior2nd = constructions.apply(lambda x : 'None' if x['Exterior1st'] == x['Exterior2nd'] else x['Exterior2nd'], axis = 1)
null_cols = constructions.columns[constructions.isna().any()].tolist()

for c in null_cols:

    impute_val = constructions[c].value_counts().index[0]

    constructions[c] = constructions[c].map(lambda x : impute_val if pd.isnull(x) else x)
constructions.head()
constructions = combine_features(constructions, 'RoofStyle', 'RoofMatl')
vals = constructions[constructions.MSSubClass == 90].HouseStyle.unique().tolist()

constructions.MSSubClass = constructions.apply(

    lambda x : (x['MSSubClass'] + vals.index(x['HouseStyle'])) if (x['MSSubClass'] == 90) else x['MSSubClass'], axis=1)
vals = constructions[constructions.MSSubClass == 190].HouseStyle.unique().tolist()

constructions.MSSubClass = constructions.apply(

    lambda x : (x['MSSubClass'] + vals.index(x['HouseStyle'])) if (x['MSSubClass'] == 190) else x['MSSubClass'], axis=1)
constructions.MSSubClass = constructions.apply(lambda x : 

                                               (x['MSSubClass'] + 2) if ((x['HouseStyle'] == '2.5Fin') & (x['MSSubClass'] == 75)) 

                                               else x['MSSubClass'], axis=1)
constructions[constructions.MSSubClass == 180].HouseStyle.value_counts()

constructions.MSSubClass = constructions.apply(lambda x : 

                                               (x['MSSubClass'] + 5) if ((x['HouseStyle'] == 'SLvl') & (x['MSSubClass'] == 180)) 

                                               else x['MSSubClass'], axis=1)
constructions.drop('HouseStyle', axis=1, inplace=True)
obj_cols = []

for c in constructions.columns:

    if constructions[c].dtype == 'O':

        print('{}: {}'.format(c, constructions[c].unique()))

        obj_cols.append(c)

print(obj_cols)
for c in obj_cols:

    constructions[c] = constructions[c].map(lambda x : str(x)) 

    constructions[c] = label_encoder.fit_transform(constructions[c])

constructions.head()
fig, axes = plt.subplots(ncols=2, nrows=9, figsize=(15,35))

for i, idx in zip(constructions.drop('Id', axis=1).columns, range(0,len(constructions.columns)-1)):

    sns.countplot(constructions[i], label=i, ax=axes.flat[idx])
constructions.describe()
constructions.info()
constructions.to_csv('constructions.csv', index=False)
disposition = full[disposition_cols].copy()

disposition.head()
null_cols = disposition.columns[disposition.isna().any()].tolist()

sns.heatmap(disposition[null_cols].isnull())
for c in null_cols:

    disposition[c] = disposition[c].map(lambda x : 0 if np.isnan(x) else int(x))
disposition['OverallHalfBath'] = disposition.HalfBath + disposition.BsmtHalfBath
disposition.describe()
disposition.info()
disposition.to_csv('disposition.csv', index=False)
measure = full[measure_cols].copy()

measure.head()
measure['CarArea'] = measure.GarageArea / disposition.GarageCars
measure['Has2ndFlr'] = measure.apply(lambda x : 0 if x['2ndFlrSF'] == 0 else 1, axis=1)
null_cols = measure.columns[measure.isna().any()].tolist()

sns.heatmap(measure[null_cols].isnull())
for c in measure.columns:

    if (c != 'Id') & (c != 'Has2ndFlr'):

        measure[c] = measure[c].map(lambda x : 0 if np.isnan(x) else float(x))

measure.describe()
measure.info()
measure.to_csv('measure.csv', index=False)
landscape = full[landscape_cols].copy()

landscape.head()
landscape['HasDoubleCond'] = landscape.apply(lambda x : 0 if x['Condition1'] == x['Condition2'] else 1, axis=1)
landscape.info()
null_cols = landscape.columns[landscape.isna().any()].tolist()

sns.heatmap(landscape[null_cols].isnull())
landscape.drop('Alley', axis=1, inplace=True)

impute_val = landscape.MSZoning.value_counts().index[0]

print(impute_val)

landscape.MSZoning = landscape.MSZoning.map(lambda x : impute_val if pd.isnull(x) else x)

landscape.LotArea = landscape.LotArea.map(lambda x : float(x))

landscape.LotFrontage = landscape.LotFrontage.map(lambda x : 0 if pd.isnull(x) else x)
landscape.info()
obj_cols = landscape.select_dtypes(include=[np.object]).columns.tolist()

for c in obj_cols:

    print(landscape[c].value_counts())
landscape.drop('Street', axis=1, inplace=True)
landscape = pd.concat([landscape, pd.get_dummies(landscape.LandSlope, prefix='LandSlope')], axis=1)

landscape.drop(['LandSlope', 'LandSlope_Sev'], axis=1, inplace=True)

landscape.head()
landscape[landscape.Condition1 != landscape.Condition2].shape
landscape.Condition2 = landscape.apply(lambda x : 'None' if x['Condition1'] == x['Condition2'] else x['Condition2'], axis = 1)
for c in landscape.select_dtypes(include=[np.object]).columns.tolist():

    landscape[c] = landscape[c].map(lambda x : str(x)) 

    landscape[c] = label_encoder.fit_transform(landscape[c])

landscape.head()
landscape.info()
landscape.describe()
landscape.to_csv('landscape.csv', index=False)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from lightgbm import LGBMRegressor



from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import mean_absolute_error
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print('train size: {}, test size {}'.format(train.shape[0], test.shape[0]))
constructions = pd.read_csv('constructions.csv')

dates = pd.read_csv('dates.csv')

disposition = pd.read_csv('disposition.csv')

landscape = pd.read_csv('landscape.csv')

measure = pd.read_csv('measure.csv')

qualities = pd.read_csv('qualities.csv')

sale_prop = pd.read_csv('sale_prop.csv')
#train_sale = train[['Id', 'SalePrice']]

test['SalePrice'] = 0

full = train[['Id', 'SalePrice']].append(test[['Id', 'SalePrice']])

full = pd.merge(dates, full, on=['Id'])

full = pd.merge(constructions, full, on=['Id'])

full = pd.merge(disposition, full, on=['Id'])

full = pd.merge(landscape, full, on=['Id'])

full = pd.merge(measure, full, on=['Id'])

full = pd.merge(qualities, full, on=['Id'])

full = pd.merge(sale_prop, full, on=['Id'])
def rfr_pred(data, pred_name, est=0, learn_rate=0, boost_type='', leaves=0, depth=-1):

    data_train = data[data.Id.isin(train.Id)].copy()

    data_test = data[data.Id.isin(test.Id)].copy()

    

    mae = 0

    lgbm = LGBMRegressor()

    if est == 0:

        print('grid search')

        parameters = {

            'num_leaves': [7, 14, 21, 28, 31, 50],

            'max_depth': [-1, 3, 5, 10],

            'learning_rate': [0.01, 0.015, 0.025],#, 0.05, 0.1, 0.15],

            'n_estimators': [50, 100, 200],#, 500, 700],

            'boosting_type' : ['dart']#['gbdt', 'dart', 'goss']

        }

        

        grid_search = GridSearchCV(lgbm, parameters, scoring = 'neg_mean_absolute_error', n_jobs= -1, cv=3)

        grid_search.fit(data.drop(['Id','SalePrice'], axis=1), data.SalePrice)

        

        mae = -1 * grid_search.best_score_



        lgbm.num_leaves = grid_search.best_params_['num_leaves']

        lgbm.max_depth = grid_search.best_params_['max_depth']

        lgbm.n_estimators = grid_search.best_params_['n_estimators']

        lgbm.learning_rate = grid_search.best_params_['learning_rate']

        lgbm.boosting_type = grid_search.best_params_['boosting_type']

        print("est {}, learn rate {}, boosting {}, leaves {}, depth {}, mae {}".format(lgbm.n_estimators,

                                                              lgbm.learning_rate,

                                                              lgbm.boosting_type,

                                                                lgbm.num_leaves,

                                                                lgbm.max_depth,           

                                                                 mae))

    

    else:

        lgbm.n_estimators = est

        lgbm.learning_rate = learn_rate

        lgbm.boosting_type = boost_type

        lgbm.num_leaves = leaves

        lgbm.max_depth = depth

        

    scores = cross_val_score(lgbm, data.drop(['Id','SalePrice'], axis=1), data.SalePrice, cv=5, scoring='neg_mean_absolute_error')

    mae = -1 * scores.mean()

        

    lgbm.fit(data_train.drop(['Id','SalePrice'], axis=1), data_train.SalePrice)

    y_pred = lgbm.predict(data_test.drop(['Id','SalePrice'], axis=1))

    

    print('mae {}'.format(mae))

    out = data_test[['Id']].copy()

    out[pred_name] = y_pred

    out[pred_name + '_err'] = mae

    return out

full_est = rfr_pred(full, 'full_est', 100, 0.01, 'dart', 50, -1)

#grid est 100, learn rate 0.01, boosting dart, leaves 50, depth -1, mae 117725.769026745

#mae 108456.84375809731

full_est
out = pd.DataFrame({'id': full_est.Id, 'SalePrice': full_est.full_est})
out.to_csv('housePrice.csv', index=False) 

out = pd.read_csv('housePrice.csv') 

out