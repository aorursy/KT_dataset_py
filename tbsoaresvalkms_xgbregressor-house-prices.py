import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pprint



from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, PredefinedSplit, cross_val_score

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, FunctionTransformer

from sklearn.metrics import mean_squared_log_error, make_scorer

from hpsklearn import HyperoptEstimator, svr, svr_linear, svr_rbf, svr_poly, svr_sigmoid, knn_regression, ada_boost_regression, gradient_boosting_regression, random_forest_regression, extra_trees_regression, sgd_regression, xgboost_regression
pd.set_option('display.max_columns', 100)

pd.set_option("display.max_rows", 400)

random_state = 42
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



df_target = df_train[['SalePrice']]

df = df_train.append(df_test)

df.drop(columns=['Id'], inplace=True)
## Function to mount info about df current.

def mount_df_info(df):

    df_info = df.dtypes.to_frame(name='type')

    df_info['count_null'] = df.isnull().values.sum(axis=0)

    df_info['nunique'] = df.nunique().values

    df_info['count_zeros'] = (df == 0).values.sum(axis=0)

    df_info['max_value'] = df.max()

    df_info['min_value'] = df.min()

    

    return df_info
df_info = mount_df_info(df)

df_info.sort_values(by='count_null', ascending=False).head()
columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']

df[columns] = df[columns].fillna('nan')

df_info = mount_df_info(df)

df_info.sort_values(by='count_null', ascending=False).head()
df.loc[df[df['GarageYrBlt'] == 2207].index, 'GarageYrBlt'] = 2010

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0).astype(int)



columns = ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']

df[columns] = df[columns].fillna('nan')

df_info = mount_df_info(df)

df_info.sort_values(by='count_null', ascending=False).head()
columns = ['BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']

df[columns] = df[columns].fillna('nan')

df_info = mount_df_info(df)

df_info.sort_values(by='count_null', ascending=False).head()
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



columns = ['Electrical', 'KitchenQual', 'MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType']



for _column in columns:

    df[_column] = df[_column].fillna(df[_column].mode()[0])



columns = ['MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars']

df[columns] = df[columns].fillna(0)



columns = ['MasVnrType', 'Functional', 'Utilities']

df[columns] = df[columns].fillna('nan')



df_info = mount_df_info(df)

df_info.sort_values(by='nunique', ascending=False).head()
columns_disc = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

                'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',

                'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1',

                'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish',

                'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']



columns_disc_qual = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',

                        'GarageQual', 'GarageCond', 'PoolQC']



columns_cont = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',

               'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

               'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',

               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']



columns_year = ['YearBuilt', 'GarageYrBlt', 'YearRemodAdd', 'YrSold', 'MoSold']



print('Columns disc', len(columns_disc))

print('Columns disc to cont', len(columns_disc_qual))

print('Columns cont', len(columns_cont))

print('Columns year', len(columns_year))
columns_disc = columns_disc + columns_disc_qual
df_test = df.iloc[df_train.shape[0]:]

df_train = df.iloc[:df_train.shape[0]]

print(df_train.shape)

print(df_test.shape)
fig, axs = plt.subplots(len(columns_cont) // 3 + 1, 3, figsize=(18, 40))

i = 0

corr_with_sales_price = df_train[columns_cont + ['SalePrice']].corr(method='spearman')['SalePrice'].sort_values(ascending=False)



for _column in corr_with_sales_price.index[1:]:

    _ax = axs[i//3, i%3]

    _ax.boxplot(df_train[_column], whis=3)

    _ax.set_title(f'{_column} - {corr_with_sales_price[_column]}')

    i+=1

fig, axs = plt.subplots(len(columns_disc) // 3 + 1, 3, figsize=(18, 60))

i = 0



for _column in columns_disc:

    _ax = axs[i//3, i%3]

    _ax.hist(df_train[_column])

    _ax.set_title(f'{_column}')

    i+=1

    
df_corr_disc = pd.DataFrame(columns=['feature', 'value', 'corr_with_sale_prices', 'count'])



for _feature in columns_disc:

    df_temp = pd.get_dummies(df_train[_feature])



    for _column in df_temp.columns:

        _corr = stats.pointbiserialr(df_temp[_column], df_train['SalePrice'])

        df_corr_disc = df_corr_disc.append({'feature': _feature, 'value': _column, 'corr_with_sale_prices': _corr[0], 'count': sum(df_temp[_column])}, ignore_index=True)

df_corr_disc.sort_values(by='corr_with_sale_prices')
df_train['new_Fireplaces'] = df_train['Fireplaces'] * df_train['FireplaceQu'].map({'nan': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)

df['new_Fireplaces'] = df['Fireplaces'] * df['FireplaceQu'].map({'nan': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)



df_train[['new_Fireplaces', 'SalePrice']].corr(method='spearman')
df_temp = df_train[df_train['YearRemodAdd'] > df_train['YearBuilt']]

plt.hist(df_temp['YearRemodAdd'] - df_temp['YearBuilt'], bins=30) #hist to see time remod after built (only build has remod) 

plt.show()



df_train['new_has_remod'] = df_train['YearRemodAdd'] > df_train['YearBuilt']

df_train['new_time_remod'] = df_train['YearRemodAdd'] - df_train['YearBuilt']

df['new_has_remod'] = df['YearRemodAdd'] > df['YearBuilt']

df['new_time_remod'] = df['YearRemodAdd'] - df['YearBuilt']



print(stats.pointbiserialr(df_train['new_has_remod'], df_train['SalePrice']))

df_train[['new_time_remod', 'SalePrice']].corr(method='spearman')
plt.hist(df_train['YrSold'] - df_train['YearBuilt'], bins=30)

plt.show()



df_train['new_time_sold'] = df_train['YrSold'] - df_train['YearBuilt']

df['new_time_sold'] = df['YrSold'] - df['YearBuilt']



df_train[['new_time_sold', 'SalePrice']].corr(method='spearman')
df_train['new_garage_after_build'] = df_train['GarageYrBlt'] > df_train['YearBuilt']

df['new_garage_after_build'] = df['GarageYrBlt'] > df['YearBuilt']



print(stats.pointbiserialr(df_train['new_garage_after_build'], df_train['SalePrice']))
df_train['new_totalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF'] + df_train['GarageArea']

df['new_totalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GarageArea']



df_train[['new_totalSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'SalePrice']].corr(method='spearman')
df_train['BsmtFullBath'] = df_train['BsmtFullBath'].astype(int)

df['BsmtFullBath'] = df['BsmtFullBath'].astype(int)



df_train['BsmtHalfBath'] = df_train['BsmtHalfBath'].astype(int)

df['BsmtHalfBath'] = df['BsmtHalfBath'].astype(int)



df_train['new_others_room'] = df_train['TotRmsAbvGrd'] - df_train['BedroomAbvGr'] - df_train['KitchenAbvGr']

df['new_others_room'] = df['TotRmsAbvGrd'] - df['BedroomAbvGr'] - df['KitchenAbvGr']



df_train['new_all_room'] = df_train['TotRmsAbvGrd'] + df_train['BsmtFullBath'] + df_train['BsmtHalfBath'] + df_train['FullBath'] + df_train['HalfBath']

df['new_all_room'] = df['TotRmsAbvGrd'] + df['BsmtFullBath'] + df['BsmtHalfBath'] + df['FullBath'] + df['HalfBath']



df_train[['TotRmsAbvGrd', 'BedroomAbvGr', 'KitchenAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'new_others_room', 'new_all_room', 'SalePrice']].corr(method='spearman')
_df_temp = pd.DataFrame()



for _columns in columns_disc_qual:

    _df_temp[_columns] = df[_columns].map({'nan': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)

    

df['new_overall'] = _df_temp[columns_disc_qual].sum(axis=1) / len(columns_disc_qual)
df.plot(kind='scatter', x='new_overall', y= 'OverallQual')
columns_disc = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

                'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',

                'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1',

                'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish',

                'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'MoSold',

                

                'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',

                'GarageQual', 'GarageCond', 'PoolQC',

                

                'new_has_remod', 'new_garage_after_build']





columns_cont = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',

               'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

               'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',

               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

                

                'new_Fireplaces', 'new_time_remod', 'new_time_sold', 'new_totalSF', 'new_others_room', 'new_all_room', 'new_overall']



columns_year = ['YearBuilt', 'GarageYrBlt', 'YearRemodAdd', 'YrSold']



print('Columns disc', len(columns_disc))

print('Columns cont', len(columns_cont))

print('Columns year', len(columns_year))
columns_cont = columns_cont + columns_year
oneHot = OneHotEncoder(handle_unknown='ignore')

df_onehot = oneHot.fit_transform(df[columns_disc])
df_disc = pd.DataFrame.sparse.from_spmatrix(df_onehot, columns=oneHot.get_feature_names(df[columns_disc].columns)).astype(bool).reset_index()

df_cont = df[columns_cont].reset_index()

df_processed = pd.concat([df_cont, df_disc], sort=False, axis=1).drop(columns=['index'])
standard = StandardScaler()

df_processed[columns_cont] = standard.fit_transform(df_processed[columns_cont])



transformer = FunctionTransformer(np.log1p)

df_target = transformer.transform(df_target)
df_processed[columns_cont].hist(figsize=(18,30))

plt.show()
df_train = df_processed.iloc[:df_train.shape[0]]

df_test = df_processed.iloc[df_train.shape[0]:]

print(df_train.shape)

print(df_test.shape)
def rmsqle(y, y_pred):

    return np.sqrt(mean_squared_log_error(np.expm1(np.abs(y)), np.expm1(np.abs(y_pred))))
def evaluate(model, _X_test, _y_test):

    _y_pred = model.predict(_X_test)

    return rmsqle(_y_test, _y_pred)
def write_submission(_result):

    df_to_submit = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

    df_to_submit['SalePrice'] = np.expm1(_result)

    df_to_submit.to_csv('/kaggle/working/to_submit.csv', index=False)

    df_to_submit.head()
rmsqle_score = make_scorer(rmsqle, greater_is_better=False)
columns_to_drop = ['MSSubClass_150', 'Utilities_nan', 'Functional_nan']

df_train.drop(columns=columns_to_drop, inplace=True)

df_test.drop(columns=columns_to_drop, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df_train, df_target, test_size=0.25, random_state=random_state)
from xgboost import XGBRegressor, plot_importance

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213,

                             random_state=7, nthread = -1)



xgb.fit(X_train, y_train)

print(rmsqle(xgb.predict(X_test), y_test))
importances = pd.Series(xgb.feature_importances_, index=list(df[columns_cont].columns) + list(set(oneHot.get_feature_names(df[columns_disc].columns))- set(columns_to_drop)))

importances.sort_values(ascending=False).head(20) * 100
lgbm = LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



lgbm.fit(X_train, y_train)

print(rmsqle(lgbm.predict(X_test), y_test))
importances = pd.Series(lgbm.feature_importances_, index=list(df[columns_cont].columns) + list(set(oneHot.get_feature_names(df[columns_disc].columns))- set(columns_to_drop)))

importances.sort_values(ascending=False).head(20) * 100
gradient = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)



gradient.fit(X_train, y_train)

print(rmsqle(gradient.predict(X_test), y_test))
importances = pd.Series(gradient.feature_importances_, index=list(df[columns_cont].columns) + list(set(oneHot.get_feature_names(df[columns_disc].columns))- set(columns_to_drop)))

importances.sort_values(ascending=False).head(20) * 100
predict_xgb = xgb.predict(df_test)

predict_lgbm = lgbm.predict(df_test)

predict_gradient = gradient.predict(df_test)



predict_final = (predict_lgbm * 0.30 + predict_xgb * 0.50 + predict_gradient * 0.20)

write_submission(predict_final)