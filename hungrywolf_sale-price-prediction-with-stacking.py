import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder, RobustScaler

from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor

import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = 100

pd.options.display.max_rows = 100

pd.options.display.width=1000
house_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

house_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
house_train.head()
house_train.columns
house_train.shape, house_test.shape
sns.distplot(house_train['SalePrice'])

plt.show()
def status_of_missing_data(df):

    count = df.isnull().sum().sort_values(ascending=False) # missing value count

    per = (df.isnull().sum()*100/df.shape[0]).sort_values(ascending=False) # missing value percentage

    missing_data = pd.concat([count, per], axis=1, keys=['Count', 'Percentage'])

    features_with_null = missing_data[missing_data['Count']>0]

    print('Total features with null values: - ', features_with_null.shape[0])

    return features_with_null
status_of_missing_data(house_train)
status_of_missing_data(house_test)
house_train.drop(columns=[ 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], inplace=True)

house_test.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], inplace=True)

house_train.shape, house_test.shape
house_train['LotFrontage'].sample(10)
house_train['LotFrontage'].fillna(house_train['LotFrontage'].median(), inplace=True)

house_test['LotFrontage'].fillna(house_train['LotFrontage'].median(), inplace=True)
house_train[['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']].sample(10)
house_train['GarageType'].fillna('No Garage', inplace=True)

house_test['GarageType'].fillna('No Garage', inplace=True)



house_train['GarageYrBlt'].fillna(round(house_train['GarageYrBlt'].median()), inplace=True)

house_test['GarageYrBlt'].fillna(round(house_train['GarageYrBlt'].median()), inplace=True)



house_train['GarageFinish'].fillna('No Garage', inplace=True)

house_test['GarageFinish'].fillna('No Garage', inplace=True)



house_train['GarageCond'].fillna('No Garage', inplace=True)

house_test['GarageCond'].fillna('No Garage', inplace=True)



house_train['GarageQual'].fillna('No Garage', inplace=True)

house_test['GarageQual'].fillna('No Garage', inplace=True)
house_train['BsmtFinType2'].fillna('No Basement', inplace=True)

house_test['BsmtFinType2'].fillna('No Basement', inplace=True)



house_train['BsmtExposure'].fillna('No Basement', inplace=True)

house_test['BsmtExposure'].fillna('No Basement', inplace=True)



house_train['BsmtQual'].fillna('No Basement', inplace=True)

house_test['BsmtQual'].fillna('No Basement', inplace=True)



house_train['BsmtFinType1'].fillna('No Basement', inplace=True)

house_test['BsmtFinType1'].fillna('No Basement', inplace=True)



house_train['BsmtCond'].fillna('No Basement', inplace=True)

house_test['BsmtCond'].fillna('No Basement', inplace=True)
house_train['MasVnrType'].fillna('None', inplace=True)

house_test['MasVnrType'].fillna('None', inplace=True)



house_train['MasVnrArea'].fillna(0.0, inplace=True)

house_test['MasVnrArea'].fillna(0.0, inplace=True)



house_train['Electrical'].fillna('SBrkr', inplace=True)



house_test['MSZoning'].fillna('RL', inplace=True)

house_test['BsmtFullBath'].fillna(0.0, inplace=True)

house_test['Utilities'].fillna('AllPub', inplace=True)

house_test['BsmtHalfBath'].fillna(0, inplace=True)

house_test['Functional'].fillna('Typ', inplace=True)

house_test['TotalBsmtSF'].fillna(house_train['TotalBsmtSF'].median(), inplace=True)

house_test['GarageArea'].fillna(house_train['GarageArea'].median(), inplace=True)

house_test['BsmtFinSF2'].fillna(house_train['BsmtFinSF2'].median(), inplace=True)

house_test['BsmtUnfSF'].fillna(house_train['BsmtUnfSF'].median(), inplace=True)

house_test['SaleType'].fillna('WD', inplace=True)

house_test['Exterior2nd'].fillna('VinylSd', inplace=True)

house_test['Exterior1st'].fillna('VinylSd', inplace=True)

house_test['KitchenQual'].fillna('Gd', inplace=True)

house_test['GarageCars'].fillna(2, inplace=True)

house_test['BsmtFinSF1'].fillna(house_train['BsmtFinSF1'].median(), inplace=True)
numeric_col= set(['1stFlrSF', '2ndFlrSF', '3SsnPorch',

              'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF',

              'EnclosedPorch',

              'Fireplaces', 'FullBath', 

              'GarageArea', 'GarageCars', 'GarageYrBlt', 'GrLivArea',

              'HalfBath',

              'KitchenAbvGr',

                'Id',

              'LotArea', 'LotFrontage', 'LowQualFinSF',

              'MasVnrArea', 'MiscVal', 

              'OpenPorchSF',

              'PoolArea',

              'ScreenPorch',

              'TotRmsAbvGrd', 'TotalBsmtSF',

              'WoodDeckSF',

              'YearBuilt', 'YearRemodAdd'])
categorical_col = set(house_train.columns) - numeric_col

categorical_col.remove('SalePrice')

categorical_col
house_train.dtypes
house_train['MSSubClass'] = house_train['MSSubClass'].astype(str)

house_train['OverallQual'] = house_train['OverallQual'].astype(str)

house_train['OverallCond'] = house_train['OverallCond'].astype(str)

house_train['MoSold'] = house_train['MoSold'].astype(str)

house_train['YrSold'] = house_train['YrSold'].astype(str)
house_train['GarageYrBlt'] = house_train['GarageYrBlt'].astype(int)
house_test.dtypes
house_test['MSSubClass'] = house_test['MSSubClass'].astype(str)

house_test['OverallQual'] = house_test['OverallQual'].astype(str)

house_test['OverallCond'] = house_test['OverallCond'].astype(str)



house_test['BsmtFinSF1'] = house_test['BsmtFinSF1'].astype(int)

house_test['BsmtFinSF2'] = house_test['BsmtFinSF2'].astype(int)

house_test['BsmtUnfSF'] = house_test['BsmtUnfSF'].astype(int)

house_test['TotalBsmtSF'] = house_test['TotalBsmtSF'].astype(int)

house_test['BsmtFullBath'] = house_test['BsmtFullBath'].astype(int)

house_test['BsmtHalfBath'] = house_test['BsmtHalfBath'].astype(int)

house_test['GarageYrBlt'] = house_test['GarageYrBlt'].astype(int)

house_test['GarageCars'] = house_test['GarageCars'].astype(int)

house_test['GarageArea'] = house_test['GarageArea'].astype(int)



house_test['MoSold'] = house_test['MoSold'].astype(str)

house_test['YrSold'] = house_test['YrSold'].astype(str)
for col in categorical_col:

    print('Feature name:- ', col)

    ax = sns.catplot(kind='strip', x=col, y='SalePrice', data=house_train)

    ax.set_xticklabels(rotation=60)

    plt.show()
house_train.drop(columns=['Street', 'Utilities', 'Condition2'], inplace=True)

house_test.drop(columns=['Street', 'Utilities', 'Condition2'], inplace=True)
categorical_col -= {'Street', 'Utilities', 'Condition2'}
le = LabelEncoder()

all_data = pd.concat([house_train, house_test])

all_data = all_data.reset_index(drop=True)

for col in categorical_col:

    all_data[col] = le.fit_transform(all_data[col])

house_train = all_data[: house_train.shape[0]]

house_test = all_data[house_train.shape[0]:]

house_test = house_test.drop('SalePrice', axis=1)

house_train.shape, house_test.shape
house_train['TotalArea'] = house_train['TotalBsmtSF'] + house_train['1stFlrSF'] + house_train['2ndFlrSF']



house_train['TotalBath'] = house_train['BsmtFullBath'] + house_train['FullBath'] + (house_train['BsmtHalfBath'] + house_train['HalfBath'])*0.5



house_test['TotalArea'] = house_test['TotalBsmtSF'] + house_test['1stFlrSF'] + house_test['2ndFlrSF']

house_test['TotalBath'] = house_test['BsmtFullBath'] + house_test['FullBath'] + (house_test['BsmtHalfBath'] + house_test['HalfBath'])*0.5



house_train.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'FullBath',

                  'BsmtHalfBath', 'HalfBath' ], axis=1, inplace=True)



house_test.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'FullBath',

                  'BsmtHalfBath', 'HalfBath' ], axis=1, inplace=True)

def corr_heatmap(corr, fig_size):

    f, axes = plt.subplots(1, 1, figsize=fig_size)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, annot=True, square=True, linewidth=0.1, linecolor='black', 

                ax=axes, center=3, fmt='0.2f')

    plt.show()
corr = house_train.corr()

corr_heatmap(corr, (55, 55))
house_train.drop(columns=[ 'TotRmsAbvGrd','GarageArea', 'Exterior2nd'], inplace=True)

house_test.drop(columns=[ 'TotRmsAbvGrd','GarageArea', 'Exterior2nd'], inplace=True)

numeric_col.difference_update(['TotRmsAbvGrd','GarageArea'])

categorical_col.difference_update(['Exterior2nd'])
corr_sale = house_train.corr()[['SalePrice']].sort_values('SalePrice', ascending=False)

corr_sale.reset_index(inplace=True)

corr_sale = corr_sale.loc[(corr_sale['SalePrice']> 0.03) | (corr_sale['SalePrice']< -0.03)]

highly_corr_feature = list(corr_sale['index'])

test_id = house_test['Id']

house_train = house_train[highly_corr_feature]

house_test = house_test[highly_corr_feature[1:]]
new_numeric = []

for feature in highly_corr_feature:

    if feature in numeric_col:

        new_numeric.append(feature)        
rs = RobustScaler()

house_train[new_numeric] = rs.fit_transform(house_train[new_numeric])

house_test[new_numeric] = rs.transform(house_test[new_numeric])
X_train, X_test, y_train, y_test = train_test_split(house_train.drop('SalePrice', axis=1), 

                                                                            house_train[['SalePrice']], test_size=0.3,

                                                                            random_state=1126)
lr = LinearRegression()

lr.fit(X_train, y_train)

lr.score(X_train, y_train)
lr.score(X_test, y_test)
y_pred = lr.predict(X_test)

np.sqrt(MSE(y_test, y_pred))
rfr = RandomForestRegressor()

params_rfr = {

    'max_depth' :  [ 9, 10, 11, 12] ,                 

    'max_features': ['auto',],

    'n_estimators': [250, 270, 290, 300, 310, 330, 350],

    'min_samples_leaf': [3, 4, 5],

    'random_state': [1997]

}



g_rfr = GridSearchCV(rfr, param_grid=params_rfr, cv=5, verbose=1, scoring='neg_mean_squared_error', n_jobs=-1)

g_rfr.fit(X_train, y_train) 

g_rfr.best_params_ 
best_rfr_model = g_rfr.best_estimator_

best_rfr_model.fit(X_train, y_train)

y_pred_rfr = best_rfr_model.predict(X_test)

np.sqrt(MSE(y_test, y_pred_rfr))               
housing_dmatrix = xgb.DMatrix(data = house_train.drop('SalePrice', axis=1), label=house_train[['SalePrice']])

g_params_grid = {

    'learning_rate': [0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4,],

    'n_estimators': [130, 140, 150, 160, 170, 180, 190, 200],

'subsample': [0.2, 0.3, 0.4, 0.5, 0.6],

    'seed' : [1997]

}



xbo = xgb.XGBRegressor()

g_xbo = GridSearchCV(estimator=xbo, param_grid=g_params_grid, scoring='neg_mean_squared_error',

                     cv=5, verbose=1, n_jobs=-1)

g_xbo.fit(X_train, y_train)

g_xbo.best_params_             
best_xgb_model = g_xbo.best_estimator_

best_xgb_model.fit(X_train, y_train)

y_pred_xgb = best_xgb_model.predict(X_test)    

np.sqrt(MSE(y_test, y_pred_xgb))         
base_model = {

    ('xgb', xgb.XGBRegressor(learning_rate=0.1, n_estimators=190, seed=1997, subsample=0.6)), 

    ('rfr', RandomForestRegressor(max_depth=12, max_features='auto', min_samples_leaf=4, 

                                  n_estimators=350, random_state=1997)),

    

}

reg = StackingRegressor(estimators=base_model, final_estimator=LinearRegression() , n_jobs=-1)

reg.fit(X_train, y_train)

y_pred_reg = reg.predict(X_test)

np.sqrt(MSE(y_test, y_pred_reg )) 
predict = reg.predict(house_test)

predict.shape
predict_df = pd.DataFrame({'Id': test_id.values,

                           'SalePrice': predict.flatten()

}, index=range(predict.shape[0]))

predict_df.head()
predict_df.to_csv('hosuing data reg_base_xgb_rfr.csv', index=False)