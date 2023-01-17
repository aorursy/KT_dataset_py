import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, LassoLars, LassoLarsCV, ElasticNet, ElasticNetCV, BayesianRidge, ARDRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from xgboost import XGBRegressor

%matplotlib inline
sns.set()
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 300)
base_path = os.path.join('../input/train.csv')
base_path
df = pd.read_csv(base_path)
df.head()
for col in df.columns:
    diff = df[col].isnull().sum()
    if diff != 0:
        print('missing values for {}: {}'.format(col, diff))
def fill_missings(df):
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    df['Alley'] = df['Alley'].fillna('Unknown')
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())
    df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
    df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
    df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('Unknown')
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['FireplaceQu'] = df['FireplaceQu'].fillna('Unknown')
    df['GarageType'] = df['GarageType'].fillna('Unknown')
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
    df['GarageFinish'] = df['GarageFinish'].fillna('Unknown')
    df['GarageQual'] = df['GarageQual'].fillna(df['GarageQual'].mode()[0])
    df['GarageCond'] = df['GarageCond'].fillna(df['GarageCond'].mode()[0])
    df['PoolQC'] = df['PoolQC'].fillna('Unknown')
    df['Fence'] = df['Fence'].fillna('Unknown')
    df['MiscFeature'] = df['MiscFeature'].fillna('Unknown')
    return df
df = fill_missings(df)
df.head()
df.info()
df.describe()
df.shape
df.set_index('Id', inplace=True)
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalPorch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
sns.scatterplot(x=df['TotalSF'], y=df['SalePrice']);
df['TotalSF'].loc[df['TotalSF'] > 6000].loc[df['SalePrice'] < 300000]
Id_to_drop = [524, 1299]
df.drop(Id_to_drop, inplace=True)
df.reset_index(drop=True, inplace=True)
df.shape
df['SalePrice_log_transform'] = np.log(df['SalePrice'])
df.head()
df_chi2_cols = ['Column', 'p_value']

chi2_comparison = pd.DataFrame(columns=df_chi2_cols)
row = 0

for col in df.columns:
    if col != 'SalePrice_log_transform':
        chi2_comparison.loc[row, 'Column'] = col
        df[[col, 'SalePrice_log_transform']].groupby(col, as_index=False).mean()
        cross_table = pd.crosstab(df['SalePrice_log_transform'], df[col])
        _, p_val, _, _ = stats.chi2_contingency(cross_table)
        chi2_comparison.loc[row, 'p_value'] = p_val
        row += 1

chi2_comparison.sort_values(by=['p_value'], inplace=True)
chi2_comparison.loc[chi2_comparison['p_value'] < 1e-50]
df_pf = chi2_comparison['Column'].loc[chi2_comparison['p_value'] < 1e-50]
df_pf_list = df_pf.tolist()
df_pf_list.remove('SalePrice')
df_pf_list
pf = PolynomialFeatures()
for col in df_pf_list:
    array = pf.fit_transform(df[col].values.reshape(-1, 1))
    df[col+'_poly1'] = array[:, 1]
    df[col+'_poly2'] = array[:, 2]
df.head()
dum_lst = ['MSSubClass', 'MSZoning', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
           'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']

for dum in dum_lst:
    df = pd.concat([df, pd.get_dummies(df[dum], prefix=dum)], axis=1)
df.drop(labels=dum_lst, axis=1, inplace=True)
df.head()
df_q = df[['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'LotFrontage', 'LotArea', 'YearBuilt',
           'YearRemodAdd', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageYrBlt',
           'GarageArea', 'MoSold', 'YrSold', 'TotalSF', 'TotalPorch']]
for col in df_q.columns:
    df_q[col].replace(to_replace=0, value=None, inplace=True)
for col in df_q.columns:
    quartiles_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
    df_q[col+'_quartiles_range'] = pd.qcut(df_q[col], q=4, duplicates='drop')
    df_q[col+'_quartiles_label'] = pd.qcut(df_q[col], q=4, labels=quartiles_labels, duplicates='drop')
    df_q[col+'_quartiles'] = df_q[col+'_quartiles_label'].astype('category', ordered=True,
                                                                 categories=quartiles_labels).cat.codes
    df_q.drop(labels=col+'_quartiles_range', axis=1, inplace=True)
    df_q.drop(labels=col+'_quartiles_label', axis=1, inplace=True)
    df_q.drop(labels=col, axis=1, inplace=True)
df = pd.concat([df, df_q], axis=1)
df.head()
df_num = df[['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
             'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
             'YearRemodAdd', 'TotalSF']]
for col in df_num.columns:
    df_num[col+'_log'] = np.log(1.01 + df_num[col])
    df_num.drop(labels=col, axis=1, inplace=True)
df = pd.concat([df, df_num], axis=1)
cols_to_drop = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
             'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
             'YearRemodAdd', 'TotalSF']

df.drop(cols_to_drop, axis=1, inplace=True)
df.head()
df['MSSubClass_150'] = 0
df.head()
object_cols = df.select_dtypes(include='object')

df.drop(labels=object_cols, axis=1, inplace=True)
df.drop(labels='SalePrice', axis=1, inplace=True)
df.head()
df.describe()
X = df.loc[:, df.columns != 'SalePrice_log_transform']
y = df['SalePrice_log_transform']
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
pipelines = [
('AdaBoostRegressor', Pipeline([('RS', RobustScaler()), ('ABR', AdaBoostRegressor(random_state=42))])),
('BaggingRegressor', Pipeline([('RS', RobustScaler()), ('BR', BaggingRegressor(random_state=42))])),
('ExtraTreesRegressor', Pipeline([('RS', RobustScaler()), ('ETR', ExtraTreesRegressor(random_state=42))])),
('GradientBoostingRegressor', Pipeline([('RS', RobustScaler()), ('GBR', GradientBoostingRegressor(random_state=42))])),
('RandomForestRegressor', Pipeline([('RS', RobustScaler()), ('RFR', RandomForestRegressor(random_state=42))])),
('GaussianProcessRegressor', Pipeline([('RS', RobustScaler()), ('GPR', GaussianProcessRegressor(random_state=42))])),
('Ridge', Pipeline([('RS', RobustScaler()), ('R', Ridge(random_state=42))])),
('Lasso', Pipeline([('RS', RobustScaler()), ('L', Lasso(random_state=42))])),
('LassoCV', Pipeline([('RS', RobustScaler()), ('LCV', LassoCV(random_state=42))])),
('LassoLars', Pipeline([('RS', RobustScaler()), ('LL', LassoLars())])),
('LassoLarsCV', Pipeline([('RS', RobustScaler()), ('LLCV', LassoLarsCV())])),
('ElasticNet', Pipeline([('RS', RobustScaler()), ('EN', ElasticNet(random_state=42))])),
('ElasticNetCV', Pipeline([('RS', RobustScaler()), ('ECV', ElasticNetCV(random_state=42))])),
('BayesianRidge', Pipeline([('RS', RobustScaler()), ('BR', BayesianRidge())])),
('ARDRegression', Pipeline([('RS', RobustScaler()), ('ARDR', ARDRegression())])),
('KNeighborsRegressor', Pipeline([('RS', RobustScaler()), ('KNR', KNeighborsRegressor())])),
('SVR', Pipeline([('RS', RobustScaler()), ('SVR', SVR())])),
('LinearSVR', Pipeline([('RS', RobustScaler()), ('LSVR', LinearSVR(random_state=42))])),
('NuSVR', Pipeline([('RS', RobustScaler()), ('NuSVR', NuSVR())])),
('DecisionTreeRegressor', Pipeline([('RS', RobustScaler()), ('DTR', DecisionTreeRegressor(random_state=42))])),
('XGBRegressor', Pipeline([('RS', RobustScaler()), ('XGBR', XGBRegressor(random_state=42))])),
('LinearRegression', Pipeline([('RS', RobustScaler()), ('LR', LinearRegression())]))
]

df_models_cols = ['Name', 'Train_Acc_Mean', 'Test_Acc_Mean', 'Test_Acc_3*STD']

models_comparison = pd.DataFrame(columns=df_models_cols)
kf = KFold(n_splits=5, random_state=42, shuffle=True)

row = 0

for name, model in pipelines:
    models_comparison.loc[row, 'Name'] = name
    cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_log_error')
    models_comparison.loc[row, 'Train_Acc_Mean'] = np.sqrt(-cv_results['train_score'].mean())
    models_comparison.loc[row, 'Test_Acc_Mean'] = np.sqrt(-cv_results['test_score'].mean())
    models_comparison.loc[row, 'Test_Acc_3*STD'] = np.sqrt(cv_results['test_score'].std() * 3)
    model.fit(X_train, y_train)
    row += 1

models_comparison.sort_values(by=['Test_Acc_Mean'], inplace=True)
models_comparison
best_model_name = models_comparison.iloc[0, 0]
Test_Acc_Mean = models_comparison.iloc[0, 2]
print('Best model: {} \nTest_Acc_Mean: {}'.format(best_model_name, Test_Acc_Mean))
param_grid = {
    'ENCV__l1_ratio': np.linspace(0.1, 1, 10), #0.5
    'ENCV__n_alphas': [10], #100
    'ENCV__max_iter': [200] #1000
}
    
pipe = Pipeline([('RS', RobustScaler()), ('ENCV', ElasticNetCV(random_state=42))])

gs = GridSearchCV(pipe, param_grid, cv=kf)

gs.fit(X_train, y_train)
print("best score: {}".format(gs.best_score_))
print("best params: {}".format(gs.best_params_))
best_model = gs.best_estimator_
best_model
best_model.fit(X_train, y_train)
R2_train = best_model.score(X_train, y_train)
R2_train
R2_test = best_model.score(X_test, y_test)
R2_test
base_path = os.path.join('../input/test.csv')
base_path
df_test = pd.read_csv(os.path.join(base_path))
df_test.head()
for col in df_test.columns:
    diff = df_test[col].isnull().sum()
    if diff != 0:
        print('missing values for {}: {}'.format(col, diff))
def fill_missings_test(df):
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    df['Alley'] = df['Alley'].fillna('Unknown')
    df['Utilities'] = df['Utilities'].fillna('Unknown')
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())
    df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
    df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
    df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('Unknown')
    df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('Unknown')
    df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
    df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].median())
    df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
    df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0) 
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
    df['FireplaceQu'] = df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
    df['GarageType'] = df['GarageType'].fillna('Unknown')
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
    df['GarageFinish'] = df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
    df['GarageCars'] = df['GarageCars'].fillna(df['GarageCars'].median())
    df['GarageArea'] = df['GarageArea'].fillna(df['GarageArea'].median())
    df['GarageQual'] = df['GarageQual'].fillna(df['GarageQual'].mode()[0])
    df['GarageCond'] = df['GarageCond'].fillna('Unknown')
    df['PoolQC'] = df['PoolQC'].fillna(df['PoolQC'].mode()[0])
    df['Fence'] = df['Fence'].fillna('Unknown')
    df['MiscFeature'] = df['MiscFeature'].fillna('Unknown')
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
    return df
df_test_clean = fill_missings_test(df_test)
df_test_clean.head()
def transform_df(df):
    
    df.set_index('Id', inplace=True)
    
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    df['TotalPorch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    
    df_pf = ['GarageArea', 'TotalBsmtSF', 'MasVnrArea', 'OverallQual', 'TotalSF', '1stFlrSF', 'LotFrontage',
             'LotArea', 'OpenPorchSF', 'GrLivArea', 'BsmtUnfSF']
    pf = PolynomialFeatures()
    for col in df_pf:
        array = pf.fit_transform(df[col].values.reshape(-1, 1))
        df[col+'_poly1'] = array[:, 1]
        df[col+'_poly2'] = array[:, 2]
            
    dum_lst = ['MSSubClass', 'MSZoning', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
               'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
               'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
    for dum in dum_lst:
        df = pd.concat([df, pd.get_dummies(df[dum], prefix=dum)], axis=1)
    df.drop(labels=dum_lst, axis=1, inplace=True)
    
    df_q = df[['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'LotFrontage', 'LotArea', 'YearBuilt',
           'YearRemodAdd', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageYrBlt',
           'GarageArea', 'MoSold', 'YrSold', 'TotalSF', 'TotalPorch']]
    for col in df_q.columns:
        df_q[col].replace(to_replace=0, value=None, inplace=True)
    for col in df_q.columns:
        quartiles_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
        df_q[col+'_quartiles_range'] = pd.qcut(df_q[col], q=4, duplicates='drop')
        df_q[col+'_quartiles_label'] = pd.qcut(df_q[col], q=4, labels=quartiles_labels, duplicates='drop')
        df_q[col+'_quartiles'] = df_q[col+'_quartiles_label'].astype('category', ordered=True,
                                                                     categories=quartiles_labels).cat.codes
        df_q.drop(labels=col+'_quartiles_range', axis=1, inplace=True)
        df_q.drop(labels=col+'_quartiles_label', axis=1, inplace=True)
        df_q.drop(labels=col, axis=1, inplace=True)
    df = pd.concat([df, df_q], axis=1)
    
    df_num = df[['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
             'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
             'YearRemodAdd', 'TotalSF']]
    for col in df_num.columns:
        df_num[col+'_log'] = np.log(1.01 + df_num[col])
        df_num.drop(labels=col, axis=1, inplace=True)
    df = pd.concat([df, df_num], axis=1)
    
    cols_to_drop = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
             'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
             'YearRemodAdd', 'TotalSF']
    df.drop(cols_to_drop, axis=1, inplace=True)
    
    object_cols = df.select_dtypes(include='object')
    df.drop(labels=object_cols, axis=1, inplace=True)
    
    return df
df_test_clean = transform_df(df_test_clean)
df_test_clean.shape
set(df_test_clean).difference(X_train)
set(X_train).difference(df_test_clean)
for col in set(X_train).difference(df_test_clean):
    df_test_clean[col] = 0
df_test_clean.head()
df_test_clean.shape
df_test_clean = df_test_clean[X_train.columns]
best_model.fit(X_train, y_train)
y_pred = np.exp(best_model.predict(df_test_clean))
y_pred
df_sample_submission = pd.DataFrame({'Id': df_test_clean.index, 'SalePrice': y_pred})
sample_submission = pd.DataFrame.to_csv(df_sample_submission, index=False)
with open ('../submissions/sample_submission.csv', 'w') as f:
    f.write(sample_submission)