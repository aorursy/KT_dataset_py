import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
total = [train, test]
train.shape
print('Int64 columns are: ' + str(len(train.loc[:,train.dtypes == np.int64].columns)))
print('Str columns are: ' + str(len(train.loc[:,train.dtypes == np.object].columns)))
print('Float64 columns are: ' + str(len(train.loc[:,train.dtypes == np.float64].columns)))
train.isnull().sum().sort_values(ascending=False)[train.isnull().sum().sort_values(ascending=False) > 0] / train.shape[0] * 100
test.isnull().sum().sort_values(ascending=False)[test.isnull().sum().sort_values(ascending=False) > 0] / test.shape[0] * 100
sns.distplot(train.SalePrice)
sns.distplot(np.random.normal(train.SalePrice.mean(), train.SalePrice.std(), 1000))
print('Skewness: ', train.SalePrice.skew())
print('Kurtosis: ', train.SalePrice.kurt())
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train.SalePrice)
sns.distplot(np.random.normal(train.SalePrice.mean(), train.SalePrice.std(), 1000), color='green')
print('Skewness: ', train.SalePrice.skew())
print('Kurtosis: ', train.SalePrice.kurt())
for dataset in total:
    dataset['MSSubClass'] = dataset['MSSubClass'].astype(np.object)
    dataset['MoSold'] = dataset['MoSold'].astype(np.object)
    dataset['YrSold'] = dataset['YrSold'].astype(np.object)
numeric_features = train.loc[:,train.dtypes == np.int64].columns.append(train.loc[:,train.dtypes == np.float64].columns)
skew_feats = []
for feat in numeric_features:
    if train[feat].skew() > 0.75:
        skew_feats.append(feat)
        
for dataset in total:
    for feat in skew_feats:
        dataset[feat] = dataset[feat].apply(np.log1p)
mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan:0}
mapping_1 = {'Gd': 4, 'Av': 3, 'Mn':2, 'No':1, np.nan: 0}
mapping_2 = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, np.nan: 0}
mapping_3 = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}
mapping_4 = {'Fin': 3, 'RFn': 2, 'Unf': 1, np.nan: 0}
mapping_5 = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, np.nan: 0}

for dataset in total:
    for column in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
        dataset[column] = dataset[column].map(mapping)
    dataset['BsmtExposure'] = dataset['BsmtExposure'].map(mapping_1)
    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].map(mapping_2)
    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].map(mapping_2)
    dataset['Functional'] = dataset['Functional'].map(mapping_3)
    dataset['GarageFinish'] = dataset['GarageFinish'].map(mapping_4)
    dataset['Fence'] = dataset['Fence'].map(mapping_5)
    dataset[['LotFrontage','GarageYrBlt','MasVnrArea','BsmtFullBath','BsmtHalfBath']] = dataset[['LotFrontage','GarageYrBlt','MasVnrArea','BsmtFullBath','BsmtHalfBath']].fillna(0)
    dataset[['MiscFeature','Alley','GarageType']] = dataset[['MiscFeature','Alley','GarageType']].fillna('No')
    dataset['MasVnrType'] = dataset['MasVnrType'].fillna('None')

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
test.at[1150, 'MasVnrType'] = 'BrkFace'
test.at[1116, 'GarageCars'] = 0
test.at[1116, 'GarageArea'] = 0
test.at[1116, 'GarageType'] = 0
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(0)
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(0)
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(0)
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)
test['MSZoning'] = test['MSZoning'].fillna(train['MSZoning'].mode()[0])
test['Utilities'] = test['Utilities'].fillna(train['Utilities'].mode()[0])
test['Functional'] = test['Functional'].fillna(train['Functional'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(train['SaleType'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])
for dataset in total:
    dataset['HasPool'] = dataset['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    dataset['Has2ndFloor'] = dataset['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    dataset['HasGarage'] = dataset['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    dataset['HasBsmt'] = dataset['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    dataset['HasFireplace'] = dataset['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    dataset['BltSoldYrDiff'] = dataset['YrSold'].astype(np.int64) - dataset['YearBuilt']
    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
    dataset['TotalBathr'] = dataset['FullBath'] + 0.5 * dataset['HalfBath'] + dataset['BsmtFullBath'] + 0.5 * dataset['BsmtHalfBath']
    dataset['TotalPorchSF'] = dataset['OpenPorchSF'] + dataset['3SsnPorch'] + dataset['EnclosedPorch'] + dataset['ScreenPorch'] + dataset['WoodDeckSF']
numeric_features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
       'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'SalePrice', 'BltSoldYrDiff',
        'TotalSF', 'TotalBathr', 'TotalPorchSF']
sns.pairplot(data=train, y_vars=['SalePrice'], x_vars=numeric_features[:5])
sns.pairplot(data=train, y_vars=['SalePrice'], x_vars=numeric_features[5:10])
sns.pairplot(data=train, y_vars=['SalePrice'], x_vars=numeric_features[10:15])
sns.pairplot(data=train, y_vars=['SalePrice'], x_vars=numeric_features[15:20])
sns.pairplot(data=train, y_vars=['SalePrice'], x_vars=numeric_features[20:25])
sns.pairplot(data=train, y_vars=['SalePrice'], x_vars=numeric_features[25:30])
sns.pairplot(data=train, y_vars=['SalePrice'], x_vars=numeric_features[30:35])
sns.pairplot(data=train, y_vars=['SalePrice'], x_vars=numeric_features[35:40])
sns.pairplot(data=train, y_vars=['SalePrice'], x_vars=['OverallQual','OverallCond','OpenPorchSF','TotalPorchSF'])
train = train.drop(train[(train['OverallQual'] == 10) & (train['SalePrice'] < 12.5)].index)
train = train.drop(train[(train['OverallCond'] == 2) & (train['SalePrice'] > 12)].index)
train = train.drop(train[(train['OpenPorchSF'] > 3.5) & (train['SalePrice'] < 11)].index)
train = train.drop(train[(train['TotalPorchSF'] > 6) & (train['SalePrice'] < 11)].index)
sns.pairplot(data=train, y_vars=['SalePrice'], x_vars=['OverallQual','OverallCond','OpenPorchSF','TotalPorchSF'])
train.isnull().sum().sort_values(ascending=False)[train.isnull().sum().sort_values(ascending=False) > 0]
test.isnull().sum().sort_values(ascending=False)[test.isnull().sum().sort_values(ascending=False) > 0]
columns = train.loc[:,train.dtypes == np.object].columns
df_dummies = pd.get_dummies(data=pd.concat([train, test]), columns=columns)
df_dummies.shape
train = df_dummies.iloc[:train.shape[0]]
test = df_dummies.iloc[train.shape[0]:].drop('SalePrice', axis=1)
X = pd.DataFrame(StandardScaler().fit_transform(train), columns=train.columns).drop(['Id', 'SalePrice'], axis=1)
y = train['SalePrice']
scaled_test = pd.DataFrame(StandardScaler().fit_transform(test),columns=test.columns).drop('Id', axis=1)
reg = RidgeCV()
reg.fit(X, y)
print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)
print("Best score using built-in RidgeCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
reg = ElasticNetCV()
reg.fit(X, y)
print("Best alpha using built-in ElasticNetCV: %f" % reg.alpha_)
print("Best score using built-in ElasticNetCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("ElasticNet picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
ridge_coef = coef[coef != 0]
enet_coef = coef[coef != 0]
imp_coef = enet_coef.sort_values()
import matplotlib
plt.figure(figsize=(8,18))
imp_coef.plot(kind = "barh")
plt.title("Feature importance using ElasticNet Model")
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
lasso_coef = coef[coef != 0]
imp_coef = lasso_coef.sort_values()
import matplotlib
plt.figure(figsize=(8,18))
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
models = [('DTR', DecisionTreeRegressor()),
          ('RFR', RandomForestRegressor()),
          ('KNR', KNeighborsRegressor()),
          ('GBR', GradientBoostingRegressor()),
          ('LR', LinearRegression()),
          ('XGB', XGBRegressor()),
          ('LGBM', LGBMRegressor()),
          ('SVR', SVR()),
          ('Ridge', Ridge(alpha=10)),
          ('Lasso', Lasso(alpha=0.003487)),
          ('ENet', ElasticNet(alpha=0.006974))]

results = []
names = []

for coef in [ridge_coef.index, enet_coef.index, lasso_coef.index]:
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=21)
        cv_results = cross_val_score(model, X[coef], y, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
params = {
    'alpha': [0.001]
    }

reg = Lasso()
rs = GridSearchCV(estimator = reg, param_grid = params, 
                               cv = 10, verbose= 5, n_jobs = -1, scoring='neg_mean_squared_error')
rs.fit(X[lasso_coef.index],y)
print(rs.best_score_)
print(rs.best_estimator_)
lasso = rs.best_estimator_
params = {
    'learning_rate': [0.01],
    'n_estimators': [2000],
    'max_depth': [11],
    'min_samples_split': [200],
    'min_samples_leaf': [10],
    'max_features': ['sqrt'],
    'subsample': [0.85]
    }

reg = GradientBoostingRegressor()
rs = GridSearchCV(estimator = reg, param_grid = params, 
                               cv = 10, verbose= 5, n_jobs = -1, scoring='neg_mean_squared_error')
rs.fit(X[lasso_coef.index],y)
print(rs.best_score_)
print(rs.best_estimator_)
gbr = rs.best_estimator_
params = {
    'alpha': [0.001]
    }

reg = ElasticNet()
rs = GridSearchCV(estimator = reg, param_grid = params, 
                               cv = 10, verbose= 5, n_jobs = -1, scoring='neg_mean_squared_error')
rs.fit(X[lasso_coef.index],y)
print(rs.best_score_)
print(rs.best_estimator_)
elasticnet = rs.best_estimator_
params = {
    'learning_rate': [0.01],
    'n_estimators': [3000],
    'max_depth': [3],
    'min_child_weight': [5],
    'gamma': [0],
    'colsample_bytree': [0.65],
    'subsample': [0.6],
    'reg_alpha':[1e-6]
    }

reg = XGBRegressor()
rs = GridSearchCV(estimator = reg, param_grid = params, 
                               cv = 10, verbose= 5, n_jobs = -1, scoring='neg_mean_squared_error')
rs.fit(X[lasso_coef.index],y)
print(rs.best_score_)
print(rs.best_estimator_)
xgboost = rs.best_estimator_
params = {
    'learning_rate': [0.01],
    'n_estimators': [3000],
    'max_depth': [3],
    'min_child_weight': [1],
    'gamma': [0],
    'colsample_bytree': [0.8],
    'subsample': [0.6],
    }

reg = LGBMRegressor()
rs = GridSearchCV(estimator = reg, param_grid = params, 
                               cv = 10, verbose= 5, n_jobs = -1, scoring='neg_mean_squared_error')
rs.fit(X[lasso_coef.index],y)
print(rs.best_score_)
print(rs.best_estimator_)
level_0 = [('ENet',elasticnet),('GBR', gbr)]
level_1 = lasso

model = StackingRegressor(estimators=level_0, final_estimator=level_1, cv=10)

cv = KFold(n_splits=10, random_state=21)
scores = cross_val_score(model, X[lasso_coef.index],y,cv=cv,scoring='neg_mean_absolute_error')
print(scores.mean())
model.fit(X[lasso_coef.index],y)
pred_stacked = model.predict(scaled_test[lasso_coef.index])
pred = np.expm1(pred_stacked)
sub = test[['Id']]
sub['SalePrice'] = pred
sub[['Id', 'SalePrice']].to_csv('pred_submission.csv', index=False, encoding='utf-8')
sub.head()