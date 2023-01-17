import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns',None)
import numpy as np
import seaborn as sns

from scipy import stats
X_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
X_train.shape, X_test.shape
X_train.head()
X_test.head()
correlation_train=X_train.corr()
sns.set(font_scale=1.2)
mask = np.triu(correlation_train.corr())
plt.figure(figsize = (20,20))
ax = sns.heatmap(correlation_train, annot=True,annot_kws={"size": 11},fmt='.1f', linewidths=.5, square=True, mask=mask)
y = X_train.SalePrice.reset_index(drop=True)
X_train.drop(['SalePrice'], axis=1, inplace=True)
plt.figure(figsize=(12,6))
sns.distplot(y)
stats.probplot(y, plot=plt)
print(f"Skewness: {y.skew():.3f}")
y = np.log1p(y)            
plt.figure(figsize=(12,6))
sns.distplot(y)
stats.probplot(y, plot=plt)
print(f"Skewness: {y.skew():.3f}")
train_test = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
train_test.shape
# Find Missing Ratio of Dataset
missing = (train_test.isnull().sum() / len(train_test)) * 100
missing = missing.drop(missing[missing == 0].index).sort_values(ascending=False)[:35]
f, ax = plt.subplots(figsize=(12, 10))
plt.xticks(rotation='90')
sns.barplot(x=missing.index, y=missing)
plt.xlabel('Features')
plt.ylabel('%')
plt.title('Percentage of missing values')
train_test['MSSubClass'] = train_test['MSSubClass'].astype(str)
train_test['MoSold'] = train_test['MoSold'].astype(str)
train_test['YrSold'] = train_test['YrSold'].astype(str)
none = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'GarageType','MasVnrType']
for col in none:
    train_test[col].replace(np.nan, 'None', inplace=True)
train_test['MSZoning'] = train_test.groupby('MSSubClass')['MSZoning'].transform(
    lambda x: x.fillna(x.mode()[0]))
freq_cols = [
    'Electrical', 'Exterior1st', 'Exterior2nd',
    'SaleType', 'Utilities'
]
for col in freq_cols:
    train_test[col].replace(np.nan, train_test[col].mode()[0], inplace=True)
qualcond = ['GarageQual', 'GarageCond', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtCond', 'BsmtQual', 'ExterCond', 'ExterQual']
for f in qualcond:
    train_test[f] = train_test[f].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
train_test['BsmtExposure'] = train_test['BsmtExposure'].replace({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NA':0})
train_test['GarageFinish'] = train_test['GarageFinish'].replace({'Fin':3, 'RFn':2, 'Unf':1, 'NA':0})
basement = ['BsmtFinType1', 'BsmtFinType2']
for f in basement:
    train_test[f] = train_test[f].replace({'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0})
functional = {'Typ': 3, 'Min1': 2.5, 'Min2': 2, 'Mod': 1.5, 'Maj1': 1, 'Maj2': 0.5, 'Sev': 0, 'Sal': 0}
train_test['Functional'] = train_test['Functional'].replace(functional)
train_test['CentralAir'] = train_test['CentralAir'].replace({'Y':1, 'N':0})
train_test.isnull().sum().sort_values(ascending=False)[:22]
cat_cols = [cname for cname in train_test.columns if  train_test[cname].dtype == "object"]
cat_cols
train = train_test.iloc[:1460]
test = train_test.iloc[1460:]
from category_encoders import CatBoostEncoder
cbe = CatBoostEncoder()
train[cat_cols] = cbe.fit_transform(train[cat_cols], y)
test[cat_cols] = cbe.transform(test[cat_cols])
outliers = [ 30, 462, 523, 588, 632, 1298, 1324]
train = train.drop(train.index[outliers])
y = y.drop(y.index[outliers])
train_test = pd.concat([train, test]).reset_index(drop=True)
train_test = train_test.drop(['Utilities', 'Street', 'PoolQC', ], axis=1)
from sklearn.impute import KNNImputer
imp = KNNImputer(n_neighbors=7, weights='distance', missing_values=np.nan)
imp_train_test = imp.fit_transform(train_test)
train_test = pd.DataFrame(imp_train_test, columns=train_test.columns)
missing = ['GarageCars', 'BsmtFinSF1', 'GarageArea', 'BsmtUnfSF', 'KitchenQual',
       'BsmtFinSF2', 'TotalBsmtSF', 'Functional', 'BsmtHalfBath',
       'BsmtFullBath', 'MasVnrArea', 'BsmtFinType1', 'BsmtFinType2',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageQual', 'GarageFinish',
       'GarageYrBlt', 'GarageCond', 'LotFrontage', 'FireplaceQu']
train_test[missing] = train_test[missing].apply(lambda x: np.round(x))
train_test
train_test['YearsSinceBuilt'] = train_test['YrSold'].astype(int) - train_test['YearBuilt']
train_test['YearsSinceRemod'] = train_test['YrSold'].astype(int) - train_test['YearRemodAdd']
train_test['TotalSF'] = train_test['TotalBsmtSF'] + train_test['1stFlrSF'] + train_test['2ndFlrSF']

train_test['Total_Bathrooms'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) +
                               train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))

train_test['TotalPorchArea'] = (train_test['OpenPorchSF'] + train_test['3SsnPorch'] +
                              train_test['EnclosedPorch'] + train_test['ScreenPorch'] +
                              train_test['WoodDeckSF'])
train_test['TotalOccupiedArea'] = train_test['TotalSF'] + train_test['TotalPorchArea']
train_test['OtherRooms'] = train_test['TotRmsAbvGrd'] - train_test['BedroomAbvGr'] - train_test['KitchenAbvGr']
train_test['haspool'] = train_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['has2ndfloor'] = train_test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train_test['hasgarage'] = train_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['hasbsmt'] = train_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train_test['hasfireplace'] = train_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
print(train_test.shape)
train = train_test.iloc[:1453]
test = train_test.iloc[1453:]
print(train.shape, test.shape, y.shape)
from catboost import CatBoostRegressor, Pool
model = CatBoostRegressor(iterations=2500,
                            learning_rate=0.03,
                            depth=6,
                            loss_function='RMSE',
                            random_seed = 10,
                            bootstrap_type='Bernoulli',
                            subsample=0.66,
                            rsm=0.7
                         )
model.fit(train, y, verbose=False, plot=False)
import shap
shap.initjs()

shap_values = model.get_feature_importance(Pool(train, y), type='ShapValues')

expected_value = shap_values[0,-1]
shap_values = shap_values[:,:-1]

shap.force_plot(expected_value, shap_values[0,:], train.iloc[0,:])
shap.force_plot(expected_value, shap_values, train)
shap.summary_plot(shap_values, train, max_display=88,  plot_type='bar')
shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame([train.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
importance_df.tail(35)
drop = importance_df[importance_df['shap_importance'] < 1.5e-3].iloc[:,0].tolist()
train_drop = train.drop(drop, axis=1)
test_drop = test.drop(drop, axis=1)
train_drop
import xgboost as xgb

import optuna


def objective(trial):
    dtrain = xgb.DMatrix(train_drop, label=y)

    param = {
        'seed': 20,
        'tree_method': 'gpu_hist',
        'max_depth': trial.suggest_int("max_depth", 3, 8),
        'eta' : trial.suggest_uniform("eta", 1e-3, 5e-2),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "gamma": trial.suggest_uniform("gamma", 1e-8, 1e-4),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.3, 1.0),        
    }
    if param['grow_policy']=="lossguide":
        param['max_leaves'] =  trial.suggest_int('max_leaves',2, 32)
    bst = xgb.cv(param, dtrain, num_boost_round=5000, nfold=10, early_stopping_rounds=50,  metrics='rmse', seed=20)
    score = bst['test-rmse-mean'].tail(1).values[0]
    return score


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)
print(study.best_trial)
from optuna.visualization import *
plot_optimization_history(study)
plot_parallel_coordinate(study)
plot_parallel_coordinate(study, params=['colsample_bytree', 'subsample'])
importances = optuna.importance.get_param_importances(study)
importance_values = list(importances.values())
param_names = list(importances.keys())
params = pd.DataFrame([param_names, importance_values]).T
params.columns = ['param_name', 'importance']
params = params.sort_values('importance', ascending=False)
sns.catplot(x='param_name', y='importance', data=params, kind='bar')
plt.xticks(rotation='45')
plot_contour(study)
plot_contour(study, params=['colsample_bytree', 'subsample'])
plot_slice(study)
plot_slice(study, params=['colsample_bytree', 'subsample'])