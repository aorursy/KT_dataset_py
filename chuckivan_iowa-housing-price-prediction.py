import os
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# All predictors are the same, but train contains the labels.
train.columns.difference(test.columns)
target = 'SalePrice'
labels = train[target]
train['Test'] = 0
test['Test'] = 1
full = pd.concat([train.drop(target, axis=1), test])
full.info()
full.describe()
# Only numerical predictors in these histograms
full.hist(bins=50, figsize=(30, 20))
plt.show()
explore = train.copy()
corr_matrix = explore.corr()
corr_matrix = corr_matrix[target].sort_values(ascending=False)
corr_matrix
high_corr_preds = []
for index in corr_matrix.index:
    if abs(corr_matrix[index]) >= 0.5:
        high_corr_preds.append(index)
high_corr_preds
scatter_matrix(explore[high_corr_preds], figsize=(20,15))
plt.show()
linear_corr_preds = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']
for pred in linear_corr_preds:
    explore.plot(kind='scatter', x=target, y=pred)
# Only 79 predictors; Analyze and decide how to deal with missing values case by case
# Useful methods: describe(), value_counts()
full.info()
med_imputer = SimpleImputer(strategy='median')
freq_imputer = SimpleImputer(strategy='most_frequent')
med_fix = ['LotFrontage']
freq_fix = [
    'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath',
    'KitchenQual', 'Functional', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
    'GarageArea', 'GarageQual', 'GarageCond', 'SaleType'
    ]
drop_fix = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
def impute(data, preds, imputer):
    for pred in preds:
        data[pred] = imputer.fit_transform(data[[pred]]).ravel()
# All missing values filled
impute(full, med_fix, med_imputer)
impute(full, freq_fix, freq_imputer)
full = full.drop(columns=drop_fix)
full.info()
# One-hot encode (nonbiased factorization) object categorical predictors 
def onehot_encode(data):
    preds = data.select_dtypes(include=[object]).columns.tolist()
    encoder = OneHotEncoder(sparse=False)
    for pred in preds:
        data[pred] = encoder.fit_transform(data[[pred]])
full_clean = full.copy()
onehot_encode(full_clean)
# Split the data again, and scale separately
train_cleaned = full_clean.loc[full_clean['Test'] == 0].drop('Test', axis=1)
test_cleaned = full_clean.loc[full_clean['Test'] == 1].drop('Test', axis=1)
def std_scaled(data):
    scaler = StandardScaler()
    # Don't scale row Id
    preds = data.select_dtypes(include=[np.int, np.float]).columns.tolist()[1:]
    for pred in preds:
        data[pred] = scaler.fit_transform(data[[pred]])
    return data
%%capture
train_prepared = std_scaled(train_cleaned)
test_prepared = std_scaled(test_cleaned)
%%capture
gbrt = GradientBoostingRegressor()
gbrt.fit(train_prepared, labels)
some_data = train_prepared[:10]
some_labels = labels[:10]
print("Predictions:", gbrt.predict(some_data))
print("Actual labs:", list(some_labels))
def train_rmse(model, train, labels):
    preds = model.predict(train)
    return np.sqrt(mean_squared_error(labels, preds))
train_rmse(gbrt, train_prepared, labels)
# Grid search for max_depth and learning_rate
# param_grid = [{
#     'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     'n_estimators': [250],
#     'max_depth': [2]
# }]
# grid_search = GridSearchCV(gbrt, param_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
# grid_search.fit(train_prepared, labels)
# grid_search.best_params_
gbrt = GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, max_depth=3)
gbrt.fit(train_prepared, labels)
train_rmse(gbrt, train_prepared, labels)
predictions = gbrt.predict(test_prepared).tolist()
submission = pd.DataFrame({'Id': test_prepared.Id, target: predictions})
submission.to_csv('submission.csv', index=False)
