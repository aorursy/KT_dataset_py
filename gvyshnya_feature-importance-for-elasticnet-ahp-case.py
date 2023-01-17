
!pip install pdpipe
# Loading neccesary packages

import numpy as np
import pandas as pd
import pdpipe as pdp

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

#

from scipy import stats
from scipy.stats import skew, boxcox_normmax, norm
from scipy.special import boxcox1p

#

from typing import Dict, List, Tuple, Sequence
def get_data_file_path(in_kaggle: bool) -> Tuple[str, str]:
    train_set_path = ''
    test_set_path = ''
    
    if in_kaggle:
        # running in Kaggle, inside 
        # 'House Prices: Advanced Regression Techniques' competition kernel container
        train_set_path = '../input/house-prices-advanced-regression-techniques/train.csv'
        test_set_path = '../input/house-prices-advanced-regression-techniques/test.csv'
    else:
        # running locally
        train_set_path = 'data/train.csv'
        test_set_path = 'data/test.csv'
    
    return train_set_path,test_set_path

# Loading datasets
in_kaggle = True
train_set_path, test_set_path = get_data_file_path(in_kaggle)

train = pd.read_csv(train_set_path)
test = pd.read_csv(test_set_path)
# check train dimension
display(train.shape)
# check test dimension
display(test.shape)
# dropping unneccessary columns, merging training and test sets

train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
# Dropping outliers after detecting them by eye

train = train.drop(train[(train['OverallQual'] < 5)
                                  & (train['SalePrice'] > 200000)].index)
train = train.drop(train[(train['GrLivArea'] > 4000)
                                  & (train['SalePrice'] < 200000)].index)
train = train.drop(train[(train['GarageArea'] > 1200)
                                  & (train['SalePrice'] < 200000)].index)
train = train.drop(train[(train['TotalBsmtSF'] > 3000)
                                  & (train['SalePrice'] > 320000)].index)
train = train.drop(train[(train['1stFlrSF'] < 3000)
                                  & (train['SalePrice'] > 600000)].index)
train = train.drop(train[(train['1stFlrSF'] > 3000)
                                  & (train['SalePrice'] < 200000)].index)

# Backing up target variables and dropping them from train data

y = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test

# Merging features

features = pd.concat([train_features, test_features]).reset_index(drop=True)
print(features.shape)
# List of NaN including columns where NaN's mean none.
none_cols = [
    'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
]

# List of NaN including columns where NaN's mean 0.

zero_cols = [
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
    'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'
]

# List of NaN including columns where NaN's actually missing gonna replaced with mode.

freq_cols = [
    'Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual',
    'SaleType', 'Utilities'
]

# Filling the list of columns above:

for col in zero_cols:
    features[col].replace(np.nan, 0, inplace=True)

for col in none_cols:
    features[col].replace(np.nan, 'None', inplace=True)

for col in freq_cols:
    features[col].replace(np.nan, features[col].mode()[0], inplace=True)
    
# Filling MSZoning according to MSSubClass
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].apply(
    lambda x: x.fillna(x.mode()[0]))

# Filling LotFrontage according to Neighborhood
features['LotFrontage'] = features.groupby(
    ['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))
# Features which numerical on data but should be treated as category.
features['MSSubClass'] = features['MSSubClass'].astype(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)
# Transforming rare values(less than 10) into one group - dimensionality reduction

others = [
    'Condition1', 'Condition2', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'Heating', 'Electrical', 'Functional', 'SaleType'
]

for col in others:
    mask = features[col].isin(
        features[col].value_counts()[features[col].value_counts() < 10].index)
    features[col][mask] = 'Other'
# Converting some of the categorical values to numeric ones.

neigh_map = {
    'MeadowV': 1,
    'IDOTRR': 1,
    'BrDale': 1,
    'BrkSide': 2,
    'OldTown': 2,
    'Edwards': 2,
    'Sawyer': 3,
    'Blueste': 3,
    'SWISU': 3,
    'NPkVill': 3,
    'NAmes': 3,
    'Mitchel': 4,
    'SawyerW': 5,
    'NWAmes': 5,
    'Gilbert': 5,
    'Blmngtn': 5,
    'CollgCr': 5,
    'ClearCr': 6,
    'Crawfor': 6,
    'Veenker': 7,
    'Somerst': 7,
    'Timber': 8,
    'StoneBr': 9,
    'NridgHt': 10,
    'NoRidge': 10
}

features['Neighborhood'] = features['Neighborhood'].map(neigh_map).astype('int')
ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['ExterQual'] = features['ExterQual'].map(ext_map).astype('int')
features['ExterCond'] = features['ExterCond'].map(ext_map).astype('int')
bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['BsmtQual'] = features['BsmtQual'].map(bsm_map).astype('int')
features['BsmtCond'] = features['BsmtCond'].map(bsm_map).astype('int')
bsmf_map = {
    'None': 0,
    'Unf': 1,
    'LwQ': 2,
    'Rec': 3,
    'BLQ': 4,
    'ALQ': 5,
    'GLQ': 6
}

features['BsmtFinType1'] = features['BsmtFinType1'].map(bsmf_map).astype('int')
features['BsmtFinType2'] = features['BsmtFinType2'].map(bsmf_map).astype('int')
heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['HeatingQC'] = features['HeatingQC'].map(heat_map).astype('int')
features['KitchenQual'] = features['KitchenQual'].map(heat_map).astype('int')
features['FireplaceQu'] = features['FireplaceQu'].map(bsm_map).astype('int')
features['GarageCond'] = features['GarageCond'].map(bsm_map).astype('int')
features['GarageQual'] = features['GarageQual'].map(bsm_map).astype('int')
# Creating new features  based on previous observations

features['TotalSF'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                       features['1stFlrSF'] + features['2ndFlrSF'])
features['TotalBathrooms'] = (features['FullBath'] +
                              (0.5 * features['HalfBath']) +
                              features['BsmtFullBath'] +
                              (0.5 * features['BsmtHalfBath']))

features['TotalPorchSF'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                            features['EnclosedPorch'] +
                            features['ScreenPorch'] + features['WoodDeckSF'])

features['YearBlRm'] = (features['YearBuilt'] + features['YearRemodAdd'])

# Merging quality and conditions

features['TotalExtQual'] = (features['ExterQual'] + features['ExterCond'])
features['TotalBsmQual'] = (features['BsmtQual'] + features['BsmtCond'] +
                            features['BsmtFinType1'] +
                            features['BsmtFinType2'])
features['TotalGrgQual'] = (features['GarageQual'] + features['GarageCond'])
features['TotalQual'] = features['OverallQual'] + features[
    'TotalExtQual'] + features['TotalBsmQual'] + features[
        'TotalGrgQual'] + features['KitchenQual'] + features['HeatingQC']

# Creating new features by using new quality indicators

features['QualGr'] = features['TotalQual'] * features['GrLivArea']
features['QualBsm'] = features['TotalBsmQual'] * (features['BsmtFinSF1'] +
                                                  features['BsmtFinSF2'])
features['QualPorch'] = features['TotalExtQual'] * features['TotalPorchSF']
features['QualExt'] = features['TotalExtQual'] * features['MasVnrArea']
features['QualGrg'] = features['TotalGrgQual'] * features['GarageArea']
features['QlLivArea'] = (features['GrLivArea'] -
                         features['LowQualFinSF']) * (features['TotalQual'])
features['QualSFNg'] = features['QualGr'] * features['Neighborhood']
# Creating some simple features

features['HasPool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['Has2ndFloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['HasGarage'] = features['QualGrg'].apply(lambda x: 1 if x > 0 else 0)

features['HasBsmt'] = features['QualBsm'].apply(lambda x: 1 if x > 0 else 0)

features['HasFireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

features['HasPorch'] = features['QualPorch'].apply(lambda x: 1 if x > 0 else 0)
possible_skewed = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'LowQualFinSF', 'MiscVal'
]

# Finding skewness of the numerical features

skew_features = np.abs(features[possible_skewed].apply(lambda x: skew(x)).sort_values(
    ascending=False))

# Filtering skewed features

high_skew = skew_features[skew_features > 0.3]

# Taking indexes of high skew

skew_index = high_skew.index

# Applying boxcox transformation to fix skewness

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
# Features to drop

to_drop = [
    'Utilities',
    'PoolQC',
    'YrSold',
    'MoSold',
    'ExterQual',
    'BsmtQual',
    'GarageQual',
    'KitchenQual',
    'HeatingQC',
]

# Dropping ML-irrelevant features

features.drop(columns=to_drop, inplace=True)
# Getting dummy variables for ategorical data
features = pd.get_dummies(data=features)
print(f'Number of missing values: {features.isna().sum().sum()}')
features.shape
# Separating train and test set

train = features.iloc[:len(y), :]
test = features.iloc[len(train):, :]
# Setting model data

X = train
X_test = test
y = np.log1p(y)
# Loading neccesary packages for modelling and feature selection
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Setting kfold for future use
kf = KFold(10, random_state=42, shuffle=True)

# Train our baseline ElasticNet Regression model for feature importance scoring/feature selection
reg = ElasticNet(random_state=0)
reg.fit(X, y)

def rfe_select_featurs(X, y, estimator, num_features) -> List[str]:
    rfe_selector = RFE(estimator=estimator, 
                       n_features_to_select=num_features, 
                       step=10, verbose=5)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features')
    
    return rfe_feature
# total list of features
colnames = X.columns
# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))
# Do FRE feature importance scoring - 
# stop the search when only the last feature is left
rfe = RFE(reg, n_features_to_select=1, verbose =3 )
rfe.fit(X, y)
ranks["RFE_ElasticNet"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
# all ranks
# Put the mean scores into a Pandas dataframe
rfe_lr_df = pd.DataFrame(list(ranks['RFE_ElasticNet'].items()), columns= ['Feature','rfe_importance'])

all_ranks = rfe_lr_df

display(all_ranks.head(10))
from sklearn.inspection import permutation_importance


# Here's how you use permutation importance
def get_permutation_importance(X, y, model) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=1,
                                random_state=0)
    
    # permutational importance results
    result_df = pd.DataFrame(colnames,  columns=['Feature'])
    result_df['permutation_importance'] = result.get('importances')
    
    return result_df

permutate_df = get_permutation_importance(X, y, reg)
permutate_df.sort_values('permutation_importance', 
                   ascending=False)[
                                    ['Feature','permutation_importance'
                                    ]
                                  ][:30].style.background_gradient(cmap='Blues')
from sklearn.base import clone 

def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append( round( (benchmark_score - drop_col_score)/benchmark_score, 4) )
    
    importances_df = pd.DataFrame(X_train.columns, columns=['Feature'])
    importances_df['drop_col_importance'] = importances
    return importances_df

drop_col_impt_df = drop_col_feat_imp(reg, X, y)


drop_col_impt_df.sort_values('drop_col_importance', 
                   ascending=False)[
                                    ['Feature','drop_col_importance'
                                    ]
                                  ][:30].style.background_gradient(cmap='Blues')
# merge drop_col_impt_df
all_ranks = pd.merge(all_ranks, drop_col_impt_df, on=['Feature'])

# merge permutate_df
all_ranks = pd.merge(all_ranks, permutate_df, on=['Feature'])

# calculate average feature importance
average_fi_pipeline = pdp.PdPipeline([
    pdp.ApplyToRows(
        lambda row: (row['drop_col_importance'] + row['permutation_importance'] + row['rfe_importance'] )/3, 
        colname='mean_feature_importance') # 'mean_feature_importance
])

all_ranks = average_fi_pipeline.apply(all_ranks)

display(all_ranks.reset_index().drop(['index'], axis=1).style.background_gradient(cmap='summer_r'))
def get_top_features_by_rank(metric_col_name: str, feature_number: int):
    features_df = all_ranks.copy()
    
    # features_df = features_df.sort_values(by=['feature_number'])
    
    # TODO: [:feature_number]
    
    # top n rows ordered by multiple columns
    features_df = features_df.nlargest(feature_number, [metric_col_name])
    
    result_list = list(features_df['Feature'])
    return result_list

def model_check(X, y, estimator, model_name, model_description, cv):
    model_table = pd.DataFrame()

    cv_results = cross_validate(estimator,
                                X,
                                y,
                                cv=cv,
                                scoring='neg_root_mean_squared_error',
                                return_train_score=True,
                                n_jobs=-1)

    train_rmse = -cv_results['train_score'].mean()
    test_rmse = -cv_results['test_score'].mean()
    test_std = cv_results['test_score'].std()
    fit_time = cv_results['fit_time'].mean()

    attributes = {
        'model_name': model_name,
        'train_score': train_rmse,
        'test_score': test_rmse,
        'test_std': test_std,
        'fit_time': fit_time,
        'description': model_description,
    }
    
    model_table = pd.DataFrame(data=[attributes])
    return model_table
# check the baseline LR
baseline = model_check(X, y, reg, 'Baseline ElasticNet Regressor', "Baseline ElasticNet (all features)", kf)
result_df = baseline
# subset of features selected by RFE feature importance
top_rfe_features = 50
rfe_features = get_top_features_by_rank('rfe_importance', top_rfe_features)
X_important_features = X[rfe_features]

model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Top RFE Features', "Top 50 RFE Features", kf)
    
# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)

top_rfe_features = 100
rfe_features = get_top_features_by_rank('rfe_importance', top_rfe_features)
X_important_features = X[rfe_features]

model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Top RFE Features', "Top 100 RFE Features", kf)

# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)

top_rfe_features = 150
rfe_features = get_top_features_by_rank('rfe_importance', top_rfe_features)
X_important_features = X[rfe_features]

model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Top RFE Features', "Top 150 RFE Features", kf)

# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)
# train ElasticNet with the top importance feautres selected via the permutation method
top_features = 12
important_features = get_top_features_by_rank('permutation_importance', top_features)
X_important_features = X[important_features]

description = "ElasticNet with top {} permutatively important features".format(top_features)
model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Permutatively Important Features', description, kf)
    
# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)


top_features = 50
important_features = get_top_features_by_rank('permutation_importance', top_features)
X_important_features = X[important_features]

description = "ElasticNet with top {} permutatively important features".format(top_features)
model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Permutatively Important Features', description, kf)
    
# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)

top_features = 100
important_features = get_top_features_by_rank('permutation_importance', top_features)
X_important_features = X[important_features]

description = "ElasticNet with top {} permutatively important features".format(top_features)
model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Permutatively Important Features', description, kf)
    
# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)

top_features = 150
important_features = get_top_features_by_rank('permutation_importance', top_features)
X_important_features = X[important_features]

description = "ElasticNet with top {} permutatively important features".format(top_features)
model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Permutatively Important Features', description, kf)
    
# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)

# train ElasticNet with the top importance feautres selected via the drop-column method
top_features = 12
important_features = get_top_features_by_rank('drop_col_importance', top_features)
X_important_features = X[important_features]

description = "ElasticNet with top {} drop-col-important features".format(top_features)
model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Drop-Column Important Features', description, kf)
    
# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)

top_features = 50
important_features = get_top_features_by_rank('drop_col_importance', top_features)
X_important_features = X[important_features]

description = "ElasticNet with top {} drop-col-important features".format(top_features)
model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Drop-Column Important Features', description, kf)
    
# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)

top_features = 100
important_features = get_top_features_by_rank('drop_col_importance', top_features)
X_important_features = X[important_features]

description = "ElasticNet with top {} drop-col-important features".format(top_features)
model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Drop-Column Important Features', description, kf)
    
# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)

top_features = 150
important_features = get_top_features_by_rank('drop_col_importance', top_features)
X_important_features = X[important_features]

description = "ElasticNet with top {} drop-col-important features".format(top_features)
model_check_df = model_check(X_important_features, y, reg, 'ElasticNet - Drop-Column Important Features', description, kf)
    
# concatenate
frames = [result_df, model_check_df]
result_df = pd.concat(frames)
display(result_df.reset_index().drop(['index'], axis=1).style.background_gradient(cmap='summer_r'))