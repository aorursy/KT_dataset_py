import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()


from catboost import CatBoostRegressor, Pool, cv

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.feature_selection import RFECV

import eli5
from eli5.sklearn import PermutationImportance

import math
import hyperopt

from numpy.random import RandomState
from os import listdir


import shap
# load JS visualization code to notebook
shap.initjs()
listdir("../input")
def run_catboost(traindf, testdf, holddf, params, n_splits=10, n_repeats=1,
                 plot=False, use_features=None, plot_importance=True, init_model=None):
    
    
    folds = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    p_hold = np.zeros(holddf.shape[0])
    y_hold = holddf.target
    p_test = np.zeros(testdf.shape[0])
    if use_features is None:
        use_features = testdf.columns.values
    
    cat_features = np.where(testdf.loc[:, use_features].dtypes=="object")[0]
    x_hold = holddf.loc[:, use_features]
    x_test = testdf.loc[:, use_features]
    
    feature_importance_df = pd.DataFrame(index=use_features)
    
    best_model = None
    best_score = None
    
    m = 0
    cv_scores = []
    for train_idx, dev_idx in folds.split(traindf):
        x_train, x_dev = traindf.iloc[train_idx][use_features], traindf.iloc[dev_idx][use_features]
        y_train, y_dev = traindf.target.iloc[train_idx], traindf.target.iloc[dev_idx]

        train_pool = Pool(x_train, y_train, cat_features=cat_features)
        dev_pool = Pool(x_dev, y_dev, cat_features=cat_features)
        model = CatBoostRegressor(**params)
        if init_model:
            model.fit(train_pool, eval_set=dev_pool, plot=plot, init_model=None)
        else:
            model.fit(train_pool, eval_set=dev_pool, plot=plot)

        # bagging predictions for test and hold out data:
        p_hold += model.predict(x_hold)/(n_splits*n_repeats)
        log_p_test = model.predict(x_test)
        p_test += (np.exp(log_p_test) - 1)/(n_splits*n_repeats)

        # predict for dev fold:
        y_pred = model.predict(x_dev)
        cv_scores.append(np.sqrt(mse(y_dev, y_pred)))
        if not best_score:
            best_score = mse(y_dev, y_pred)
            best_model = model
        elif mse(y_dev, y_pred) < best_score:
            best_model = model
            best_score = mse(y_dev, y_pred)
        m+=1

    print("hold out rmse: " + str(np.sqrt(mse(y_hold, p_hold))))
    print("cv mean rmse: " + str(np.mean(cv_scores)))
    print("cv std rmse: " + str(np.std(cv_scores)))
    
    results = {"best_model": best_model,
               "p_hold": p_hold,
               "p_test": p_test,
               "hold_rmse": np.sqrt(mse(y_hold, p_hold)),
               "cv_scores": cv_scores}
    return results
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", index_col=0)
train.head()
train.shape
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", index_col=0)
test.head()
test.shape
train.shape[0] / test.shape[0]
train["LogSalePrice"] = train.SalePrice.apply(np.log)
combined = train.drop(["SalePrice", "LogSalePrice"], axis=1).append(test)
nan_percentage = combined.isnull().sum().sort_values(ascending=False) / combined.shape[0]
missing_val = nan_percentage[nan_percentage > 0]
num_candidates = list(combined.dtypes[combined.dtypes!="object"].index.values)
num_candidates
unique_counts = combined.loc[:, num_candidates].nunique().sort_values()

plt.figure(figsize=(20,5))
sns.barplot(unique_counts.index, unique_counts.values, palette="Oranges_r")
plt.xticks(rotation=90);
plt.yscale("log")
cat_candidates = list(combined.dtypes[combined.dtypes=="object"].index.values)
num_to_cats = ["BsmtHalfBath", "HalfBath", "KitchenAbvGr", "BsmtFullBath", "Fireplaces", "FullBath", "GarageCars",
               "BedroomAbvGr", "OverallCond", "OverallQual", "TotRmsAbvGrd", "MSSubClass", "YrSold", "MoSold", 
               "GarageYrBlt", "YearRemodAdd"]

for feat in num_to_cats:
    num_candidates.remove(feat)
    cat_candidates.append(feat)
    combined[feat] = combined[feat].astype("object")
    train[feat] = train[feat].astype("object")
    test[feat] = test[feat].astype("object")
num_candidates
len(num_candidates)
cat_candidates = combined.dtypes[combined.dtypes=="object"].index.values
cat_candidates
frequencies = []
for col in cat_candidates:
    overall_freq = combined.loc[:, col].value_counts().max() / combined.shape[0]
    frequencies.append([col, overall_freq])

frequencies = np.array(frequencies)
freq_df = pd.DataFrame(index=frequencies[:,0], data=frequencies[:,1], columns=["frequency"])
sorted_freq = freq_df.frequency.sort_values(ascending=False)

plt.figure(figsize=(20,5))
sns.barplot(x=sorted_freq.index[0:30], y=sorted_freq[0:30].astype(np.float), palette="Blues_r")
plt.xticks(rotation=90);
example = "Utilities"
combined.loc[:,example].value_counts()
example = "Street"
combined.loc[:,example].value_counts()
example = "Condition2"
combined.loc[:,example].value_counts()
cats_to_drop = ["Utilities"]
combined = combined.drop(cats_to_drop, axis=1)
train = train.drop(cats_to_drop, axis=1)
test = test.drop(cats_to_drop, axis=1)
cat_candidates = combined.dtypes[combined.dtypes=="object"].index.values
cat_candidates
def build_map(useless_levels, plugin_level, train_levels):
    plugin_map = {}
    for level in useless_levels:
        plugin_map[level] = plugin_level
    for level in train_levels:
        plugin_map[level] = level
    return plugin_map
def clean_test_levels(train, test):
    for col in test.columns:
        train_levels = set(train[col].unique())
        test_levels = set(test[col].unique())
        in_test_not_in_train = test_levels.difference(train_levels)
        if len(in_test_not_in_train)>0:
            close_to_mean_level = train.groupby(col).LogSalePrice.mean() - train.SalePrice.apply(np.log).mean()
            close_to_mean_level = close_to_mean_level.apply(np.abs)
            plugin_level = close_to_mean_level.sort_values().index.values[0]
            in_test_not_in_train = list(in_test_not_in_train)
            plugin_map = build_map(in_test_not_in_train, plugin_level, train_levels)
            test[col] = test[col].map(plugin_map)
    return train, test
#train, test = clean_test_levels(train, test)
test["MSSubClass"].value_counts()
outlier_ids = set()
outlier_ids = outlier_ids.union(set(train[train.LotArea > 60000].index.values))
outlier_ids = outlier_ids.union(set(train[train.LotFrontage > 200].index.values))
outlier_ids = outlier_ids.union(set(train[(train.LotFrontage > 150) & (train.SalePrice.apply(np.log) < 11)].index.values))
outlier_ids = outlier_ids.union(set(train[train.GrLivArea > 4500].index.values))
outlier_ids = outlier_ids.union(set(train[train["1stFlrSF"] > 4000].index.values))
outlier_ids = outlier_ids.union(set(train[train.MasVnrArea > 1400].index.values))
outlier_ids = outlier_ids.union(set(train[train["BsmtFinSF1"] > 5000].index.values))
outlier_ids = outlier_ids.union(set(train[train.TotalBsmtSF > 6000].index.values))
outlier_ids = outlier_ids.union(set(train[(train.OpenPorchSF > 500) & (np.log(train.SalePrice) < 11)].index.values))
outlier_ids
train.shape
train = train.drop(list(outlier_ids))
combined = combined.drop(list(outlier_ids))
train.shape
def impute_na_trees(df, col):
    if df[col].dtype == "object":
        df[col] = df[col].fillna("None")
        df[col] = df[col].astype("object")
    else:
        df[col] = df[col].fillna(0)
    return df

for col in combined.columns:
    combined = impute_na_trees(combined, col)
num_candidates = combined.dtypes[combined.dtypes!="object"].index.values
len(num_candidates)
cat_candidates = combined.dtypes[combined.dtypes=="object"].index.values
len(cat_candidates)
combined.isnull().sum().sum()
combined["TotalSF"] = combined["1stFlrSF"] + combined["2ndFlrSF"] + combined["TotalBsmtSF"] 
combined["GreenArea"] = combined["LotArea"] - combined["GrLivArea"] - combined["GarageArea"]
traindf_orig = combined.iloc[0:train.shape[0]].copy()
traindf_orig.loc[:, "target"] = train.LogSalePrice
testdf = combined.iloc[train.shape[0]::].copy()
# traindf, holddf = train_test_split(traindf_orig, test_size=0.25, random_state=0)
# print((traindf.shape, holddf.shape, testdf.shape))
# all_rmse = [ results['hold_rmse'] ]
# num_chunks = [1]
org_params = {
    'iterations': 10000,
    'learning_rate': 0.08,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': True,
    'loss_function': 'RMSE',
    'od_type': 'Iter',
    'od_wait': 1000,
    'one_hot_max_size': 20,
    'l2_leaf_reg': 100,
    'depth': 3,
    'rsm': 0.6,
    'random_strength': 2,
    'bagging_temperature': 10
}
num_rows = traindf_orig.shape[0]
num_chunks_array = []
rmse_hold = []
for num_chunks in range(1,11):
    print(f"running with {num_chunks} chunks")
    for chunk_id in range(num_chunks):
        chunk_start = math.floor(chunk_id*num_rows/num_chunks)
        chunk_end = math.ceil((chunk_id+1)*num_rows/num_chunks)
        traindf, holddf = train_test_split(traindf_orig.iloc[chunk_start:chunk_end], test_size=0.25, random_state=0)
        print((traindf.shape, holddf.shape, testdf.shape))
        if chunk_id == 0:
            results = run_catboost(traindf, 
                       testdf,
                       holddf,
                       org_params,
                       plot=False,
                       n_splits=3,
                       n_repeats=1)
        else:
            results = run_catboost(traindf, 
                       testdf,
                       holddf,
                       org_params,
                       plot=False,
                       n_splits=3,
                       n_repeats=1,
                       init_model = best_model)
        best_model = results['best_model']
        num_chunks_array.append(num_chunks)
        rmse_hold.append(results['hold_rmse'])
        print(f"{num_chunks} {results['hold_rmse']}")

plt.plot(num_chunks_array, rmse_hold)