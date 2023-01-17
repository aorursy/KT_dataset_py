%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
plt.style.use("ggplot")
import pandas as pd

import numpy as np
import random

np.random.seed(1234)

random.seed(1234)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
train_df.head()
train_df.dtypes
train_df["MSZoning"].value_counts()
all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
all_df
all_df["SalePrice"]
from sklearn.preprocessing import LabelEncoder
categories = all_df.columns[all_df.dtypes == "object"]

print(categories)
all_df["Alley"].value_counts()
for cat in categories:

    le = LabelEncoder()

    print(cat)

    

    all_df[cat].fillna("missing", inplace=True)    

    le = le.fit(all_df[cat])

    all_df[cat] = le.transform(all_df[cat])

    all_df[cat] = all_df[cat].astype("category")
all_df
train_df_le = all_df[~all_df["SalePrice"].isnull()]

test_df_le = all_df[all_df["SalePrice"].isnull()]
import lightgbm as lgb
from sklearn.model_selection import KFold

folds = 3

kf = KFold(n_splits=folds)
lgbm_params = {

    "objective":"regression",

    "random_seed":1234

}
train_X = train_df_le.drop(["SalePrice", "Id"], axis=1)

train_Y = train_df_le["SalePrice"]
from sklearn.metrics import mean_squared_error
models = []

rmses = []

oof = np.zeros(len(train_X))



for train_index, val_index in kf.split(train_X):

    X_train = train_X.iloc[train_index]

    X_valid = train_X.iloc[val_index]

    y_train = train_Y.iloc[train_index]

    y_valid = train_Y.iloc[val_index]

        

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)    

    

    model_lgb = lgb.train(lgbm_params, 

                          lgb_train, 

                          valid_sets=lgb_eval, 

                          num_boost_round=100,

                          early_stopping_rounds=20,

                          verbose_eval=10,

                         )    

    

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

    tmp_rmse = np.sqrt(mean_squared_error(np.log(y_valid), np.log(y_pred)))

    print(tmp_rmse)    

              

    models.append(model_lgb)    

    rmses.append(tmp_rmse)

    oof[val_index] = y_pred 
sum(rmses)/len(rmses)
for model in models:

    lgb.plot_importance(model,importance_type="gain", max_num_features=15)
train_df["SalePrice"].describe()
train_df["SalePrice"].plot.hist(bins=20)
np.log(train_df['SalePrice'])
np.log(train_df['SalePrice']).plot.hist(bins=20)
train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])
train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)

train_Y = train_df_le["SalePrice_log"]
models = []

rmses = []

oof = np.zeros(len(train_X))



for train_index, val_index in kf.split(train_X):

    X_train = train_X.iloc[train_index]

    X_valid = train_X.iloc[val_index]

    y_train = train_Y.iloc[train_index]

    y_valid = train_Y.iloc[val_index]

        

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)    

    

    model_lgb = lgb.train(lgbm_params, 

                          lgb_train, 

                          valid_sets=lgb_eval, 

                          num_boost_round=100,

                          early_stopping_rounds=20,

                          verbose_eval=10,

                         )    

    

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    print(tmp_rmse)    

              

    models.append(model_lgb)    

    rmses.append(tmp_rmse)

    oof[val_index] = y_pred 
sum(rmses)/len(rmses)
all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
categories = all_df.columns[all_df.dtypes == "object"]

print(categories)
all_df.isnull().sum().sort_values(ascending=False).head(40)
all_df.PoolQC.value_counts()
all_df.loc[~all_df["PoolQC"].isnull(), "PoolQC"] = 1

all_df.loc[all_df["PoolQC"].isnull(), "PoolQC"] = 0
all_df.PoolQC.value_counts()
all_df.loc[~all_df["MiscFeature"].isnull(), "MiscFeature"] = 1

all_df.loc[all_df["MiscFeature"].isnull(), "MiscFeature"] = 0
all_df.loc[~all_df["Alley"].isnull(), "Alley"] = 1

all_df.loc[all_df["Alley"].isnull(), "Alley"] = 0
HighFacility_col = ["PoolQC","MiscFeature","Alley"]

for col in HighFacility_col:

    if all_df[col].dtype == "object":

        if len(all_df[all_df[col].isnull()]) > 0:

            all_df.loc[~all_df[col].isnull(), col] = 1

            all_df.loc[all_df[col].isnull(), col] = 0
all_df["hasHighFacility"] = all_df["PoolQC"] + all_df["MiscFeature"] + all_df["Alley"]
all_df["hasHighFacility"] = all_df["hasHighFacility"].astype(int)
all_df["hasHighFacility"].value_counts()
all_df = all_df.drop(["PoolQC","MiscFeature","Alley"],axis=1)
all_df.describe().T
train_df_num = train_df.select_dtypes(include=[np.number])
nonratio_features = ["Id", "MSSubClass", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "MoSold", "YrSold"]
num_features = sorted(list(set(train_df_num) - set(nonratio_features)))
num_features
train_df_num_rs = train_df_num[num_features]
for col in num_features:

    if train_df_num_rs.describe()[col]["75%"] == 0:

        print(col, len(train_df_num_rs[train_df_num_rs[col] == 0]))
for col in num_features:

    if train_df_num_rs[col].nunique() < 15:

        print(col, train_df_num_rs[col].nunique())
for col in num_features:

    tmp_df = train_df_num_rs[(train_df_num_rs[col] > train_df_num_rs[col].mean() + train_df_num_rs[col].std()*3) | \

    (train_df_num_rs[col] < train_df_num_rs[col].mean() - train_df_num_rs[col].std()*3)]

    print(col, len(tmp_df))
all_df.plot.scatter(x="BsmtFinSF1", y="SalePrice")
all_df[all_df["BsmtFinSF1"] > 5000]
all_df.plot.scatter(x="TotalBsmtSF", y="SalePrice")
all_df[all_df["TotalBsmtSF"] > 6000]
all_df.plot.scatter(x="GrLivArea", y="SalePrice")
all_df[all_df["GrLivArea"] > 5000]
all_df.plot.scatter(x="1stFlrSF", y="SalePrice")
all_df[all_df["1stFlrSF"] > 4000]
all_df = all_df[(all_df['BsmtFinSF1'] < 2000) | (all_df['SalePrice'].isnull())]

all_df = all_df[(all_df['TotalBsmtSF'] < 3000) | (all_df['SalePrice'].isnull())]

all_df = all_df[(all_df['GrLivArea'] < 4500) | (all_df['SalePrice'].isnull())]

all_df = all_df[(all_df['1stFlrSF'] < 2500) | (all_df['SalePrice'].isnull())]

all_df = all_df[(all_df['LotArea'] < 100000) | (all_df['SalePrice'].isnull())]
categories = categories.drop(["PoolQC","MiscFeature","Alley"])
for cat in categories:

    le = LabelEncoder()

    print(cat)

    

    all_df[cat].fillna("missing", inplace=True)    

    le = le.fit(all_df[cat])

    all_df[cat] = le.transform(all_df[cat])

    all_df[cat] = all_df[cat].astype("category")
train_df_le = all_df[~all_df["SalePrice"].isnull()] 

test_df_le = all_df[all_df["SalePrice"].isnull()] 



train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])

train_X = train_df_le.drop(["SalePrice","SalePrice_log", "Id"], axis=1)

train_Y = train_df_le["SalePrice_log"]
models = []

rmses = []

oof = np.zeros(len(train_X))



for train_index, val_index in kf.split(train_X):

    X_train = train_X.iloc[train_index]

    X_valid = train_X.iloc[val_index]

    y_train = train_Y.iloc[train_index]

    y_valid = train_Y.iloc[val_index]

        

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)    

    

    model_lgb = lgb.train(lgbm_params, 

                          lgb_train, 

                          valid_sets=lgb_eval, 

                          num_boost_round=100,

                          early_stopping_rounds=20,

                          verbose_eval=10,

                         )    

    

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    print(tmp_rmse)    

              

    models.append(model_lgb)    

    rmses.append(tmp_rmse)

    oof[val_index] = y_pred 
sum(rmses)/len(rmses)
all_df[["YearBuilt","YearRemodAdd","GarageYrBlt","YrSold"]].describe()
all_df["Age"] = all_df["YrSold"] - all_df["YearBuilt"]
train_df_le = all_df[~all_df["SalePrice"].isnull()] 

test_df_le = all_df[all_df["SalePrice"].isnull()] 



train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])

train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)

train_Y = train_df_le["SalePrice_log"]
models = []

rmses = []

oof = np.zeros(len(train_X))



for train_index, val_index in kf.split(train_X):

    X_train = train_X.iloc[train_index]

    X_valid = train_X.iloc[val_index]

    y_train = train_Y.iloc[train_index]

    y_valid = train_Y.iloc[val_index]

        

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)    

    

    model_lgb = lgb.train(lgbm_params, 

                          lgb_train, 

                          valid_sets=lgb_eval, 

                          num_boost_round=100,

                          early_stopping_rounds=20,

                          verbose_eval=10,

                         )    

    

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    print(tmp_rmse)    

              

    models.append(model_lgb)    

    rmses.append(tmp_rmse)

    oof[val_index] = y_pred 
sum(rmses)/len(rmses)
all_df[["LotArea","MasVnrArea","BsmtUnfSF","TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea","WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "LotFrontage"]].describe()
all_df["TotalSF"] = all_df["TotalBsmtSF"] + all_df["1stFlrSF"] + all_df["2ndFlrSF"]

all_df["Total_Bathrooms"] = all_df["FullBath"] + all_df["HalfBath"] + all_df["BsmtFullBath"] + all_df["BsmtHalfBath"]
all_df["Total_PorchSF"] = all_df["WoodDeckSF"] + all_df["OpenPorchSF"] + all_df["EnclosedPorch"] + all_df["3SsnPorch"] + all_df["ScreenPorch"]
all_df["hasPorch"] = all_df["Total_PorchSF"].apply(lambda x: 1 if x > 0 else 0)

all_df = all_df.drop("Total_PorchSF",axis=1)
train_df_le = all_df[~all_df["SalePrice"].isnull()] 

test_df_le = all_df[all_df["SalePrice"].isnull()] 



train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])

train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)

train_Y = train_df_le["SalePrice_log"]
models = []

rmses = []

oof = np.zeros(len(train_X))



for train_index, val_index in kf.split(train_X):

    X_train = train_X.iloc[train_index]

    X_valid = train_X.iloc[val_index]

    y_train = train_Y.iloc[train_index]

    y_valid = train_Y.iloc[val_index]

        

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)    

    

    model_lgb = lgb.train(lgbm_params, 

                          lgb_train, 

                          valid_sets=lgb_eval, 

                          num_boost_round=100,

                          early_stopping_rounds=20,

                          verbose_eval=10,

                         )    

    

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    print(tmp_rmse)    

              

    models.append(model_lgb)    

    rmses.append(tmp_rmse)

    oof[val_index] = y_pred 
sum(rmses)/len(rmses)
import optuna
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=1234, shuffle=False,  stratify=None)
def objective(trial):

    params = {

        "objective":"regression",

        "random_seed":1234,

        "learning_rate":0.05,        

        "n_estimators":1000,        

        

        "num_leaves":trial.suggest_int("num_leaves",4,64),

        "max_bin":trial.suggest_int("max_bin",50,200),        

        "bagging_fraction":trial.suggest_uniform("bagging_fraction",0.4,0.9),

        "bagging_freq":trial.suggest_int("bagging_freq",1,10),

        "feature_fraction":trial.suggest_uniform("feature_fraction",0.4,0.9),

        "min_data_in_leaf":trial.suggest_int("min_data_in_leaf",2,16),                

        "min_sum_hessian_in_leaf":trial.suggest_int("min_sum_hessian_in_leaf",1,10),

    }

    

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)    

    

    model_lgb = lgb.train(params, lgb_train, 

                          valid_sets=lgb_eval, 

                          num_boost_round=100,

                          early_stopping_rounds=20,

                          verbose_eval=10,)    

    

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

    score =  np.sqrt(mean_squared_error(y_valid, y_pred))

    

    return score
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials=50)

study.best_params
lgbm_params = {

    "objective":"regression",

    "random_seed":1234,

    "learning_rate":0.05, 

    "n_estimators":1000,

    "num_leaves":12,

    "bagging_fraction": 0.8319278029616157,

    "bagging_freq": 5,

    "feature_fraction": 0.4874544371547538,    

    "max_bin":189, 

    "min_data_in_leaf":13, 

    "min_sum_hessian_in_leaf":4

}
models = []

rmses = []

oof = np.zeros(len(train_X))



for train_index, val_index in kf.split(train_X):

    X_train = train_X.iloc[train_index]

    X_valid = train_X.iloc[val_index]

    y_train = train_Y.iloc[train_index]

    y_valid = train_Y.iloc[val_index]

        

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)    

    

    model_lgb = lgb.train(lgbm_params, 

                          lgb_train, 

                          valid_sets=lgb_eval, 

                          num_boost_round=100,

                          early_stopping_rounds=20,

                          verbose_eval=10,

                         )    

    

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    print(tmp_rmse)    

              

    models.append(model_lgb)    

    rmses.append(tmp_rmse)

    oof[val_index] = y_pred 
sum(rmses)/len(rmses)
test_X = test_df_le.drop(["SalePrice", "Id"], axis=1)
preds = []



for model in models:

    pred = model.predict(test_X)

    preds.append(pred)
preds_array = np.array(preds)

preds_mean = np.mean(preds_array, axis=0)
preds_exp = np.exp(preds_mean)
len(preds_exp)
submission["SalePrice"] = preds_exp
submission.to_csv("./houseprices_submit01.csv",index=False)
from sklearn.ensemble import RandomForestRegressor as rf
hasnan_cat = []

for col in all_df.columns:

    tmp_null_count = all_df[col].isnull().sum()

    if (tmp_null_count > 0) & (col != "SalePrice"):

        print(col, tmp_null_count)

        hasnan_cat.append(col)
all_df[hasnan_cat].describe()
for col in all_df.columns:

    tmp_null_count = all_df[col].isnull().sum()

    if (tmp_null_count > 0) & (col != "SalePrice"):

        print(col, tmp_null_count)

        all_df[col] = all_df[col].fillna(all_df[col].median())
train_df_le = all_df[~all_df["SalePrice"].isnull()]

test_df_le = all_df[all_df["SalePrice"].isnull()]

train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])
train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)

train_Y = train_df_le["SalePrice_log"]
folds = 3

kf = KFold(n_splits=folds)
models_rf = []

rmses_rf = []

oof_rf = np.zeros(len(train_X))

for train_index, val_index in kf.split(train_X):

    X_train = train_X.iloc[train_index]

    X_valid = train_X.iloc[val_index]

    y_train = train_Y.iloc[train_index]

    y_valid = train_Y.iloc[val_index]

    model_rf = rf(

        n_estimators=50,

        random_state=1234

    )

    model_rf.fit(X_train, y_train)

    y_pred = model_rf.predict(X_valid)

    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    print(tmp_rmse)

    models_rf.append(model_rf)

    rmses_rf.append(tmp_rmse)

    oof_rf[val_index] = y_pred
sum(rmses_rf)/len(rmses_rf)
test_X = test_df_le.drop(["SalePrice","Id"], axis=1)
preds_rf = []

for model in models_rf:

    pred = model.predict(test_X)

    preds_rf.append(pred)
preds_array_rf = np.array(preds_rf)

preds_mean_rf = np.mean(preds_array_rf, axis=0)

preds_exp_rf = np.exp(preds_mean_rf)

submission["SalePrice"] = preds_exp_rf
submission.to_csv("./houseprices_submit02.csv",index=False)
import xgboost as xgb
categories = train_X.columns[train_X.dtypes == "category"]
for col in categories:

    train_X[col] = train_X[col].astype("int8")

    test_X[col] = test_X[col].astype("int8")
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=1234, shuffle=False,  stratify=None)
def objective(trial):

    xgb_params = {

    "learning_rate":0.05,

    "seed":1234,        

    "max_depth":trial.suggest_int("max_depth",3,16),

    "colsample_bytree":trial.suggest_uniform("colsample_bytree",0.2,0.9),

    "sublsample":trial.suggest_uniform("sublsample",0.2,0.9),

    }

    xgb_train = xgb.DMatrix(X_train, label=y_train)

    xgb_eval = xgb.DMatrix(X_valid, label=y_valid)

    evals = [(xgb_train, "train"), (xgb_eval, "eval")]

    model_xgb = xgb.train(xgb_params, xgb_train,

    evals=evals,

    num_boost_round=1000,

    early_stopping_rounds=20,

    verbose_eval=10,)

    y_pred = model_xgb.predict(xgb_eval)

    score = np.sqrt(mean_squared_error(y_valid, y_pred))

    return score
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials=50)

study.best_params
xgb_params = {

"learning_rate":0.05,

"seed":1234,

"max_depth": 6,

"colsample_bytree": 0.330432640328732,

"sublsample": 0.7158427239902707

}
models_xgb = []

rmses_xgb = []

oof_xgb = np.zeros(len(train_X))

for train_index, val_index in kf.split(train_X):

    X_train = train_X.iloc[train_index]

    X_valid = train_X.iloc[val_index]

    y_train = train_Y.iloc[train_index]

    y_valid = train_Y.iloc[val_index]

    xgb_train = xgb.DMatrix(X_train, label=y_train)

    xgb_eval = xgb.DMatrix(X_valid, label=y_valid)

    evals = [(xgb_train, "train"), (xgb_eval, "eval")]

    model_xgb = xgb.train(xgb_params, xgb_train,

    evals=evals,

    num_boost_round=1000,

    early_stopping_rounds=20,

    verbose_eval=20,)

    y_pred = model_xgb.predict(xgb_eval)

    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    print(tmp_rmse)

    models_xgb.append(model_xgb)

    rmses_xgb.append(tmp_rmse)

    oof_xgb[val_index] = y_pred
sum(rmses_xgb)/len(rmses_xgb)
xgb_test = xgb.DMatrix(test_X)
preds_xgb = []

for model in models_xgb:

    pred = model.predict(xgb_test)

    preds_xgb.append(pred)
preds_array_xgb= np.array(preds_xgb)

preds_mean_xgb = np.mean(preds_array_xgb, axis=0)

preds_exp_xgb = np.exp(preds_mean_xgb)

submission["SalePrice"] = preds_exp_xgb
submission.to_csv("./houseprices_submit03.csv",index=False)
preds_ans = preds_exp_xgb * 0.5 + preds_exp * 0.5
submission["SalePrice"] = preds_ans
submission.to_csv("./houseprices_submit04.csv",index=False)
train_df_le_dn = train_df_le.dropna()
train_df_le_dn
from sklearn import preprocessing
train_scaled = preprocessing.scale(train_df_le_dn.drop(["Id"],axis=1))
train_scaled
train_scaled_df = pd.DataFrame(train_scaled)

train_scaled_df.columns = train_df_le_dn.drop(["Id"],axis=1).columns
train_scaled_df
from sklearn.cluster import KMeans
np.random.seed(1234)
house_cluster = KMeans(n_clusters=4).fit_predict(train_scaled)
train_scaled_df["km_cluster"] = house_cluster
train_scaled_df["km_cluster"].value_counts()
cluster_mean = train_scaled_df[["km_cluster","SalePrice","TotalSF","OverallQual","Age","Total_Bathrooms","YearRemodAdd","GarageArea",

                                "MSZoning","OverallCond","KitchenQual","FireplaceQu"]].groupby("km_cluster").mean().reset_index()
cluster_mean = cluster_mean.T
cluster_mean
cluster_mean[1:].plot(figsize=(12,10), kind="barh" , subplots=True, layout=(1, 4) , sharey=True)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

house_pca = pca.fit(train_scaled).transform(train_scaled)
house_pca
house_pca_df = pd.DataFrame(house_pca)

house_pca_df.columns = ["pca1","pca2"]
train_scaled_df = pd.concat([train_scaled_df, house_pca_df], axis=1)
train_scaled_df
my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for cl in train_scaled_df['km_cluster'].unique():

    plt.scatter(train_scaled_df.loc[train_scaled_df["km_cluster"] == cl ,'pca1'], train_scaled_df.loc[train_scaled_df["km_cluster"] == cl ,'pca2'], label=cl, c=my_colors[cl], alpha=0.6)

plt.legend()

plt.show()
pca_comp_df = pd.DataFrame(pca.components_,columns=train_scaled_df.drop(["km_cluster","pca1","pca2"],axis=1).columns).T

pca_comp_df.columns = ["pca1","pca2"]
pca_comp_df
train_df_le['SalePrice'].plot.hist(bins=20)
train_df_le['SalePrice'].describe()
train_df['SalePrice'].quantile(0.9)
train_df_le.loc[train_df["SalePrice"] >= 278000, "high_class"] = 1
train_df_le["high_class"] = train_df_le["high_class"].fillna(0)
train_df_le.head()
!pip install pydotplus
from sklearn import tree

import pydotplus

from six import StringIO
tree_x = train_df_le[["TotalSF","OverallQual","Age","GrLivArea","GarageCars","Total_Bathrooms","GarageType",

"YearRemodAdd","GarageArea","CentralAir","MSZoning","OverallCond","KitchenQual","FireplaceQu","1stFlrSF"]]

tree_y = train_df_le[["high_class"]]
clf = tree.DecisionTreeClassifier(max_depth=4)

clf = clf.fit(tree_x, tree_y)
dot_data = StringIO()

tree.export_graphviz(clf, out_file=dot_data,feature_names=tree_x.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
from IPython.display import Image

Image(graph.create_png())