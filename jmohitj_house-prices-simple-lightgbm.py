import numpy as np 
import pandas as pd 
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(5)
## combine the datasets 
train['is_train'] = 1
test['is_train'] = 0
test['SalePrice'] = 0.0

combined = pd.concat([train, test])
print (train.shape)
print (test.shape)
print (combined.shape )
miss = combined.isna().sum(axis=0).to_frame().rename(columns={0:"count"})
miss[miss['count'] > 0].sort_values("count", ascending = False)
## Maynot drop any missing columns, just impute with something  
drop_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
combined[drop_columns] = combined[drop_columns].fillna("NA")
miss = combined.isna().sum(axis=0).to_frame().rename(columns={0:"count"})
miss[miss['count'] > 0].sort_values("count", ascending = False)
## Fill the LogFrontage Values using following table 
lookup = combined.groupby(['LotConfig', 'LotShape']).agg({"LotFrontage" : "mean"}).reset_index()
def fill_lf(r):
    if str(r['LotFrontage']).lower() != "nan":
        return r['LotFrontage']
    else:
        return lookup[(lookup["LotConfig"] == r["LotConfig"]) & (lookup["LotShape"] == r["LotShape"])]["LotFrontage"].iloc(0)[0]
combined['LotFrontage'] = combined.apply(lambda r : fill_lf(r), axis = 1)

combined['GaragePresent'] = combined["GarageArea"].apply(lambda x : 1 if x>0.0 else 0 )
combined['BasementPresent'] = combined["TotalBsmtSF"].apply(lambda x : 1 if x>0.0 else 0 )
cols_to_fix = [c for c in combined.columns if "Garage" in c]
cols_to_fix += [c for c in combined.columns if "Bsmt" in c]
cols_to_fix += [c for c in combined.columns if "Mas" in c]

for c in cols_to_fix:
    if combined[c].dtype == "object":
        combined[c] = combined[c].fillna("NA")
    else:
        combined[c] = combined[c].fillna(0.0)
miss = combined.isna().sum(axis=0).to_frame().rename(columns={0:"count"})
miss = miss[miss['count'] > 0].sort_values("count", ascending = False)
miss
for c in miss.index:
    combined[c] = combined[c].fillna(combined[c].mode().iloc(0)[0])
miss = combined.isna().sum(axis=0).to_frame().rename(columns={0:"count"})
miss = miss[miss['count'] > 0].sort_values("count", ascending = False)
miss
ignore_cols = ['SalePrice','is_train', 'target', 'Id']
num_cols = combined._get_numeric_data().columns
num_cols = [f for f in num_cols if f not in ignore_cols]
cat_cols = [f for f in combined.columns if f not in num_cols]
cat_cols = [f for f in cat_cols if f not in ignore_cols]
features = num_cols + cat_cols
check =[ ]
for c in combined._get_numeric_data().columns:
    if combined[c].var() < 1.0:
        check.append (c)
        
check = []
for c in cat_cols:
    if len(combined[c].value_counts()) < 3:
        check.append (c)
        
combined = combined.drop(['Street', "Utilities"], axis=1)
features.remove("Street")
features.remove("Utilities")
combined.head()
import seaborn as sns 
import matplotlib.pyplot as plt 

plt.figure(figsize=(12,12))
sns.heatmap(combined[features].corr())
from scipy.stats import norm, skew #for some statistics
from scipy.special import boxcox1p

skewed_feats = combined[num_cols].apply(lambda x: skew(x)).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[abs(skewness) > 0.75]
skewed_features = skewness.index

lam = 0.15
for c in skewed_features:
    combined[c] = boxcox1p(combined[c], lam)
combined.head()
## Numerical Features : Standard Scaler 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scl = RobustScaler()
combined[num_cols] = scl.fit_transform(combined[num_cols].values)
updated_combined = pd.get_dummies(combined[features + ["is_train", "SalePrice"]])
updated_features = [f for f in updated_combined.columns if f not in ["is_train", "SalePrice"]]
updated_combined[updated_features].head()
# ## Categorical Features - Label Encoding / OneHot Encoding / MeanEncoding 
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# le = LabelEncoder()
# for c in cat_cols:
#     combined[c] = le.fit_transform(combined[c].values)
def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    for t, v in groups.items():
        dcols = frame[v].to_dict()

        vs = list(dcols.values())
        ks = list(dcols.keys())
        lvs = len(vs)

        for i in range(lvs):
            for j in range(i+1,lvs):
                if vs[i] == vs[j]: 
                    dups.append(ks[i])
                    break

    return dups     

dupcols = duplicate_columns(updated_combined[updated_features])
for c in dupcols:
    updated_features.remove(c)
##### Not working will -- will use only K fold CV

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()

# feats = [c for c in combined.columns if c not in ['is_train', 'Id', 'SalePrice']]
# model.fit(combined[feats], combined['is_train'])

# preds = model.predict_proba(combined[feats])
# model.classes_

# combined['adval'] = preds[:,0]

# val_x = combined[combined['is_train'] == 1].sort_values("adval").head(int(len(train) * 0.20))
# train_x = combined[combined['is_train'] == 1].sort_values("adval").tail(len(train) - int(len(train) * 0.20))
# test_x = combined[combined['is_train'] == 0]

# drops = ['is_train', 'adval', 'Id']
# val_x = val_x.drop(drops, axis = 1)
# train_x = train_x.drop(drops, axis = 1)
# test_x = test_x.drop(drops, axis = 1)

# print (train_x.shape, val_x.shape, test_x.shape)
## Training a Simple LGB 
# feats = [f for f in combined.columns if f not in ['Id', 'SalePrice', 'is_train']]

## LightGBM with K-Fold Cross Validation 
import lightgbm as lgb 
import numpy as np 

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def run_lgb(train_X, train_y, val_X, val_y):
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'verbose': -1,
        'seed': 3,
        'learning_rate': 0.06,
        'bagging_seed' : 3,
        
        'subsample': 0.9691,
        'colsample_bytree':  0.4415,
        'max_depth': 3,
        'num_leaves': 6,
        'reg_alpha': 0.05,
        'reg_lambda': 1.05,
         }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 400, valid_sets=[lgval, lgtrain], early_stopping_rounds=50, verbose_eval=100)
    return model

train_x = updated_combined[updated_combined['is_train'] == 1][updated_features]
test_x = updated_combined[updated_combined['is_train'] == 0][updated_features]
train_y = np.log1p( updated_combined[updated_combined['is_train'] == 1]["SalePrice"] )


## Single Run 

# model = run_lgb(train_x, train_y), val_x, val_x)
# lgb_val1 = model.predict(val_x[feats], num_iteration=model.best_iteration)

# import matplotlib.pyplot as plt 
# fig, ax = plt.subplots(figsize=(12,18))
# lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
# ax.grid(False)
# plt.title("LightGBM - Feature Importance", fontsize=15)
# plt.show()
from sklearn.model_selection import KFold
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True)

# tst_pred_lgb = pd.DataFrame()
# split = 0
# for train_index, test_index in kf.split(train_x):
#     split += 1 
#     print ("Split : ", split)
#     X_train, X_dev = train_x.iloc[train_index], train_x.iloc[test_index]
#     y_train, y_dev = train_y.iloc[train_index], train_y.iloc[test_index]
    
#     ## LightGBM
#     model_lgb = run_lgb(X_train, y_train, X_dev, y_dev)

from sklearn.model_selection import cross_val_score


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x.values)
    rmse = np.sqrt(-cross_val_score(model, train_x.values, train_y, scoring="neg_mean_squared_error", cv = kf))
    return (rmse)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=6,
                              learning_rate=0.06, n_estimators=600,
                              max_bin = 55, bagging_fraction = 0.9691,
                              bagging_freq = 5, feature_fraction = 0.4415,
                              max_depth = 3, reg_alpha=0.05, reg_lambda=1.05,
                              feature_fraction_seed=9, bagging_seed=9)

score = rmsle_cv(model_lgb)
score.mean()
def fp(mod, tr, y, ts):
    mod.fit(tr, y)
    train_pred = mod.predict(tr)
    test_pred = np.expm1(mod.predict(ts))
    print (rmsle(y, train_pred))
    return test_pred

lgb_preds = fp(model_lgb, train_x, train_y, test_x)
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

def lgb_eval(num_leaves=8, 
             max_depth=15,
             bagging_fraction=0.5, 
             feature_fraction=0.5,
             lambda_l1=0.0,
             lambda_l2=0.0):
    
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "bagging_seed" : 42,
        "verbosity" : -1,
        "num_threads" : 4,

        "num_leaves" : int(num_leaves),
        "max_depth" : int(max_depth),
        "bagging_fraction" : bagging_fraction,
        "feature_fraction" : feature_fraction,
        "lambda_l2" : lambda_l2,
        "lambda_l1" : lambda_l1,
        # "min_child_samples" : int(min_child_samples),

        "learning_rate" : 0.08,
        "subsample_freq" : 5,
    }

    lgtrain = lgb.Dataset(train_x, label = train_y) 
    cv_result = lgb.cv(params, lgtrain, 1000, early_stopping_rounds=50, stratified=False, nfold=5, metrics=['rmse'])
    return -cv_result['rmse-mean'][-1]


def param_tuning(init_points, num_iter, **args):
    lgbBO = BayesianOptimization(lgb_eval, {
                                            'num_leaves' : (6, 28),
                                            'max_depth': (3, 16),
                                            'bagging_fraction': (0.1, 1.0),
                                            'feature_fraction': (0.1, 1.0),
                                            'lambda_l1' : (0.0, 5.0),
                                            'lambda_l2' : (0.0, 5.0)
                                            })

    lgbBO.maximize(init_points=init_points, n_iter=num_iter, **args)
    return lgbBO

result = param_tuning(5, 40)
# result.res['max']#['max_params']
## Can Fine Tune More -- Later
import xgboost as xgb

def run_xgb(train_X, train_y, val_X, val_y):
    params = {
              'objective': 'reg:linear', 
              'eval_metric': 'rmse',
              'random_state': 42, 
              'silent': True,
        
              'eta': 0.08,
              'max_depth': 3, 
              'subsample': 0.8634, 
              'colsample_bytree': 0.3623,
              'alpha' : 0.1017,
              'lambda' : 0.2852
            }
    
    tr_data = xgb.DMatrix(train_X, train_y)
    val_data = xgb.DMatrix(val_X, val_y)
    watchlist = [(tr_data, 'train'), (val_data, 'valid')]
    model_xgb = xgb.train(params, tr_data, 1000, watchlist, maximize=False, early_stopping_rounds = 50, verbose_eval=50)
    return model_xgb

# tst_pred_xgb = pd.DataFrame()
# split = 0
# for train_index, test_index in kf.split(train_x):
#     split += 1 
#     print ("Split : ", split)
#     X_train, X_dev = train_x.iloc[train_index], train_x.iloc[test_index]
#     y_train, y_dev = train_y.iloc[train_index], train_y.iloc[test_index]
    
#     model_xgb = run_xgb(X_train, y_train, X_dev, y_dev)
#     tst_pred_xgb[split] = model_xgb.predict(xgb.DMatrix(test_x), ntree_limit=model_xgb.best_ntree_limit)  
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score = rmsle_cv(model_xgb)
score.mean()
xgb_preds = fp(model_xgb, train_x, train_y, test_x)
def xgb_eval(max_depth, subsample, colsample_bytree, alpha, lambda_):
    params = {
              'objective': 'reg:linear', 
              'eval_metric': 'rmse',
              'random_state': 42, 
              'silent': True,
        
              'eta': 0.08,
              'max_depth': int(max_depth), 
              'subsample': max(min(subsample, 1), 0), 
              'colsample_bytree': max(min(colsample_bytree, 1), 0),                
              'alpha' : alpha,
              'lambda' : lambda_
            }
    dtrain = xgb.DMatrix(train_x, train_y)
    cv_result = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5)
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]


def param_tuning_xgb(init_points, num_iter):
    xgbBO = BayesianOptimization(xgb_eval, {
                                            'max_depth': (3, 16),
                                            'subsample': (0.3, 1.0),
                                            'colsample_bytree': (0.3, 1.0),
                                            'alpha' : (0.0, 5.0),
                                            'lambda_' : (0.0, 5.0)
                                            })
    
    xgbBO.maximize(init_points=init_points, n_iter=num_iter, acq='ei', xi=0.0)
    return xgbBO

# result = param_tuning_xgb(5, 30)
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
predDF = pd.DataFrame()
predDF['lgb'] = lgb_preds
predDF['xgb'] = xgb_preds

sub = pd.read_csv("../input/sample_submission.csv")
sub['SalePrice'] = predDF.mean(axis=1)
sub.to_csv("submission.csv", index = False)
sub.head()
# - pass categorical features directle 

# ## ideas to try : 
# - Dataset PreProcessing : V1
# - Feature Engineering : V1
    # - Check Outliers

# - Model Tuninig : V1
# - Stacking / Blending