import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
import warnings
warnings.filterwarnings("ignore")
%%time
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.head()
def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]
rows = 150_000
segments = int(np.floor(train.shape[0] / rows))

X_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min','q95','q99', 'q05','q01',
                                'abs_max', 'abs_mean', 'abs_std', 'trend', 'abs_trend'])
y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

for segment in tqdm(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    y_train.loc[segment, 'time_to_failure'] = y
    
    X_train.loc[segment, 'ave'] = x.mean()
    X_train.loc[segment, 'std'] = x.std()
    X_train.loc[segment, 'max'] = x.max()
    X_train.loc[segment, 'min'] = x.min()
    X_train.loc[segment, 'q95'] = np.quantile(x,0.95)
    X_train.loc[segment, 'q99'] = np.quantile(x,0.99)
    X_train.loc[segment, 'q05'] = np.quantile(x,0.05)
    X_train.loc[segment, 'q01'] = np.quantile(x,0.01)
    
    X_train.loc[segment, 'abs_max'] = np.abs(x).max()
    X_train.loc[segment, 'abs_mean'] = np.abs(x).mean()
    X_train.loc[segment, 'abs_std'] = np.abs(x).std()
    X_train.loc[segment, 'trend'] = add_trend_feature(x)
    X_train.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)
    
X_train.head()
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
for seg_id in tqdm(X_test.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
    X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)
    X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)
    X_test.loc[seg_id, 'q05'] = np.quantile(x,0.05)
    X_test.loc[seg_id, 'q01'] = np.quantile(x,0.01)
    
    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()
    X_test.loc[seg_id, 'abs_mean'] = np.abs(x).mean()
    X_test.loc[seg_id, 'abs_std'] = np.abs(x).std()
    X_test.loc[seg_id, 'trend'] = add_trend_feature(x)
    X_test.loc[seg_id, 'abs_trend'] = add_trend_feature(x, abs_values=True)

X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_train_scaled.head()
X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns)
X_test_scaled.head()
param = {'num_leaves': 31,
         'min_data_in_leaf': 32, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.001,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread": 4,
         "verbosity": -1}
features = X_train_scaled.columns
import time
import lightgbm as lgb

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(X_train_scaled))
predictions = np.zeros(len(X_test_scaled))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_scaled.values, y_train.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(X_train_scaled.iloc[trn_idx][features], label=y_train.iloc[trn_idx])
    val_data = lgb.Dataset(X_train_scaled.iloc[val_idx][features], label=y_train.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(X_train_scaled.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(X_test_scaled[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, y_train)**0.5))
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,16))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
%%time
import xgboost as xgb

xgb_params = {'eta': 0.001, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'alpha':0.1,
          'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'random_state':folds}


folds = KFold(n_splits=5, random_state=4520)
oof_xgb = np.zeros(len(X_train_scaled))
predictions_xgb = np.zeros(len(X_test_scaled))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_scaled.values, y_train.values)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(data=X_train_scaled.iloc[trn_idx][features], label=y_train.iloc[trn_idx])
    val_data = xgb.DMatrix(data=X_train_scaled.iloc[val_idx][features], label=y_train.iloc[val_idx])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("-" * 10 + "Xgboost " + str(fold_) + "-" * 10)
    num_round = 11000
    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=50, verbose_eval=1000)
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(X_train_scaled.iloc[val_idx][features]), ntree_limit=xgb_model.best_ntree_limit+50)

    predictions_xgb += xgb_model.predict(xgb.DMatrix(X_test_scaled[features]), ntree_limit=xgb_model.best_ntree_limit+50) / folds.n_splits
    
np.save('oof_xgb', oof_xgb)
np.save('predictions_xgb', predictions_xgb)
print("CV score: {:<8.5f}".format(mean_squared_error(oof_xgb, y_train)**0.5))
# %%time
# from catboost import CatBoostRegressor
# folds = KFold(n_splits=5, random_state=4520)
# oof_cat = np.zeros(len(X_train_scaled))
# predictions_cat = np.zeros(len(X_test_scaled))

# for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_scaled.values, y_train.values)):
#     print("fold n°{}".format(fold_ + 1))
#     trn_data, trn_y = X_train_scaled.iloc[trn_idx][features], y_train.iloc[trn_idx]
#     val_data, val_y = X_train_scaled.iloc[val_idx][features], y_train.iloc[val_idx]
#     print("-" * 10 + "Catboost " + str(fold_) + "-" * 10)
#     cb_model = CatBoostRegressor(iterations=8000, learning_rate=0.01, depth=8, l2_leaf_reg=20, bootstrap_type='Bernoulli',  eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
#     cb_model.fit(trn_data, trn_y, eval_set=(val_data, val_y), use_best_model=True, verbose=True,)
    
#     oof_cat[val_idx] = cb_model.predict(val_data)
#     predictions_cat += cb_model.predict(X_test_scaled[features]) / folds.n_splits
    
# np.save('oof_cat', oof_cat)
# np.save('predictions_cat', predictions_cat)
# np.sqrt(mean_squared_error(y_train.values, oof_cat))
train_stack = np.vstack([oof, oof_xgb]).transpose()
test_stack = np.vstack([predictions,predictions_xgb]).transpose()

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_stack = np.zeros(train_stack.shape[0])
predictions_stack = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, y_train)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y_train.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], y_train.iloc[val_idx].values

    print("-" * 10 + "Ridge Regression" + str(fold_) + "-" * 10)
#     cb_model = CatBoostRegressor(iterations=3000, learning_rate=0.1, depth=8, l2_leaf_reg=20, bootstrap_type='Bernoulli',  eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
#     cb_model.fit(trn_data, trn_y, eval_set=(val_data, val_y), cat_features=[], use_best_model=True, verbose=True)
    clf = BayesianRidge()
    clf.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf.predict(val_data)
    predictions_stack += clf.predict(test_stack) / 5


print("CV score: {:<8.5f}".format(mean_squared_error(oof, y_train)**0.5))
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['time_to_failure'] = predictions_stack
sample_submission.to_csv('Bayesian_Ridge_Stacking.csv', index=False)
sample_submission.shape