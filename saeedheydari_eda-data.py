import os
print(os.listdir("../input"))

import gc
import os
import tqdm
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
IS_LOCAL = False
if(IS_LOCAL):
    PATH="../input/LANL/"
else:
    PATH="../input/"
os.listdir(PATH)
%%time
train_df = pd.read_csv(PATH+'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))
pd.options.display.precision = 15
train_df.head(10)
train_ad_sample_df = train_df['acoustic_data'].values[::100]
train_ttf_sample_df = train_df['time_to_failure'].values[::100]

def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(train_ad_sample_df, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_ttf_sample_df, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)

plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)
del train_ad_sample_df
del train_ttf_sample_df
train_ad_sample_df = train_df['acoustic_data'].values[:6291455]
train_ttf_sample_df = train_df['time_to_failure'].values[:6291455]
plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% of data")
del train_ad_sample_df
del train_ttf_sample_df


rows = 150000
segments = int(np.floor(train_df.shape[0] / rows))
print("Number of segments: ", segments)


shifts = 10
total_segments = segments * shifts
shift = int(rows/shifts)
print("Total number of segments: ", total_segments)
print("Shift size:", shift)
train_columns = ['mean', 'std', 'var', 'max', 'min', 'abs_max', 'A0', 
                       'Amean', 'Astd', 'Amax', 'Amin',                        
                       'Rmean', 'Rstd', 'Rmax', 'Rmin',
                       'Imean', 'Istd', 'Imax', 'Imin']
X_train = pd.DataFrame(index=range(total_segments), dtype=np.float64,
                       columns=train_columns)
y_train = pd.DataFrame(index=range(total_segments), dtype=np.float64,
                       columns=['time_to_failure'])
def create_features(seg_id, seg, X):
    xc = seg['acoustic_data'].values
    zc = np.fft.fft(xc)
        
    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'var'] = xc.var()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()
    X.loc[seg_id, 'abs_max'] = np.abs(xc).max()
    X.loc[seg_id, 'A0'] = abs(zc[0])
    X.loc[seg_id, 'Amean'] = np.abs(zc).mean()
    X.loc[seg_id, 'Astd'] = np.abs(zc).std()
    X.loc[seg_id, 'Amax'] = np.abs(zc).max()
    X.loc[seg_id, 'Amin'] = np.abs(zc).min()
    X.loc[seg_id, 'Rmean'] = np.real(zc).mean()
    X.loc[seg_id, 'Rstd'] = np.real(zc).std()
    X.loc[seg_id, 'Rmax'] = np.real(zc).max()
    X.loc[seg_id, 'Rmin'] = np.real(zc).min()
    X.loc[seg_id, 'Imean'] = np.imag(zc).mean()
    X.loc[seg_id, 'Istd'] = np.imag(zc).std()
    X.loc[seg_id, 'Imax'] = np.imag(zc).max()
    X.loc[seg_id, 'Imin'] = np.imag(zc).min()
for seg_id in tqdm(range(total_segments)):
    seg = train_df.iloc[seg_id*shift:seg_id*shift+rows]
    create_features(seg_id, seg, X_train)
    y_train.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
X_train.head(10)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_train_scaled.head(10)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
for seg_id in tqdm(X_test.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    create_features(seg_id, seg, X_test)
      
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_test_scaled.head(10)
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=2019)
params = {'num_leaves': 51,
         'min_data_in_leaf': 10, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.05,
         "min_child_samples": 40,
         "boosting": "gbdt",
         "feature_fraction": 0.7,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 2019}
oof = np.zeros(len(X_train_scaled))
predictions = np.zeros(len(X_test_scaled))
feature_importance_df = pd.DataFrame()
#run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_scaled,y_train.values)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    
    X_tr, X_val = X_train_scaled.iloc[trn_idx], X_train_scaled.iloc[val_idx]
    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    model = lgb.LGBMRegressor(**params, n_estimators = 5000, n_jobs = -1)
    model.fit(X_tr, y_tr, 
                    eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
                    verbose=200, early_stopping_rounds=200)
    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)
    #feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    #predictions
    predictions += model.predict(X_test_scaled, num_iteration=model.best_iteration_) / folds.n_splits
cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:50].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]
plt.figure(figsize=(6,5))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
submission['time_to_failure'] = predictions
submission.to_csv('submission.csv')