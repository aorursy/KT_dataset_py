import numpy as np 
import pandas as pd
from sklearn import *
import lightgbm as lgb
from sklearn.metrics import f1_score
from tqdm import tqdm
train = pd.read_csv("/kaggle/input/ionswitchingkl/datasets/trainK.csv")
test = pd.read_csv("/kaggle/input/ionswitchingkl/datasets/testK.csv")
def features(df):
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10_000) - 1).values
    df['batch'] = df.index // 50_000
    df['batch_index'] = df.index  - (df.batch * 50_000)
    df['batch_slices'] = df['batch_index']  // 5_000
    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)
    
    for c in ['batch','batch_slices2']:
        d = {}
        d['mean'+c] = df.groupby([c])['signal'].mean()
        d['median'+c] = df.groupby([c])['signal'].median()
        d['max'+c] = df.groupby([c])['signal'].max()
        d['min'+c] = df.groupby([c])['signal'].min()
        d['std'+c] = df.groupby([c])['signal'].std()
        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))
        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))
        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))
        for v in d:
            df[v] = df[c].map(d[v].to_dict())
        df['range'+c] = df['max'+c] - df['min'+c]
        df['maxtomin'+c] = df['max'+c] / df['min'+c]
        df['abs_avg'+c] = (df['abs_min'+c] + df['abs_max'+c]) / 2
    
    #add shifts
    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])
    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]
    df['signal_shift_+2'] = [0,0,] + list(df['signal'].values[:-2])
    df['signal_shift_-2'] = list(df['signal'].values[2:]) + [0,0]
    df['signal_shift_+3'] = [0,0,0,] + list(df['signal'].values[:-3])
    df['signal_shift_-3'] = list(df['signal'].values[3:]) + [0,0,0]
    for i in df[df['batch_index']==0].index:
        df['signal_shift_+1'][i] = np.nan
        df['signal_shift_+2'][i] = np.nan
        df['signal_shift_+3'][i] = np.nan
    for i in df[df['batch_index']==49999].index:
        df['signal_shift_-1'][i] = np.nan
        df['signal_shift_-2'][i] = np.nan
        df['signal_shift_-3'][i] = np.nan

    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:
        df[c+'_msignal'] = df[c] - df['signal']
        
    return df

train = features(train)
test = features(test)
WINDOWS=[10,50,100,500]
def create_rolling_features(df):
    for window in WINDOWS:
        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()
        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()
        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()
        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()
        df["rolling_min_max_ratio_" + str(window)] = df["rolling_min_" + str(window)] / df["rolling_max_" + str(window)]
        df["rolling_min_max_diff_" + str(window)] = df["rolling_max_" + str(window)] - df["rolling_min_" + str(window)]

    df = df.replace([np.inf, -np.inf], np.nan)    
    df.fillna(0, inplace=True)
    return df

train = create_rolling_features(train)
test = create_rolling_features(test)

#================Model building ===============================
col = [c for c in train.columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]
#x1, x2, y1, y2 = model_selection.train_test_split(train[col], train['open_channels'], test_size=0.1, random_state=7)

def f1_score_calc(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def lgb_Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average="macro")
    return ('KaggleMetric', score, True)
params = { 'n_estimators':1500,
          'boosting_type': 'gbdt',
          'max_depth' : 12,
          'nthread': 3, # Updated from nthread
          'num_leaves': 207,
          'learning_rate': 0.08,
          'max_bin': 200,
          'subsample_freq': 1,
           'lambda_l2': 0.10,
          'lambda_l1': 0.30,
          'min_split_gain': 0.06,
          'min_child_weight': 27,
          'scale_pos_weight': 1,
          'feature_fraction':0.93,
          'bagging_fraction':0.93,
          'min_data_in_leaf':21,
          'metric' : 'rmse'}
#=================prediction=============
X = train[col].values
Y = train['open_channels'].values
d_train = lgb.Dataset(X,Y)
model = lgb.train(params, d_train, 1500)
#Submission dataset
y_test = model.predict(test[col])

y_test = np.round(np.clip(y_test, 0, 10)).astype(int)
test["open_channels"] = y_test
test[['time','open_channels']].to_csv('submission_rollingM.csv', index=False, float_format='%.4f')
