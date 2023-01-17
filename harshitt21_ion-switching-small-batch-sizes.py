import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, GridSearchCV, KFold



import xgboost

import lightgbm

import tqdm

import os,gc

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore')
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if(col != 'time'):

            col_type = df[col].dtype



            if col_type != object:

                c_min = df[col].min()

                c_max = df[col].max()

                if str(col_type)[:3] == 'int':

                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)  

                else:

                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                        df[col] = df[col].astype(np.float16)

                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                        df[col] = df[col].astype(np.float32)

                    else:

                        df[col] = df[col].astype(np.float64)

            else:

                df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train = pd.read_csv('../input/liverpool-ion-switching/train.csv')

test = pd.read_csv('../input/liverpool-ion-switching/test.csv')

print(f' Shape of train : {train.shape}')

print(f' Shape of test : {test.shape}')
train.head(10)
train.tail()
test.head(10)
train.describe()
fig, ax = plt.subplots(1,2, figsize=(20,6))

sns.countplot(train.open_channels, ax=ax[0])

sns.distplot(train.open_channels, ax=ax[1])
train['open_channels'].value_counts()
fig, ax = plt.subplots(6,2,figsize=(20,36))

ax = ax.flatten()

for i in range(11):

    sns.distplot(train[train['open_channels'] == i].signal, ax=ax[i]).set_title(f'Distribution of signal for {i} open channels')
describe_df = train.groupby(['open_channels']).signal.describe()

describe_df
_, ax = plt.subplots(4, 2, figsize=(20,24))

ax = ax.flatten()

for k,i in enumerate(describe_df.columns):

      sns.lineplot(describe_df.index, describe_df[i], ax=ax[k], lw=2).set_title(f'{i} signal vs open_channels')
train['signal_diff'] = train['signal'].diff()

train['open_channels_diff'] = train['open_channels'].diff()

train = train.fillna(0)

train
_, ax = plt.subplots(5,1,figsize=(20,30))

ax = ax.flatten()

for i in range(5):

    sns.scatterplot('signal_diff', 'open_channels_diff', data=train[int(1e6)*i:int(1e6)*i + int(1e5)], ax=ax[i]).set_title(f'Change in signal vs Change in open_channels (signal range {int(1e6)*i}:{int(1e6)*i + int(1e5)})')
_, ax = plt.subplots(10,2, figsize=(20, 60))

ax = ax.flatten()

k=0

for i in range(10):

    x = 500000

    sample = train.iloc[x*i:x*(1+i)]

    sns.distplot(sample.signal, ax=ax[k], color='g').set_title(f'Distribution of signal for batch {i+1}')

    sns.countplot(sample.open_channels, ax=ax[k+1]).set_title(f'Count of open_channels for batch {i+1}')

    k = k+2
plt.figure(figsize=(25,8))

sns.lineplot(train.time[:10000], train.signal[:10000])

sns.lineplot(train.time[:10000], train.open_channels[:10000])
plt.figure(figsize=(25,8))

sns.lineplot(train.time[100000:110000], train.signal[100000:110000])

sns.lineplot(train.time[100000:110000], train.open_channels[100000:110000])
plt.figure(figsize=(25,8))

sns.lineplot(train.time[200000:210000], train.signal[200000:210000])

sns.lineplot(train.time[200000:210000], train.open_channels[200000:210000])
plt.figure(figsize=(25,8))

sns.lineplot(train.time[1000000:1010000], train.signal[1000000:1010000])

sns.lineplot(train.time[1000000:1010000], train.open_channels[1000000:1010000])
plt.figure(figsize=(25,8))

sns.lineplot(train.time[2000000:2010000], train.signal[2000000:2010000])

sns.lineplot(train.time[2000000:2010000], train.open_channels[2000000:2010000])
plt.figure(figsize=(25,8))

sns.lineplot(train.time[3000000:3010000], train.signal[3000000:3010000])

sns.lineplot(train.time[3000000:3010000], train.open_channels[3000000:3010000])
plt.figure(figsize=(20,6))

sns.distplot(test.signal, bins=500, color='red')
test.signal.describe([0, .25, .5, .75, .98])
def add_group(data,size):

    rows_per_group=size

    groups =[]

    group_no=0

    

    for i in range(0,len(data),rows_per_group):

        groups.extend([group_no]*rows_per_group)

        group_no+=1

    print('Total Groups for size {}:'.format(size),len(set(groups)))

    

    groups=groups[:len(data)]

    return groups
data_df = pd.concat([train, test], sort=False).reset_index(drop=True)





# Create Groups (1k,2k,5k)

data_df['batch_1k'] = data_df[['time']].apply(lambda x:add_group(x,1000) )

data_df['batch_2k'] = data_df[['time']].apply(lambda x:add_group(x,2000)) 

data_df['batch_5k'] = data_df[['time']].apply(lambda x:add_group(x,5000) )
data_df = reduce_mem_usage(data_df)





batch_cols = [i for i in data_df.columns if 'batch' in i]

for i in batch_cols:



    data_df[f'signal_{i}_mean'] = data_df.groupby(i)['signal'].transform('mean')

    data_df[f'signal_{i}_median'] = data_df.groupby(i)['signal'].transform('median')

    data_df[f'signal_{i}_min'] = data_df.groupby(i)['signal'].transform('min')

    data_df[f'signal_{i}_max'] = data_df.groupby(i)['signal'].transform('max')

    data_df[f'signal_{i}_std'] = data_df.groupby(i)['signal'].transform('std')

    data_df[f'signal_{i}_skew'] = data_df.groupby(i)['signal'].transform('skew')



    data_df[f'signal_{i}_diff_max_min'] = data_df[f'signal_{i}_max'] - data_df[f'signal_{i}_min']

    data_df[f'signal_{i}_ratio_max_min'] = data_df[f'signal_{i}_max'] / data_df[f'signal_{i}_min']

    

    data_df[f'signal_{i}_shift_1'] = data_df.groupby(i).shift(1)['signal']

    data_df[f'signal_{i}_shift_-1'] = data_df.groupby(i).shift(-1)['signal']

    data_df[f'signal_{i}_shift_2'] = data_df.groupby(i).shift(2)['signal']

    data_df[f'signal_{i}_shift_-2'] = data_df.groupby(i).shift(-2)['signal']

    data_df[f'signal_{i}_shift_3'] = data_df.groupby(i).shift(3)['signal']

    data_df[f'signal_{i}_shift_-3'] = data_df.groupby(i).shift(-3)['signal']

    

    data_df[f'signal_{i}_rolling_W2_mean'] = data_df.groupby(i)['signal'].rolling(2).mean().reset_index(drop=True)

    data_df[f'signal_{i}_rolling_W10_mean'] = data_df.groupby(i)['signal'].rolling(10).mean().reset_index(drop=True)

    data_df[f'signal_{i}_rolling_W100_mean'] = data_df.groupby(i)['signal'].rolling(100).mean().reset_index(drop=True)





    data_df[f'signal_{i}_rolling_W2_median'] = data_df.groupby(i)['signal'].rolling(2).median().reset_index(drop=True)

    data_df[f'signal_{i}_rolling_W10_median'] = data_df.groupby(i)['signal'].rolling(10).median().reset_index(drop=True)

    data_df[f'signal_{i}_rolling_W100_median'] = data_df.groupby(i)['signal'].rolling(100).median().reset_index(drop=True)





    data_df[f'signal_{i}_rolling_W2_min'] = data_df.groupby(i)['signal'].rolling(2).min().reset_index(drop=True)

    data_df[f'signal_{i}_rolling_W10_min'] = data_df.groupby(i)['signal'].rolling(10).min().reset_index(drop=True)

    data_df[f'signal_{i}_rolling_W100_min'] = data_df.groupby(i)['signal'].rolling(100).min().reset_index(drop=True)





    data_df[f'signal_{i}_rolling_W2_max'] = data_df.groupby(i)['signal'].rolling(2).max().reset_index(drop=True)

    data_df[f'signal_{i}_rolling_W10_max'] = data_df.groupby(i)['signal'].rolling(10).max().reset_index(drop=True)

    data_df[f'signal_{i}_rolling_W100_max'] = data_df.groupby(i)['signal'].rolling(100).max().reset_index(drop=True)





    data_df[f'signal_{i}_rolling_W2_std'] = data_df.groupby(i)['signal'].rolling(2).std().reset_index(drop=True)

    data_df[f'signal_{i}_rolling_W10_std'] = data_df.groupby(i)['signal'].rolling(10).std().reset_index(drop=True)

    data_df[f'signal_{i}_rolling_W100_std'] = data_df.groupby(i)['signal'].rolling(100).std().reset_index(drop=True)



    data_df = reduce_mem_usage(data_df)
data_df.head()
train = data_df[data_df['train']==1]

test = data_df[data_df['train']==0]

train['open_channels'] = train['open_channels'].astype(int)
del data_df

gc.collect()
FEATURES = train.drop(['time', 'signal', 'open_channels', 'train', 'batch_1k',

       'batch_2k', 'batch_5k'],1).columns

y = train['open_channels']



submission = pd.DataFrame()

submission['time'] = test['time']

submission



KFOLD_SPLITS = 10

SHUFFLE = True

NUM_BOOST_ROUNDS = 2500

EARLY_STOPPING_ROUNDS = 50

VERBOSE_EVAL = 500
cv = KFold(n_splits=KFOLD_SPLITS, shuffle=SHUFFLE, random_state=21)

cv



params = {'learning_rate': 0.05,

          'max_depth': -1,

          'num_leaves': 2**8+1,

          'feature_fraction': 0.8,

          'bagging_fraction': 0.8,

          'objective':'regression',

          'metric':'rmse'

         }



oof_df = train[['signal', 'open_channels']].copy()

feature_importance_df = pd.DataFrame()



fold_ = 1

for train_idx, val_idx in cv.split(train[FEATURES], y):

    X_train, X_val = train[FEATURES].iloc[train_idx], train[FEATURES].iloc[val_idx]

    y_train, y_val = y[train_idx], y[val_idx]

    

    train_set = lightgbm.Dataset(X_train, y_train)

    val_set = lightgbm.Dataset(X_val, y_val)

    

    model = lightgbm.train(params,

                          train_set,

                          num_boost_round=NUM_BOOST_ROUNDS,

                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,

                          verbose_eval=VERBOSE_EVAL,

                          valid_sets=[train_set, val_set]

                          )

    

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)

    val_preds = np.round(np.clip(val_preds, 0, 10)).astype(int)

    

    test_preds = model.predict(test[FEATURES], num_iteration=model.best_iteration)

    test_preds = np.round(np.clip(test_preds, 0, 10)).astype(int)



    oof_df.loc[oof_df.iloc[val_idx].index, 'oof'] = val_preds

    submission[f'open_channels_fold{fold_}'] = test_preds

    

    f1 = f1_score(oof_df.loc[oof_df.iloc[val_idx].index]['open_channels'],

                  oof_df.loc[oof_df.iloc[val_idx].index]['oof'],

                            average = 'macro')

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = FEATURES

    fold_importance_df["importance"] = model.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    print(f'Fold {fold_} - validation f1: {f1:0.5f}')

    

    fold_ += 1



cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:100].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



best_features.sort_values(by="importance",ascending=False).to_csv('./FeatureIMP.csv', index=False)



print(f1_score(oof_df['open_channels'],

                    oof_df['oof'],

                    average = 'macro'))



submission['open_channels'] = submission.drop(['time'],1).median(axis=1).astype(int)

submission[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')