import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as stat

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score



import warnings

warnings.filterwarnings("ignore")



n_leaves = 32

depth = 6

n_shift = 2
train = pd.read_csv('../input/data-without-drift-with-kalman-filter/train.csv')

test = pd.read_csv('../input/data-without-drift-with-kalman-filter/test.csv')

submission = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        if col != 'time':

            col_type = df[col].dtypes

            if col_type in numerics:

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
def shift(df, n):

    for i in range(1,n+1):

        df['signal-' + str(i)] = [0]*i + list(df['signal'].values[:-i])

        df['signal+' + str(i)] = list(df['signal'].values[i:]) + [0]*i

    return df
t = train[:500000]

outlier = list(t[(t.signal>-1)&(t.open_channels==0)].index)

train.at[outlier, 'signal'] = -1.6



t = train[500000:1000000]

outlier = list(t[(t.signal>0)&(t.open_channels==0)].index)

train.at[outlier, 'signal'] = -1.6



del t
for i in range(6):

    fig,axes=plt.subplots(1,2)

    df1 = pd.concat([train[:3500000], train[4000000:]], axis=0)

    df2 = pd.concat([train[:2000000], train[2500000:3500000], train[4000000:4500000]], axis=0)

    vec1=df1[df1.open_channels == i].signal

    vec2=df2[df2.open_channels == i].signal

    sns.distplot(vec1,bins=100,ax=axes[0], kde =False).set_title('open_channels={0}'.format(i))

    sns.distplot(vec2,bins=100,ax=axes[1], kde =False).set_title('open_channels={0}'.format(i))

del df1, df2
dic_model = {}

d = {}

for i in range(10):

    d['batch{0}'.format(i)] = shift(train[500000*i:500000*(i+1)], n_shift).drop('time',axis=1)

df1 = pd.concat([d['batch0'],d['batch1'],d['batch2'],d['batch3'],d['batch5'],d['batch6'],d['batch8']], axis=0).sample(frac=1)

df2 = pd.concat([d['batch4'], d['batch9']], axis=0)
stat_info = {}

for val in df1.open_channels.value_counts().index:

    stat_info['std_g1c{0}'.format(val)] = df1[df1.open_channels == val].signal.std()

    stat_info['mean_g1c{0}'.format(val)] = df1[df1.open_channels == val].signal.mean()

for val in df2.open_channels.value_counts().index:

    stat_info['std_g2c{0}'.format(val)] = df2[df2.open_channels == val].signal.std()

    stat_info['mean_g2c{0}'.format(val)] = df2[df2.open_channels == val].signal.mean()

def normal_distribution_feature(df, group):   

    cols = df.columns

    for col in df.columns:

        if col != 'open_channels':

            if group == 1:

                for val in range(6):

                    df[col+'_norm_c{0}'.format(val)] = stat.norm(stat_info['mean_g1c{0}'.format(val)],stat_info['std_g1c{0}'.format(val)]).pdf(df[col])

            else:

                for val in range(1,11):

                    df[col+'_norm_c{0}'.format(val)] = stat.norm(stat_info['mean_g2c{0}'.format(val)],stat_info['std_g2c{0}'.format(val)]).pdf(df[col])



             

    return df
df1 = reduce_mem_usage(normal_distribution_feature(df1, 1))

df2 = reduce_mem_usage(normal_distribution_feature(df2, 2))
def fit_lgb(X, y, upper_bound):

    lomodel = []

    kf = KFold(n_splits=5, random_state=1, shuffle=False)

    

    params = {'objective': 'huber',

              'num_leaves': n_leaves,

              'max_depth':depth,

              "metric": 'rmse',

              'n_jobs': -1,

              'random_state': 1

              }

    

    lopred=[]

    for train_index, test_index in kf.split(X):

        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]

        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

    

        d_train = lgb.Dataset(X_train, label=y_train)

        d_valid = lgb.Dataset(X_valid, label=y_valid)    

        

        model = lgb.train(params,

                          train_set=d_train,

                          num_boost_round=1000,

                          valid_sets=d_valid,

                          verbose_eval='None',

                          early_stopping_rounds=30,

                          learning_rates=lambda iter:max(0.1**(iter//10),0.001),

                         )

        

        lomodel.append(model)

        pred = model.predict(X_valid, num_iteration=model.best_iteration)

        pred = np.round(np.clip(pred,0,upper_bound)).astype(int)

        lopred.append(pred)

        print(f1_score(y_valid, pred, average=None))

        print('\n')



    

    return lomodel,lopred



dic_model['lomodel1'],lopred1 = fit_lgb(df1.drop(['open_channels'], axis = 1), df1.open_channels, 5)

dic_model['lomodel2'],lopred2 = fit_lgb(df2.drop(['open_channels'], axis = 1), df2.open_channels, 10)
lopred = lopred1+lopred2

pred = np.array([])

for i in lopred:

    pred = np.append(pred, i)

    

print('f1 score for each group:',f1_score(np.concatenate((df1.open_channels,df2.open_channels)), pred, average=None),

      'macro f1 score:',f1_score(np.concatenate((df1.open_channels,df2.open_channels)), pred, average='macro'))
def clip(data, group):

    if group == 1:

        data = np.clip(data, 0,5)

    else:

        data = np.clip(data, 0,10)

    return data





test_group = [1,1,1,1,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1]

sub = np.array([])

for i in range(20):

    group = test_group[i]

    lomodel = dic_model['lomodel'+str(group)]

    data = normal_distribution_feature(shift(test[100000*i:100000*(i+1)], n_shift).drop('time',axis=1),group)

    pred = 0

    for model in lomodel:

        pred += model.predict(data, num_iteration=model.best_iteration)

    pred = np.round(clip((pred / len(lomodel)), group)).astype(int)

    sub = np.append(sub, pred)
submission['open_channels'] = np.array(np.round(sub,0), np.int)

submission.to_csv('submission.csv', index=False, float_format='%.4f')