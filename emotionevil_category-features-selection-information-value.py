import os
import gc
import time
import psutil
import numpy as np
import pandas as pd
import random as rn
import seaborn as sns

from tqdm import tqdm
from matplotlib import pyplot as plt
from contextlib import contextmanager
from tensorflow import set_random_seed

rn.seed(5)
np.random.seed(7)
set_random_seed(2)
os.environ['PYTHONHASHSEED'] = '3'
timer_depth = -1
@contextmanager
def timer(name):
    t0 = time.time()
    global timer_depth
    timer_depth += 1
    yield
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print('----'*timer_depth + f'>>[{name}] done in {time.time() - t0:.0f} s ---> memory used: {memoryUse:.4f} GB', '')
    if(timer_depth == 0):
        print('\n')
    timer_depth -= 1
base_path = '../input'
with timer('read application data'):
    app_train = pd.read_csv(os.path.join(base_path, 'application_train.csv'))
print(f'app_train shape : {app_train.shape}')
app_train_target = app_train.pop('TARGET')
print('pop application train TARGET')
print(f'app_train shape : {app_train.shape}')
app_train = app_train.select_dtypes(include=['object'])
app_train = app_train.fillna('XNA')
print(app_train.shape)
app_train.head()
def cal_woe(app_train, app_train_target):
    num_events = app_train_target.sum()
    num_non_events = app_train_target.shape[0] - app_train_target.sum()

    feature_list = []
    feature_iv_list = []
    for col in app_train.columns:
        if app_train[col].unique().shape[0] == 1:
            del app_train[col]
            print('remove constant col', col)

        with timer('cope with %s' % col):
            feature_list.append(col)

            woe_df = pd.DataFrame()
            woe_df[col] = app_train[col]
            woe_df['target'] = app_train_target
            events_df = woe_df.groupby(col)['target'].sum().reset_index().rename(columns={'target' : 'events'})
            events_df['non_events'] = woe_df.groupby(col).count().reset_index()['target'] - events_df['events']
            def cal_woe(x):
                return np.log( ((x['non_events']+0.5)/num_non_events) / ((x['events']+0.5)/num_events)  )
            events_df['WOE_'+col] = events_df.apply(cal_woe, axis=1)

            def cal_iv(x):
                return x['WOE_'+col]*(x['non_events'] / num_non_events - x['events'] / num_events)
            events_df['IV_'+col] = events_df.apply(cal_iv, axis=1)

            feature_iv = events_df['IV_'+col].sum()
            feature_iv_list.append(feature_iv)

            events_df = events_df.drop(['events', 'non_events', 'IV_'+col], axis=1)
            app_train = app_train.merge(events_df, how='left', on=col)
    iv_df = pd.DataFrame()
    iv_df['feature'] = feature_list
    iv_df['IV'] = feature_iv_list
    iv_df = iv_df.sort_values(by='IV', ascending=False)
    return app_train, iv_df

with timer('calculate WOE and IV'):
    app_train, iv_df = cal_woe(app_train, app_train_target)
app_train.head()
iv_df
iv_df.loc[iv_df['IV']>0.02]


