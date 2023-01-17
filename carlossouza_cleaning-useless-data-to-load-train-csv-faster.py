import numpy as np

import pandas as pd
%%time

train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
train.shape[0], train['installation_id'].unique().shape[0]
t1 = train.groupby(['installation_id', 'type']).agg({'game_session': 'count'}).reset_index()

t2 = t1.pivot(index='installation_id', columns='type', values='game_session').reset_index()

t2
t2['Assessment'].isna().sum(), t2['Assessment'].isna().sum() / t2.shape[0]
useful_installation_ids = t2[~t2['Assessment'].isna()]['installation_id'].values

clean_train = train[train['installation_id'].isin(useful_installation_ids)]

clean_train.shape[0], clean_train.shape[0] / train.shape[0]
useful_installation_ids_sl = train_labels['installation_id'].unique()

clean_train_sl = train[train['installation_id'].isin(useful_installation_ids_sl)]

clean_train_sl.shape[0], clean_train_sl.shape[0] / train.shape[0]