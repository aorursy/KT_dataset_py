import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.shape
df_train.isna().sum()
sns.countplot(data=df_train, x='experiment', hue='event')
sns.countplot(data=df_train, x='event')
plt.figure(figsize=(15, 10))

sns.violinplot(x='event', y='time', data=df_train.sample(5000))
experiments = {'CA': 0, 'DA': 1, 'SS': 2, 'LOFT': 3}

df_train["experiment"] = df_train["experiment"].apply(lambda x: experiments[x])

df_test["experiment"] = df_test["experiment"].apply(lambda x: experiments[x])
events = {'A': 0, 'B':1, 'C':2, 'D':3}

df_train["event"] = df_train["event"].apply(lambda x: events[x])
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df_train, test_size=0.3, random_state=420)
import lightgbm as lgb

from tqdm import tqdm_notebook as tqdm
features = ["crew", "seat", "eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr"]

params = {"objective" : "multiclass",

              "num_class": 4,

              "metric" : "multi_error",

              "num_leaves" : 30,

              "min_child_weight" : 50,

              "learning_rate" : 0.1,

              "bagging_fraction" : 0.7,

              "feature_fraction" : 0.7,

              "bagging_seed" : 420,

              "verbosity" : -1

            }
lg_train = lgb.Dataset(train_data[features], label=(train_data["event"]))

lg_test = lgb.Dataset(test_data[features], label=(test_data["event"]))

model = lgb.train(params, lg_train, 1000, valid_sets=[lg_test], early_stopping_rounds=50, verbose_eval=100)
predictions = model.predict(test_data[features], num_iteration=model.best_iteration)
submission = pd.DataFrame(np.concatenate((np.arange(len(test_data))[:, np.newaxis], predictions), axis=1), columns=['id', 'A', 'B', 'C', 'D'])

submission['id'] = submission['id'].astype(int)
submission.head()
submission.to_csv('submission.csv', index=False)
submission.shape