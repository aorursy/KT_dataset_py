import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')



df_train = pd.read_csv('../input/artificial-data-leaks/train.csv')

df_test = pd.read_csv('../input/artificial-data-leaks/test.csv')



df = pd.DataFrame({'Value': df_train['col8'].value_counts().index,

                  'Count in train': df_train['col8'].value_counts()})

df['Count in test'] = df_test['col8'].value_counts()

df = df.fillna(0).astype(int)

df
import lightgbm as lgb

import matplotlib.pyplot as plt

from sklearn import metrics



RS = 0

ROUNDS = 500

TARGET = 'target'



params = {

    'objective': 'binary',

    'metric': 'auc',

    'boosting': 'gbdt',

    'learning_rate': 0.1,

    'verbose': 0,

    'num_leaves': 64,

    'bagging_fraction': 0.8,

    'bagging_seed': RS,

    'feature_fraction': 0.9,

    'feature_fraction_seed': RS,

    'max_bin': 100,

    'max_depth': 5

}



x_train = lgb.Dataset(df_train.drop(TARGET, axis=1), df_train[TARGET])

model = lgb.train(params, x_train, num_boost_round=ROUNDS)



lgb.plot_tree(model, tree_index=0, figsize=(800, 48), show_info=['split_gain'])

plt.show()
df['Mean target in train'] = df_train.groupby('col8')['target'].mean()

df_mean = df[df['Value'].isin(range(90,110))]

df_mean
plt.bar(df_mean.index, df_mean['Mean target in train'])

pass
df_train['Leak1'] = df_train['col8'].apply(lambda x: 1 if x % 6 < 3 else 0)

df_mean['Leak1 mean'] = df_train.groupby('col8')['Leak1'].mean()

plt.bar(df_mean.index, df_mean['Leak1 mean'])

pass
df_train = pd.read_csv('../input/artificial-data-leaks/train.csv')

df_test = pd.read_csv('../input/artificial-data-leaks/test.csv')



df = pd.concat([df_train, df_test])



df['Leak1'] = df['col8'].apply(lambda x: 1 if x % 6 < 3 else 0)



df_train = df[:df_train.shape[0]]

df_test = df[df_train.shape[0]:]



x_train = lgb.Dataset(df_train.drop(TARGET, axis=1), df_train[TARGET])

model = lgb.train(params, x_train, num_boost_round=ROUNDS)

preds = model.predict(df_test.drop(TARGET, axis=1))



score = metrics.roc_auc_score(df_test[TARGET], preds)

print('Test AUC score:',score)



fig, axs = plt.subplots(ncols=2, figsize=(15,6))

lgb.plot_importance(model, importance_type='split', ax=axs[0], title='Feature importance (split)')

lgb.plot_importance(model, importance_type='gain', ax=axs[1], title='Feature importance (gain)')

pass



#Baseline not-tuned model, raw features:  AUC 0.74632

#First leak found, same model parameters: AUC 0.74675

#All leaks found, same model parameters:  AUC 0.93927
df_train = pd.read_csv('../input/artificial-data-leaks/train.csv')

df_test = pd.read_csv('../input/artificial-data-leaks/test.csv')



df = pd.concat([df_train, df_test])



df['Leak1'] = df['col8'].apply(lambda x: 1 if x % 6 < 3 else 0)

df.drop(['col8'], axis=1, inplace=True)



df_train = df[:df_train.shape[0]]

df_test = df[df_train.shape[0]:]



x_train = lgb.Dataset(df_train.drop(TARGET, axis=1), df_train[TARGET])

model = lgb.train(params, x_train, num_boost_round=ROUNDS)

preds = model.predict(df_test.drop(TARGET, axis=1))



score = metrics.roc_auc_score(df_test[TARGET], preds)

print('Test AUC score:',score)



fig, axs = plt.subplots(ncols=2, figsize=(15,6))

lgb.plot_importance(model, importance_type='split', ax=axs[0], title='Feature importance (split)')

lgb.plot_importance(model, importance_type='gain', ax=axs[1], title='Feature importance (gain)')

pass



#Baseline not-tuned model, raw features:  AUC 0.74632

#First leak found:                        AUC 0.74675

#First leak found, original col8 dropped: AUC 0.74715

#All leaks found, same model parameters:  AUC 0.93927