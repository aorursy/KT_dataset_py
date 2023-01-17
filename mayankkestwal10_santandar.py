import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/train.csv',index_col='ID_code')

df.head()
trues = df.loc[df['target']==1]

falses = df.loc[df['target']!=1].sample(frac=1)[:len(trues)]

data = pd.concat([trues,falses],ignore_index=True).sample(frac=1)

data.head()
y = df['target']

X = df.drop('target',axis=1)



test = pd.read_csv('../input/test.csv',index_col='ID_code')
import lightgbm as lgb

from sklearn.model_selection import KFold
# K-fold corss validation with 10 fold

n_splits = 5

kf = KFold(n_splits=n_splits)



# model with LGB

import warnings

warnings.filterwarnings('ignore')



param = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.0122, 'num_rounds': 7000, 'verbose': 1, 'n_estimators':500}

test_pred = np.zeros(len(test))

for fold, (train_indx, val_indx) in enumerate(kf.split(y)):

    print("Fold {}".format(fold+1))

    train_set = lgb.Dataset(X.iloc[train_indx], label=y.iloc[train_indx])

    val_set = lgb.Dataset(X.iloc[val_indx], label=y.iloc[val_indx])

    model = lgb.train(param, train_set, valid_sets=val_set, verbose_eval=500)

    test_pred += model.predict(test)/n_splits
test_pred
submission = pd.read_csv('../input/sample_submission.csv')

sub = pd.DataFrame(test_pred,columns=['target'])

submission.update(sub)
submission.to_csv('lightgbm2.csv',index=False)
submission.head()