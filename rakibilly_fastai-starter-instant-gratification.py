%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.tabular import *

from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
path = Path('../input/')

train = pd.read_csv(path/'train.csv')

test = pd.read_csv(path/'test.csv')
train.head()
dep_var = 'target'

cat_names = ['wheezy-copper-turtle-magic']

cont_names = train.columns.tolist()

cont_names = [e for e in cont_names if e not in ('id', 'target', 'wheezy-copper-turtle-magic')]

procs = [FillMissing, Categorify, Normalize]
np.random.seed(1984)

idx = np.random.randint(0, len(train), size=np.int(.1*len(train)))
bs=1024

data = TabularDataBunch.from_df('.', train, dep_var=dep_var, valid_idx=idx, 

                                procs=procs, cat_names=cat_names, 

                                cont_names=cont_names, test_df=test, bs=bs)



data.show_batch(3)
learn = tabular_learner(data, layers=[500, 200, 100, 60], metrics=accuracy, ps=0.2).to_fp16()
lr_find(learn) #, start_lr=1e-7, end_lr=10, num_it=100

learn.recorder.plot()
learn.fit_one_cycle(10, 3e-2)
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
val_preds = learn.get_preds(DatasetType.Valid)

y_true = train.iloc[idx].target.values

y_preds = val_preds[0][:,1].numpy()

roc_auc_score(y_true, y_preds)

test_preds = learn.get_preds(DatasetType.Test)
sub_df = pd.read_csv(path/'sample_submission.csv')

sub_df.target = test_preds[0][:,1].numpy()
sub_df.to_csv('submission.csv', index=False)
sub_df.head()