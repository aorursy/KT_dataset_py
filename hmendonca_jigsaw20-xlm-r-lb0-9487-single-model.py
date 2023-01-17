!git clone https://github.com/henrique/jigsaw20.git



import sys

sys.path.append('/kaggle/working/jigsaw20')
import numpy as np

import pandas as pd

from train import train



params = dict(

    pooling='first',

    optimizer='LAMB',

    batch_size=27,

    lr=0.000277952,

    mom_min=0.806579,

    mom_max=0.922184,

    div_factor=55.477,

    final_div_factor=1123.49,

    weight_decay=7.72285e-06,

    dropout=0.4,

    loss_fn='bce',

    label_smoothing=0.0483175,

    warm_up=1.2361,

    epochs=41,

    steps_per_epoch=250,

    dataset='../input/jigsaw20-ds-tt6-36/jigsaw20_ds1789117tt6_fold5.npz',

    path=f'jigsaw',

    tpu_id=None, gcs=None,

    seed=1083,

)



# auc = train(**params)
# clean up repo

!rm -r jigsaw20
!cp /kaggle/input/jigsaw-tt6f5-20200614-091231-val0-965669/* .

!ls -shS
from matplotlib import pyplot as plt

_ = pd.read_csv('submission.csv').toxic.hist(bins=100, log=True, alpha=0.6)

_ = pd.read_csv('valid_oof.csv').groupby('toxic').pred.hist(bins=100, log=True, alpha=0.5)

plt.legend(['test', 'val0-normal', 'val1-toxic'])



pd.read_csv('history.csv')
pd.read_csv('params0.965669.csv').T
sub = pd.read_csv('submission.csv')

test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')



sub.loc[test["lang"] == "es", "toxic"] *= 1.06

sub.loc[test["lang"] == "fr", "toxic"] *= 1.04

sub.loc[test["lang"] == "it", "toxic"] *= 0.97

sub.loc[test["lang"] == "pt", "toxic"] *= 0.96

sub.loc[test["lang"] == "tr", "toxic"] *= 0.98

# min-max norm

sub.toxic -= sub.toxic.min()

sub.toxic /= sub.toxic.max()

sub.toxic.hist(bins=100, log=True, alpha=0.6)



sub.to_csv('submission0.csv', index=False)
ensemble = pd.read_csv('../input/jigsaw20xiwuhanjmtc2ndplacesolution/submission.csv')



# min-max norm

ensemble.toxic -= ensemble.toxic.min()

ensemble.toxic /= ensemble.toxic.max()



ensemble.toxic.hist(bins=100, log=True, alpha=0.6)

sub.toxic.hist(bins=100, log=True, alpha=0.6)



ensemble.toxic = ensemble.toxic * 0.8 + sub.toxic * 0.2

ensemble.to_csv('submission.csv', index=False)