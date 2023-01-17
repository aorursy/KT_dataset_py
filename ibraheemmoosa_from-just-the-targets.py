import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.dummy import DummyClassifier

from sklearn.metrics import log_loss, make_scorer

from pathlib import Path
data_dir = Path('../input/lish-moa')
tr_tg = pd.read_csv(data_dir/'train_targets_scored.csv', index_col='sig_id')
Y = tr_tg.iloc[:,0:].values

Y.shape
def mean_log_loss(Y, Y_pred):

    return np.mean([log_loss(Y[:,i], Y_pred[i], labels=[0, 1]) for i in range(len(Y_pred))])
clf_prior = DummyClassifier(strategy='prior')
clf_prior.fit(Y, Y)
mean_log_loss(Y, clf_prior.predict_proba(Y))
list(zip(tr_tg.columns, map(lambda x: x[1], clf_prior.class_prior_)))
submission = pd.read_csv(data_dir/'sample_submission.csv', index_col='sig_id')
for i, p in enumerate(clf_prior.class_prior_):

    submission.iloc[:, i] = p[1]
submission.to_csv('submission.csv', index=True)