%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastai.tabular import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
orig_df = pd.read_csv('../input/creditcard.csv')
orig_df.head()
orig_df.loc[orig_df['Class'] == 0].shape[0], orig_df.loc[orig_df['Class'] == 1].shape[0]
smote = SMOTE()
X, y = smote.fit_sample(orig_df.drop(columns=['Class']), orig_df['Class'])
y = y[..., np.newaxis]
Xy = np.concatenate((X, y), axis=1)
df = pd.DataFrame.from_records(Xy, columns=list(orig_df))
df.Class = df.Class.astype(int)
df.head()
df.loc[df['Class'] == 0].shape[0], df.loc[df['Class'] == 1].shape[0]
#procs = [FillMissing, Categorify, Normalize]
procs = [Normalize]
idx = np.arange(df.shape[0])
train, valid, train_idx, valid_idx = train_test_split(df, idx, test_size=0.05, random_state=42)
train_idx.shape, valid_idx.shape
dep_var = 'Class'
cat_names = []
data = TabularDataBunch.from_df('.', df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
print(data.train_ds.cont_names)  # `cont_names` defaults to: set(df)-set(cat_names)-{dep_var}
learn = tabular_learner(data, layers=[1000, 500], ps=[0.001, 0.01], emb_szs={}, metrics=[])
learn.lr_find(stop_div=True, num_it=100)
learn.recorder.plot()
learn.fit_one_cycle(1, 1e-2)
learn.recorder.plot_lr()
[train_preds, train_targets] = learn.get_preds(ds_type=DatasetType.Train)
train_preds = to_np(train_preds)[:, 1]
train_targets = to_np(train_targets)
precision, recall, thresholds = precision_recall_curve(train_targets, train_preds)
plt.plot(recall, precision, marker='.')
auprc = auc(recall, precision)
auprc
[valid_preds, valid_targets] = learn.get_preds(ds_type=DatasetType.Valid)
valid_preds = to_np(valid_preds)[:, 1]
valid_targets = to_np(valid_targets)
precision, recall, thresholds = precision_recall_curve(valid_targets, valid_preds)
plt.plot(recall, precision, marker='.')
auprc = auc(recall, precision)
auprc
F1 = [2*p*r/(p+r) for p, r in zip(precision, recall)]
idx = np.argmax(F1)
np.max(F1), precision[idx], recall[idx], thresholds[idx]
preds = valid_preds.copy()
preds[preds >= thresholds[idx]] = 1
preds[preds < thresholds[idx]] = 0
f1_score(valid_targets, preds)
