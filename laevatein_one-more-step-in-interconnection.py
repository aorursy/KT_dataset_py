import numpy as np, pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from tqdm import tqdm_notebook



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style("whitegrid")



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
cols = [c for c in train_data.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



magic_idx = []

magic_pred = []

magic_auc = []



for i in tqdm_notebook(range(512)):

    train2 = train_data[train_data['wheezy-copper-turtle-magic'] == i]

    train2.reset_index(drop=True, inplace=True)



    clf = LogisticRegression(solver='liblinear', penalty='l1', C=0.05)

    clf.fit(train2[cols], train2['target'])



    for j in range(0, 512):

        val = train_data[train_data['wheezy-copper-turtle-magic'] == j]

        preds = clf.predict_proba(val[cols])[:, 1]

        auc = roc_auc_score(val['target'], preds)

        magic_idx.append(i)

        magic_pred.append(j)

        magic_auc.append(auc)
magic_mx = pd.DataFrame({'magic_fit':magic_idx, 'magic_pred':magic_pred, 'auc':magic_auc})

magic_mx = magic_mx[['magic_fit', 'magic_pred', 'auc']]

magic_mx.head()
magic_mx_pt = pd.pivot_table(magic_mx, index='magic_fit', columns='magic_pred', values='auc')



plt.style.use({'figure.figsize':(18, 15), 'font.size':15}) # set the size of plots

sns.heatmap(magic_mx_pt)
sns.heatmap(magic_mx_pt > 0.6) # higher correlated
cols = [c for c in train_data.columns if c not in ['id', 'wheezy-copper-turtle-magic']]



magic_num = []

col_name = []

corr_ls = []

for i in tqdm_notebook(range(512)):

    tmp = train_data[train_data['wheezy-copper-turtle-magic'] == i]

    correlations = tmp[cols].corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

    correlations = correlations[correlations['level_0'] != correlations['level_1']]

    corr = correlations[correlations['level_0'] == 'target']

    magic_num.append(i)

    col_name.append(corr['level_1'].iloc[0])

    corr_ls.append(corr[0].iloc[0])
corr_under_magic = pd.DataFrame({'magic':magic_num, 'feature':col_name, 'corr':corr_ls})

corr_under_magic.head()
corr_under_magic['feature'].nunique()
corr_under_magic['feature'].value_counts()
cols = [c for c in train_data.columns if c not in ['id', 'wheezy-copper-turtle-magic']]



magic_num = []

col_name = []

corr_ls = []

for i in tqdm_notebook(range(512)):

    tmp = train_data[train_data['wheezy-copper-turtle-magic'] == i]

    correlations = tmp[cols].corr().abs().unstack().sort_values(kind="quicksort", ascending=True).reset_index()

    correlations = correlations[correlations['level_0'] != correlations['level_1']]

    corr = correlations[correlations['level_0'] == 'target']

    magic_num.append(i)

    col_name.append(corr['level_1'].iloc[0])

    corr_ls.append(corr[0].iloc[0])
corr_under_magic = pd.DataFrame({'magic':magic_num, 'feature':col_name, 'corr':corr_ls})

corr_under_magic.head()
corr_under_magic['feature'].nunique()
corr_under_magic['feature'].value_counts()