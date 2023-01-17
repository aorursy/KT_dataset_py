import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
%matplotlib inline
import re
from itertools import product
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from scipy import stats
dat = pd.read_pickle("../input/gstore-revenue-data-preprocessing/train.pkl")
dat.apply(lambda x: np.sum(pd.isna(x)))
cat_columns = [c for c in dat.columns if str(dat[c].dtype) == 'category']
missing_count = dat.apply(lambda x: np.sum(pd.isna(x)))
col_w_missing = list(missing_count[missing_count > 0].index)
col_w_missing
missing = dat.copy()
for col in col_w_missing:
    missing['miss_' + col] = pd.isnull(dat[col])
zero_revenue = missing['totals.transactionRevenue'] == 0
missing['miss_totals.transactionRevenue'] = zero_revenue
col_w_missing.append('totals.transactionRevenue')
ind_miss_p = np.full((len(cat_columns), len(col_w_missing)), np.nan)
for i, j in product(
        range(len(cat_columns)), range(len(col_w_missing))):
    chi2, p, dof, ex = stats.chi2_contingency(
        missing.groupby([cat_columns[i], 'miss_' + col_w_missing[j]
                         ]).size().unstack().fillna(0).astype(np.int))
    ind_miss_p[i, j] = p
    
miss_ind_test_output = pd.DataFrame(
    ind_miss_p,
    index=cat_columns,
    columns=['miss_' + c for c in col_w_missing])
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(data=miss_ind_test_output, ax=ax, linewidths=0.01)
ax.set_title("p-values of chi2 independence test of categorical values vs missingness")
plt.show()
ind_miss2miss_p = np.full((len(col_w_missing), len(col_w_missing)), 0.)
for i, j in product(range(len(col_w_missing)), range(len(col_w_missing))):
    if i < j:
        chi2, p, dof, ex = stats.chi2_contingency(
            missing.groupby([
                'miss_' + col_w_missing[i], 'miss_' + col_w_missing[j]
            ]).size().unstack().fillna(0).astype(np.int))
        ind_miss2miss_p[i, j] = p
        ind_miss2miss_p[j, i] = ind_miss2miss_p[i, j]
    elif i == j:
        ind_miss2miss_p[i, j] = 0

miss2miss_p_output = pd.DataFrame(
    ind_miss2miss_p,
    index=['miss_' + c for c in col_w_missing],
    columns=['miss_' + c for c in col_w_missing])

g = sns.clustermap(
    data=miss2miss_p_output, figsize=(12, 12), linewidths=0.01)
g.ax_col_dendrogram.set_title("pairwise p-value of column missingness independence test")
plt.show()
ind_miss2miss_mcc = np.full((len(col_w_missing), len(col_w_missing)), 0.)
for i, j in product(range(len(col_w_missing)), range(len(col_w_missing))):
    if i < j:
        ind_miss2miss_mcc[i, j] = matthews_corrcoef(
            missing['miss_' + col_w_missing[i]],
            missing['miss_' + col_w_missing[j]])
        ind_miss2miss_mcc[j, i] = ind_miss2miss_mcc[i, j]
    elif i == j:
        ind_miss2miss_mcc[i, j] = 1

miss2miss_mcc_output = pd.DataFrame(
    ind_miss2miss_mcc,
    index=['miss_' + c for c in col_w_missing],
    columns=['miss_' + c for c in col_w_missing])
miss2miss_mcc_output.index.name = 'predicted'
miss2miss_mcc_output.columns.name = 'input'

g = sns.clustermap(
    data=miss2miss_mcc_output, figsize=(12, 12), linewidths=0.01)
g.ax_col_dendrogram.set_title("pairwise MCC score of column missingness")
plt.show()
ind_miss2miss_auc = np.full((len(col_w_missing), len(col_w_missing)), 0.)
for i, j in product(range(len(col_w_missing)), range(len(col_w_missing))):
        score1 = roc_auc_score(missing['miss_' + col_w_missing[i]],
                                          missing['miss_' + col_w_missing[j]])
        score2 = roc_auc_score(missing['miss_' + col_w_missing[i]],
                                          ~missing['miss_' + col_w_missing[j]])
        ind_miss2miss_auc[i, j] = max(score1, score2)
        
miss2miss_auc_output = pd.DataFrame(
    ind_miss2miss_auc,
    index=['miss_' + c for c in col_w_missing],
    columns=['miss_' + c for c in col_w_missing])
miss2miss_auc_output.index.name = 'predicted'
miss2miss_auc_output.columns.name = 'input'

g = sns.clustermap(data=miss2miss_auc_output, figsize=(12, 12), linewidths=0.01)
g.ax_col_dendrogram.set_title("pairwise AUC score of column missingness")
plt.show()
cur_dict = dict()
cols = [c for c in col_w_missing if c != 'totals.transactionRevenue']
for c in cols:
    fpr_p, tpr_p, _ = roc_curve(~missing['miss_totals.transactionRevenue'],
                                missing['miss_' + c])
    fpr_n, tpr_n, _ = roc_curve(~missing['miss_totals.transactionRevenue'],
                                ~missing['miss_' + c])
    auc_p, auc_n = auc(fpr_p, tpr_p), auc(fpr_n, tpr_n)
    if auc_p >= 0.55:
        cur_dict[c] = [fpr_p, tpr_p, auc_p]
    elif auc_n >= 0.55:
        cur_dict[c] = [fpr_n, tpr_n, auc_n]

plt.figure(figsize=(12, 12))
lw = 2
for c, v in cur_dict.items():
    plt.plot(v[0], v[1], lw=lw, label="{0}  AUC={1}".format(c, v[2]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
missing.groupby(['miss_totals.transactionRevenue', 'miss_totals.bounces']).size().unstack().fillna(0)
X = missing.loc[:, [
    c for c in missing.columns if re.match(r'miss_', c) is not None
    and c != 'miss_totals.transactionRevenue'
]]
y = ~missing['miss_totals.transactionRevenue']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=7777)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_X, train_y)
preds = clf.predict(test_X)
probs = clf.predict_proba(test_X)
print(classification_report(test_y, preds))
print(classification_report(test_y, probs[:, 1] > 0.01))
pd.DataFrame(
    confusion_matrix(test_y, probs[:, 1] > 0.01),
    columns=['pred_miss', 'pred_exist'],
    index=['miss', 'exist'])
fpr, tpr, _ = roc_curve(test_y, probs[:, 1])
auc_score = auc(fpr, tpr)
plt.figure(figsize=(12, 12))
lw = 2
c = 'totals.bounces'
v = cur_dict[c]
plt.plot(v[0], v[1], lw=lw, label="{0}  AUC={1}".format(c, v[2]))
plt.plot(fpr, tpr, lw=lw, label="{0}  AUC={1}".format('RF classifier', auc_score))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('ROC of RF Classifier Based on Missingness')
plt.show()
