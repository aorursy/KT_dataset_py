%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score,\
    confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

from scipy import stats

sns.set_style("darkgrid")
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.shape
df.columns
df.info()
df.describe()
vc = df.Class.value_counts()
sns.barplot(x=vc.index, y=vc.values, data=vc)
print(vc)
print(vc/vc.sum())
fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,20))
for i in range(28):
    sns.distplot(df[df['Class'] == 0][f'V{i+1}'], ax=ax[i//5,i%5], label='non-fraud')
    sns.distplot(df[df['Class'] == 1][f'V{i+1}'], ax=ax[i//5,i%5], label='fraud')
    ax[i//5,i%5].legend()
sns.distplot(df[df['Class'] == 0]['Time'], ax=ax[5,3], label='non-fraud')
sns.distplot(df[df['Class'] == 1]['Time'], ax=ax[5,3], label='fraud')
ax[5,3].legend()
sns.distplot(df[df['Class'] == 0]['Amount'], ax=ax[5,4], label='non-fraud')
sns.distplot(df[df['Class'] == 1]['Amount'], ax=ax[5,4], label='fraud')
ax[5,4].legend()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
sns.scatterplot(x='V17', y='V14', hue='Class', data=df, ax=ax[0])
sns.scatterplot(x='V12', y='V14', hue='Class', data=df, ax=ax[1])
corr = df.corr()
corr.Class
class OneWaySkewReduction(BaseEstimator, TransformerMixin):
    """
    Custom transformer to reduce skewness. The process consists in
    squaring the signal before using the box-cox transformation.
    """
    def __init__(self):
        # Stores the best box-cox lambda after fitting
        self._params = {}
    
    def fit(self, X, y=None):
        # For each column in the training data, find the lambda
        # that maximizes the log-likelihood in the box-cox transf.
        for col in X:
            self._params[col] = None
            _, self._params[col] = self.skew_reduction(X[col])
        return self
    
    def transform(self, X, y=None):
        _X = X.copy()
        for col in X:
            _X[col] = self.skew_reduction(X[col], self._params[col])
        return _X
    
    def skew_reduction(self, X, lmbda=None):
        # Custom transformation function
        return stats.boxcox(np.power(X, 2) + np.finfo(float).eps, lmbda=lmbda)
X = df.drop('Class', axis=1)
y = df['Class']

# Find which coolumns in the training set have abs(skew) > 0.75
skewed_cols = X.columns[np.where(X.skew().abs() > 0.75)]

# Create the custom transformer
transformer = make_column_transformer(
    (StandardScaler(), ['Time']),
    (OneWaySkewReduction(), skewed_cols.values),
    remainder='drop'
)

# Replace the raw values in X
X[pd.Index(['Time']).append(skewed_cols)] = transformer.fit_transform(X)
Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=0.20,\
                                      random_state=0, stratify=y)
trvc = Ytr.value_counts()
tevc = Yte.value_counts()
trvc = trvc/trvc.sum()
tevc = tevc/tevc.sum()
print("Training dataset proportion")
print(trvc)
print("\nTesting dataset proportion")
print(tevc)
smote = SMOTE(sampling_strategy='minority')
Xsm, Ysm = smote.fit_sample(Xtr, Ytr)
Xsm.shape
# Metrics for comparison
mtrs = {'Recall': recall_score, 'Accuracy': accuracy_score,\
        'Precision': precision_score, 'F1': f1_score}

# Models (set random_state for reproducibility)
mdls = {'LR Smote': LogisticRegression(random_state=0),
        'LR Imb.': LogisticRegression(random_state=0),
        'RF Smote': RandomForestClassifier(random_state=0),
        'RF Imb.': RandomForestClassifier(random_state=0),
        'AB Smote': AdaBoostClassifier(random_state=0),
        'AB Imb.': AdaBoostClassifier(random_state=0)}
# Fit models using either the origintal training set or the
# SMOTE-balanced one
for m in mdls:
    if m.endswith('Smote'):
        mdls[m].fit(Xsm, Ysm)
    else:
        mdls[m].fit(Xtr, Ytr)
# Compute the scoring metrics in the same dataset used for training
tr_res = []
for m in mdls:
    if m.endswith('Smote'):
        Ypr = mdls[m].predict(Xsm)
        tr_res.append([mtrs[mt](Ysm, Ypr) for mt in mtrs])
    else:
        Ypr = mdls[m].predict(Xtr)
        tr_res.append([mtrs[mt](Ytr, Ypr) for mt in mtrs])
pd.DataFrame(tr_res, columns=mtrs.keys(), index=mdls.keys())
te_res = {}
cf_mat = {}
for m in mdls:
    Ypr = mdls[m].predict(Xte)
    te_res[m] = [mtrs[mt](Yte, Ypr) for mt in mtrs]
    cf_mat[m] = confusion_matrix(Yte, Ypr)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,12))
ticks = ['Non-fraud','Fraud']
k = 0
for i in cf_mat:
    with sns.plotting_context('notebook', font_scale=1.5):
        sns.heatmap(cf_mat[i]/np.sum(cf_mat[i]), annot=True,\
                fmt='.2%', cmap='Blues', cbar=False, xticklabels=ticks,\
                yticklabels=ticks, ax=ax[k//3,k%3], )
    ax[k//3,k%3].set_ylabel('True label', fontsize=18)
    ax[k//3,k%3].set_xlabel('Predicted label', fontsize=18)
    ax[k//3,k%3].set_title(i, fontsize=20)
    k = k + 1
pd.DataFrame(te_res, index=mtrs.keys()).T
# Compute the precision-recall curve
rpcurve = {}
for m in mdls:
    Ysc = mdls[m].predict_proba(Xte)[:,1]
    precision, recall, threshold = precision_recall_curve(Yte, Ysc)
    rpcurve[m] = pd.DataFrame(np.vstack((precision, recall)).T, columns=['Precision', 'Recall'])

# Plot the precision-recall curve
fig, ax = plt.subplots(figsize=(10,5))
for m in mdls:
    area = auc(rpcurve[m].Recall, rpcurve[m].Precision)
    sns.lineplot(x='Recall', y='Precision', data=rpcurve[m], ax=ax, label=f'{m} ({area})')