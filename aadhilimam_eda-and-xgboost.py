# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import pandas as pd 
import seaborn as sns


from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from category_encoders import CountEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

from sklearn.multioutput import MultiOutputClassifier

import os
import warnings
warnings.filterwarnings('ignore')
train_features.shape
plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
sns.countplot(x='cp_type', data=train_features, palette='pastel')
plt.title('Train: Control and treated samples', fontsize=15, weight='bold')
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(x='cp_dose', data=train_features, palette='Purples')
plt.title('Train: Treatment Doses: Low and High',weight='bold', fontsize=18)
plt.show()
plt.figure(figsize=(15,5))
sns.distplot( train_features['cp_time'], color='red', bins=5)
plt.title("Train: Treatment duration ", fontsize=15, weight='bold')
plt.show()
train_features.cp_type.value_counts(normalize=True).plot(kind='pie', figsize=(15, 5), fontsize=12,
                                                         title='CP Type', autopct='%1.1f%%')
plt.show()
train_features.cp_time.value_counts(normalize=True).plot(kind='bar', figsize=(12, 5), fontsize=14,
                                                         title='CP Time', xlabel='Time')
plt.show()
train_features.cp_dose.value_counts(normalize=True).plot(kind='bar', figsize=(12, 5), fontsize=14,
                                                         title='CP Dose', xlabel='Dose')
plt.show()
GENE_COLS = [c for c in train_features.columns if c[:2] == 'g-']
CELL_COLS = [c for c in train_features.columns if c[:2] == 'c-']
print('Number of gene columns:', len(GENE_COLS))
print('Number of cell columns:', len(CELL_COLS))
ax = train_features.set_index('sig_id') \
    .sample(10)[GENE_COLS] \
    .T.plot(figsize=(15, 5))
plt.suptitle('Gene Features for 10 Random Samples', fontsize=20)
ax.get_legend().remove()
plt.show()
tmp_df = train_features.loc[:, ['g-0', 'g-1', 'g-2', 'c-97', 'c-98', 'c-99']]

plt.figure(figsize=(8, 8))
sns.heatmap(tmp_df.corr(), annot=True)
plt.show()
SEED = 42
NFOLDS = 5

np.random.seed(SEED)
# drop id col
X = train_features.iloc[:,1:].to_numpy()
X_test = test_features.iloc[:,1:].to_numpy()
y = train_targets_scored.iloc[:,1:].to_numpy() 
clf = Pipeline([('encode', CountEncoder(cols=[0, 2])),
                ('classify', MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist')))
               ])

params = {'classify__estimator__colsample_bytree': 0.652231655518253,
          'classify__estimator__gamma': 3.6975211709521023,
          'classify__estimator__learning_rate': 0.05033414197773552,
          'classify__estimator__max_delta_step': 2.070593162427692,
          'classify__estimator__max_depth': 10,
          'classify__estimator__min_child_weight': 31.579959348704868,
          'classify__estimator__n_estimators': 166,
          'classify__estimator__subsample': 0.8638628715886625,
          'encode__min_group_size': 0.4160029192647806}

clf.set_params(**params)

oof_preds = np.zeros(y.shape)
test_preds = np.zeros((test_features.shape[0], y.shape[1]))
kf = KFold(n_splits=NFOLDS)
for fn, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print('Starting fold: ', fn)
    X_train, X_val = X[trn_idx], X[val_idx]
    y_train, y_val = y[trn_idx], y[val_idx]
    clf.fit(X_train, y_train)
    val_preds = clf.predict_proba(X_val) # list of preds per class
    val_preds = np.array(val_preds)[:,:,1].T # take the positive class
    oof_preds[val_idx] = val_preds
    
    preds = clf.predict_proba(X_test)
    preds = np.array(preds)[:,:,1].T # take the positive class
    test_preds += preds / NFOLDS
print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))
# create the submission file
sample_submission.iloc[:,1:] = test_preds
sample_submission.to_csv('submission.csv', index=False)