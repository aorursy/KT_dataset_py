!pip install ../input/pytorch-tabnet/pytorch_tabnet-1.2.0-py3-none-any.whl
import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

from itertools import cycle

import torch

from pytorch_tabnet.tab_model import TabNetClassifier,TabNetRegressor

pd.set_option('max_columns', 50)

plt.style.use('seaborn-dark')

color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
!ls -GFlash --color ../input/lish-moa/
ss = pd.read_csv('../input/lish-moa/sample_submission.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
GENE_COLS = [c for c in train_features.columns if c[:2] == 'g-']

CELL_COLS = [c for c in train_features.columns if c[:2] == 'c-']

print('Number of gene columns:', len(GENE_COLS))

print('Number of cell columns:', len(CELL_COLS))
from sklearn.metrics import log_loss

def kaggle_metric_np(targets, preds):

    """

    Kaggle metric for MoA competition targets and preds

    in numpy format.

    """

    assert targets.shape[1] == 206

    assert preds.shape[1] == 206

    metrics = []

    for t in range(206):

        metrics.append(log_loss(targets[:, t], preds[:, t], labels=[0, 1]))

    return np.mean(metrics)
from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

# from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegressionCV

from sklearn.svm import LinearSVC 

from sklearn.metrics import log_loss

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



LABEL_ENCODE_COLS = ['cp_type','cp_time','cp_dose']

for l in LABEL_ENCODE_COLS:

    le = LabelEncoder()

    train_features[f'{l}_le'] = le.fit_transform(train_features[l])

    test_features[f'{l}_le'] = le.transform(test_features[l])



FEATURES = GENE_COLS + CELL_COLS + ['cp_type_le','cp_time_le','cp_dose_le']

TARGETS = [t for t in train_targets_scored.columns if t != 'sig_id']



df = train_features[FEATURES]

test_df = test_features[FEATURES]

y = train_targets_scored[TARGETS]







# X = train_features[FEATURES].values

# X_test = test_features[FEATURES].values

# y = train_targets_scored[TARGETS].values



# Needed

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)







# Not needed

# X_full = np.concatenate([X, X_test])



# Standard Scale

# scale = StandardScaler()

# scale.fit(X_full)

# X_train = scale.transform(X_train)

# X_val = scale.transform(X_val)

# X_test = scale.transform(X_test)



# # Apply PCA

# pca = PCA(n_components=100, svd_solver='full')

# pca.fit(X_full)

# X_train = pca.transform(X_train)

# X_val = pca.transform(X_val)

# X_test = pca.transform(X_test)

# print(X_train.shape, X_val.shape, X_test.shape)
from sklearn.model_selection import KFold

NUM_FOLDS=5



df = df.dropna().reset_index(drop=True)

df["kfold"] = -1

y = y.dropna().reset_index(drop=True)

y["kfold"] = -1



df = df.sample(frac=1,random_state=2020).reset_index(drop=True)

y = y.sample(frac=1,random_state=2020).reset_index(drop=True)



kf = KFold(n_splits=NUM_FOLDS)



for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):

    df.loc[val_, 'kfold'] = fold

    y.loc[val_,'kfold'] = fold
y.loc[val_,'kfold']
y_test = np.zeros((test_df.shape[0],len(TARGETS), NUM_FOLDS))
features=FEATURES

target_features = TARGETS

def run(fold):

    df_train = df[df.kfold != fold]

    df_valid = df[df.kfold == fold]

    

    X_train = df_train[features].values

    Y_train = y[y.kfold!=fold][TARGETS].values

#     Y_train = df_train[target_features].values

    

    X_valid = df_valid[features].values

    Y_valid = y[y.kfold==fold][TARGETS].values

#     Y_valid = df_valid[target_features].values

    

    y_oof = np.zeros((df_valid.shape[0],len(target_features)))   # Out of folds validation

    

    print("--------Training Begining for fold {}-------------".format(fold+1))

     

    model.fit(X_train = X_train,

             y_train = Y_train,

             X_valid = X_valid,

             y_valid = Y_valid,

             max_epochs = 1000,

             patience =70)

              

    

    print("--------Validating For fold {}------------".format(fold+1))

    

    y_oof = model.predict(X_valid)

    y_test[:,:,fold] = model.predict(test_df.values)

    

    val_score = kaggle_metric_np(Y_valid,y_oof)

    

    print("Validation score: {:<8.5f}".format(val_score))

    

    # VISUALIZTION

    plt.figure(figsize=(12,6))

    plt.plot(model.history['train']['loss'])

    plt.plot(model.history['valid']['loss'])
import warnings

warnings.simplefilter("ignore")



# clf = OneVsRestClassifier(SVC(probability=True))

model = TabNetRegressor(n_d=64,

                       n_a=64,

                       n_steps=8,

                       gamma=1.9,

                       n_independent=4,

                       n_shared=5,

                       seed=2020,

                       optimizer_fn = torch.optim.Adam,

                       scheduler_params = {"milestones": [150,250,300,350,400,450],'gamma':0.2},

                       scheduler_fn=torch.optim.lr_scheduler.MultiStepLR)

# clf.fit(X_train = X_train,

#              y_train = y_train,

#              X_valid = X_val,

#              y_valid = y_val,

#              max_epochs = 1000,

#              patience =70)

# clf.fit(X_train,y_train)

# pred_train = clf.predict_proba(X_train)

# pred_val = clf.predict_proba(X_val)

# pred_test = clf.predict_proba(X_test)
run(fold=0)
run(fold=1)
run(fold=2)
run(fold=3)
run(fold=4)
# run(fold=5)
# run(fold=6)
y_test = y_test.mean(axis=-1)
sub = pd.DataFrame(y_test, columns=TARGETS)

sub['sig_id'] = test_features['sig_id'].values
sub.shape, ss.shape
sub.to_csv('submission.csv', index=False)
sub