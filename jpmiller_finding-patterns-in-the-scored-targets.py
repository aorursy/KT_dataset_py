import pandas as pd
targets =  pd.read_csv("../input/lish-moa/train_targets_scored.csv",

                 index_col=['sig_id'])

targets_0_idx = targets[targets.sum(axis=1)==0].index



train_features =  pd.read_csv("../input/lish-moa/train_features.csv",

                 index_col=['sig_id'])

control_idx = train_features.query('cp_type=="ctl_vehicle"').index



diffs = len(set(control_idx) - set(targets_0_idx))

test_features =  pd.read_csv("../input/lish-moa/test_features.csv",

                 index_col=['sig_id'])

test_control_idx = test_features.query('cp_type=="ctl_vehicle"').index



ctrl_pct = len(set(test_control_idx))/len(test_features)



print(diffs, ctrl_pct)
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt





targets_treated = targets.drop(targets_0_idx)

corr_mtx = abs(targets_treated.corr())



corr_map = corr_mtx[corr_mtx>=.7]

plt.figure(figsize=(12,8))

sns.heatmap(corr_map, cmap="viridis")
pairs = (corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))

                     .stack()

                     .sort_values(ascending=False))

pairs[:4]
targets.sum(axis=1).hist()
import networkx as nx



pairs_df = (1-pairs).reset_index()

G = nx.from_pandas_edgelist(pairs_df[:20], source='level_0', target='level_1', edge_attr=0)



graph_opts = dict(arrows=False,

                  node_size=5,

                  width=2,

                  alpha=0.8,

                  font_size=12,

                  font_color='darkblue',

                  edge_color='darkgray'

                 )



fig= plt.figure(figsize=(12,10))

nx.draw_spring(G, with_labels=True, **graph_opts)
import missingno as msno



cols_sorted = targets_treated.sum().sort_values(ascending=False).index

targets_visible = targets_treated[cols_sorted].replace(0, np.nan)

                                                # you can also use pd.NA with pandas v1+



msno.matrix(targets_visible.iloc[:, :50].sample(n=1000), sort='descending', color=(1,0,0))
top_cols, idx = np.unique(pairs_df.iloc[:4, :2].values.flatten(), return_index=True)

                                                    # use df.to_numpy() for 1.0+

msno.matrix(targets_visible.loc[targets_treated[top_cols].any(axis=1), 

                top_cols[idx]].sort_values(['flt3_inhibitor', 'pdgfr_inhibitor', 'kit_inhibitor']), 

                sort=None, color=(1,0,0))

targets.proteasome_inhibitor.sum()/len(targets)
msno.dendrogram(targets_visible)
from multiprocessing import cpu_count

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss, make_scorer



nproc = cpu_count()
X = train_features

y = targets.proteasome_inhibitor



X[['cp_type', 'cp_dose']] = X[['cp_type', 'cp_dose']].astype('category').apply(lambda x: x.cat.codes)



rf_model = RandomForestClassifier(n_estimators=100, random_state=10, verbose=True, n_jobs=nproc)

scorer = make_scorer(log_loss)

skf = StratifiedKFold(n_splits=5, random_state=24)

%%time



perm = PermutationImportance(rf_model, scoring=scorer, cv=skf)

perm.fit(X, y)
weights = eli5.explain_weights_dfs(perm, feature_names=X.columns.tolist())

weights_df = weights['feature_importances']

weights_df[:15]
from itertools import combinations



train_features_with = train_features.copy()

important = weights_df.loc[:5, 'feature'].tolist()

for pair in combinations(important, 2):

    col = "_".join(pair)

    train_features_with[col] = train_features_with[pair[0]] * train_features_with[pair[1]]



train_features_with[:5]

X_with = train_features_with

rf_model_with = RandomForestClassifier(n_estimators=100, random_state=10, verbose=True, n_jobs=nproc)

%%time



losses = np.zeros((2, 3, 5))

for i in range(3):

    skf = StratifiedKFold(n_splits=5, random_state=i*8)

    scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    losses[0,i] = cross_val_score(rf_model, X, y, scoring=scorer, cv=skf, n_jobs=nproc)

    losses[1,i] = cross_val_score(rf_model_with, X_with, y, scoring=scorer, cv=skf, n_jobs=nproc)
print(f"baseline: {np.mean(losses[0])} mean, {np.std(losses[0])} std dev "

      f"with features: {np.mean(losses[1])} mean, {np.std(losses[1])} std dev"

      )