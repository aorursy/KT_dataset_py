import pandas as pd
targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')



targets_scored.shape
targets_scored.head()
train_features = pd.read_csv('../input/lish-moa/train_features.csv')



train_features.shape
train_features.head()
train_features.cp_type.unique().tolist()
trt_cp_idx = train_features[train_features.cp_type == 'trt_cp'].index
targets_scored = targets_scored.iloc[trt_cp_idx.values.tolist()]



targets_scored.shape
import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
targets_scored.loc[:, 'fold'] = -1 # Create a new column `fold` containing `-1`s.

targets_scored = targets_scored.sample(frac=1).reset_index(drop=True) # Shuffle the rows.

targets = targets_scored.drop('sig_id', axis=1).values # Extract the targets as an array.
%%capture

mskf = MultilabelStratifiedKFold(n_splits=5)
for fold_, (train_, valid_) in enumerate(mskf.split(X=targets_scored, y=targets)):

    targets_scored.loc[valid_, 'fold'] = fold_

    

targets_scored.to_csv('./targets_with_folds.csv', index=False)