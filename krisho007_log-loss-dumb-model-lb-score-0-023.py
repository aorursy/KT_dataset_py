import pandas as pd

import numpy as np
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

train_targets_positive = train_targets_scored.sum()[1:]

train_targets_positive.sort_values(ascending=True)
remainder = train_targets_positive % 6

number_to_be_added = 6 - remainder

number_to_be_added = number_to_be_added % 6

train_targets_positive_corrected = train_targets_positive + number_to_be_added
dumb_pred_for_each_class = train_targets_positive_corrected/(23814 + number_to_be_added.sum())
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
submission.iloc[:, 1:] = dumb_pred_for_each_class.values
# https://www.kaggle.com/c/lish-moa/discussion/180165 

vehicle_indices = test_features[test_features["cp_type"]=="ctl_vehicle"].index.tolist()

submission.iloc[vehicle_indices, 1:] = np.zeros((1, 206))
submission
submission.to_csv("submission.csv", index=False)