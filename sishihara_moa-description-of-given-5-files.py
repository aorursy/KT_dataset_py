import pandas as pd
train = pd.read_csv("../input/lish-moa/train_features.csv")

test = pd.read_csv("../input/lish-moa/test_features.csv")

train_targets_scored = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

train_targets_nonscored = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")

sub = pd.read_csv("../input/lish-moa/sample_submission.csv")
train.shape, test.shape
train.head()
sub.head()
train_targets_scored.shape
train_targets_scored.head()
train_targets_nonscored.shape
train_targets_nonscored.head()