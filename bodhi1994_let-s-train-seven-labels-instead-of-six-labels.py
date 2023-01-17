import numpy as np
import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train_labels = train.iloc[:, [2, 3, 4, 5, 6, 7]]
train_labels.head()
mask = (train_labels.sum(axis=1) == 0)
train_labels['non_toxic'] = mask.astype(int)
train_labels.head(10)
X_train = train["comment_text"].fillna("fillna").values
y_train = train_labels.values
X_test = test["comment_text"].values
