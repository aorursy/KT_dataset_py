import pandas as pd

import numpy as np

import pickle
test_ids = pd.read_csv("../input/ieee-sol1/test_ids.csv")
sub = pd.read_csv("../input/ieee-sol1/blend_of_blends_1.csv")
sub["uid"] = test_ids["uid3"].values
sub["isFraud"] = sub["uid"].map(sub.groupby("uid")["isFraud"].quantile(0.9))
del sub["uid"]
sub.head()
sub.to_csv('submission.csv', index=False)
