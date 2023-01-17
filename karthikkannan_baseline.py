# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

test_features = test_df.iloc[:, 1:]

test_ids = test_df["id"]
features = train_df.iloc[:,2:]

target = train_df["target"]
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(features, target)
preds = rf.predict(test_features)
preds_df = pd.DataFrame(list(zip(test_ids, preds)), columns = ["id", "target"])
preds_df.to_csv("./baseline_submission.csv", index=False)