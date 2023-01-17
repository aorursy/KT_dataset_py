import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train_v2.csv")

train.head()
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier()

clf.fit(X=train[['C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'amount']], y=train['isfraud'])
test = pd.read_csv("../input/test_v2.csv")
preds = clf.predict_proba(test[['C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'amount']])

preds = [ x[1] for x in preds ]

preds = pd.concat([test['id'],pd.Series(preds)], axis=1)

preds.columns = [['id', 'isfraud']]
preds.to_csv("submission_dt_baseline_v2.csv", index=None)