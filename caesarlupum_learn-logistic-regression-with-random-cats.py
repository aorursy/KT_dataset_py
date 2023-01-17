import numpy as np 

import pandas as pd 

import gc

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import HTML

pd.set_option('max_columns', 50)
HTML('<iframe width="921" height="587" src="https://www.youtube.com/embed/zM4VZR0px8E" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>' )
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
labels = train.pop('target')

train_id = train.pop("id")

test_id = test.pop("id")
labels = labels.values

data = pd.concat([train, test])



columns = [i for i in data.columns]

dummies = pd.get_dummies(data,columns=columns, drop_first=True,sparse=True)



train = dummies.iloc[:train.shape[0], :]

test = dummies.iloc[train.shape[0]:, :]



del dummies,data

gc.collect()



train = train.sparse.to_coo().tocsr()

test = test.sparse.to_coo().tocsr()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.20, solver="lbfgs", tol=0.020, max_iter=2020)



lr.fit(train, labels)

lr_pred = lr.predict_proba(train)[:, 1]

lr_pred
from sklearn.metrics import roc_auc_score

score = roc_auc_score(labels, lr_pred)



print(f"{score:.6f}")
submission = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

submission["id"] = test_id

submission["target"] = lr.predict_proba(test)[:, 1]

submission.to_csv("submission.csv", index=False)
submission.head()