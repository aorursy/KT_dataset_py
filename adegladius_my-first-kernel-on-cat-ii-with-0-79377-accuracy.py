# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load dependencies

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
train = pd.read_csv('/kaggle/input/categorical-data/train.csv')

test = pd.read_csv('/kaggle/input/categorical-data/test.csv')



labels = train.pop('target') 

labels = labels.values



train_id = train.pop("id")

test_id = test.pop("id")
train.shape, test.shape
train.head(5)
data = pd.concat([train, test])

data["ord_5a"] = data["ord_5"].str[0]

data["ord_5b"] = data["ord_5"].str[1]
columns = [i for i in data.columns]



dummies = pd.get_dummies(data,

                         columns=columns,

                         drop_first=True,

                         sparse=True)



del data
train = dummies.iloc[:train.shape[0], :]

test = dummies.iloc[train.shape[0]:, :]



del dummies
train = train.fillna(0)

train.head(5)
print(train.shape)

print(test.shape)
train = train.sparse.to_coo().tocsr()

test = test.sparse.to_coo().tocsr()
lr_cv = LogisticRegressionCV(Cs=7,

                        solver="lbfgs",

                        tol=0.0001,

                        max_iter=30000,

                        cv=5)



lr_cv.fit(train, labels)



lr_cv_pred = lr_cv.predict_proba(train)[:, 1]

score = roc_auc_score(labels, lr_cv_pred)



print("score: ", score)
test_pred = lr_cv.predict_proba(test)[:, 1]
submiss = pd.DataFrame({"id": test_id, "target": test_pred})

submiss.to_csv('Submission(LogisticRegression).csv', index=False)