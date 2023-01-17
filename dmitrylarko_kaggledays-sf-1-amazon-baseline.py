import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# Loading data directly from CatBoost

from catboost.datasets import amazon



train, test = amazon()
print("Train shape: {}, Test shape: {}".format(train.shape, test.shape))
train.head(5)
test.head(5)
train.apply(lambda x: len(x.unique()))
import itertools

target = "ACTION"

col4train = [x for x in train.columns if x!=target]



col1 = 'ROLE_CODE'

col2 = 'ROLE_TITLE'



pair = len(train.groupby([col1,col2]).size())

single = len(train.groupby([col1]).size())



print(col1, col2, pair, single)
col4train = [x for x in col4train if x!='ROLE_TITLE']
#linear - OHE

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
X = ohe.fit_transform(train[col4train])

y = train["ACTION"].values
from sklearn.model_selection import cross_validate



model = LogisticRegression(

                penalty='l2',  

                C=1.0, 

                fit_intercept=True, 

                random_state=432,

                solver = 'liblinear',

                max_iter = 1000,

        )

stats = cross_validate(model, X, y, groups=None, scoring='roc_auc', 

                       cv=5, n_jobs=2, return_train_score = True)

stats = pd.DataFrame(stats)

stats.describe().transpose()
X = ohe.fit_transform(train[col4train])

y = train["ACTION"].values

X_te = ohe.transform(test[col4train])



model.fit(X,y)

predictions = model.predict_proba(X_te)[:,1]



submit = pd.DataFrame()

submit["Id"] = test["id"]

submit["ACTION"] = predictions



submit.to_csv("submission.csv", index = False)