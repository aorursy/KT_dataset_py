import pandas as pd

import numpy as np



import xgboost as xgb

from sklearn.cross_validation import cross_val_score

from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelBinarizer
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
Y_train = train['species']

id_train = train['id']

X_train = train.drop(['species', 'id'], axis=1)

classes = np.unique(Y_train)

Y_train_bin = LabelBinarizer().fit_transform(Y_train)



id_test = test['id']

X_test = test.drop(['id'], axis=1)
lb = LabelBinarizer()

lb.fit(Y_train)

print(lb.classes_)
"Number of unique classes: {0}".format(len(classes))
xb = xgb.XGBClassifier(n_estimators=500, objective='multi:softprob')

probas = xb.fit(X_train, Y_train).predict_proba(X_test)
result = pd.DataFrame(index=id_test.values, columns=xb.classes_, data=probas)

result.to_csv("./xgb_benchmark.csv", index_label='id')
! head -n 2 xgb_benchmark.csv
cross_val_score(xgb.XGBClassifier(n_estimators=100), X_train, Y_train, cv=5)