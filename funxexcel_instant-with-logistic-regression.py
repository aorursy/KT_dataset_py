import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

train.info()
test = pd.read_csv("../input/test.csv")

test.info()
X = train.drop(['id','target'], axis = 1)

X.head(1)
y = train['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver = 'liblinear')
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

predictions[:5]
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score



print(accuracy_score(y_test,predictions))

print("\n")

print(confusion_matrix(y_test,predictions))

print("\n")

print(classification_report(y_test,predictions))
TestForPred = test.drop(['id'], axis = 1)
t_pred = logmodel.predict(TestForPred).astype(int)
id = test['id']
logSub = pd.DataFrame({'id': id, 'target':t_pred })

logSub.head()
logSub.to_csv("1_Logistics_Regression_Submission.csv", index = False)