import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split



# Read data

train = pd.read_csv('../input/train.csv')

y = train['label'].values

X = train[train.columns[1:]].values

X_test = pd.read_csv('../input/test.csv').values



# split data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=42)

print(X_train.shape)

print(X_val.shape)
from sklearn import svm



clf = svm.SVC(kernel='poly', C=100, gamma='auto', degree=3, coef0=1, decision_function_shape='ovo')

clf.fit(X_train, y_train)



y_pred = clf.predict(X_val)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import balanced_accuracy_score



print(confusion_matrix(y_val, y_pred))

print(balanced_accuracy_score(y_val, y_pred))
y_pred_test = clf.predict(X_test)
# output result

dataframe = pd.DataFrame({"ImageId": list(range(1,len(y_pred_test)+1)), "Label": y_pred_test})

dataframe.to_csv('output.csv', index=False, header=True)