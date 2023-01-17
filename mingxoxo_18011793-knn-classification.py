import pandas as pd

import numpy as np



train = pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv')

train_x = train.loc[:, '0':'7']

train_y = train['8']



print(train.shape)

print(train_x.shape)

print(train_y.shape)

train_x = np.array(train_x)

train_y = np.array(train_y)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5, p = 2)

knn.fit(train_x, train_y)
#예측

y_train_pred = knn.predict(train_x)

print((train_y!=y_train_pred).sum()) #틀린 갯수 확인
test = pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv')

test_x = test.loc[:, '0':'7']

test_x = np.array(test_x)

y_test_pred = knn.predict(test_x)

y_test_pred
submit = pd.read_csv('../input/logistic-classification-diabetes-knn/submission_form.csv')

for i in range(len(y_test_pred)):

  submit['Label'][i] = y_test_pred[i]

submit['Label'] =submit['Label'].astype(int)

submit.to_csv("submission.csv", index = False, header = True)