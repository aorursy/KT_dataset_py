import os
print(os.listdir("../input"))

import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train.shape
# заполняем Y из датасета и удаляем его из таблицы
Y  = train['label'].values
train.drop(['label'], axis=1, inplace=True)
# оставшиеся данные кидаем в матрицу X
X = train.values
X.shape
X[0]
img_0 = X[0].reshape( (28,28))
img_1 = X[1].reshape((28,28))
plt.figure()
plt.imshow(img_0, cmap='gray')
plt.show()
plt.figure()
plt.imshow(img_1, cmap='gray')
plt.show()
Y[0]
Y[1]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train.shape
X = X / 255.0
X_valid = test_data.values
X_valid[0]
X_train = X_train / 255.0
X_train.shape
Y_train.shape
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=9, n_estimators=2000, n_jobs=-1, verbose=1)
clf.fit(X_train, Y_train)
from sklearn.metrics import accuracy_score
Y_predict = clf.predict(X_test)
accuracy_score(Y_test, Y_predict)
predict = clf.predict(test_data)
predict
sample = pd.read_csv('sample_submission.csv')
sample.shape

predict.shape
import numpy as np

sample
l = sample['ImageId'].values
l
predict
type(predict)
import csv
type(sample)
sample_array = sample['ImageId'].values
type(sample_array)
sample_array[0]
predict = pd.Series(predict,name="Label")
predict.shape
s2 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)

s2.to_csv('out.csv')
s2.shape
