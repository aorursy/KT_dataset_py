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
dataset = pd.read_csv("../input/train.csv")
dataset
X = dataset.iloc[:, dataset.columns !="label"]
y = dataset.iloc[:, dataset.columns == "label"]
y.info()
X = X / 255.0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
from sklearn import svm
classifier = svm.SVC(gamma='scale')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
classifier.score(X_test, y_test)
import matplotlib.pyplot as plt
array = np.random.randint(10, size=6)
for i in array:
    plt.imshow(X_test[i].reshape(28,28))
    plt.xlabel("Prediction : {} / Real_value : {}".format(y_pred[i], y_test.values[i]))
    plt.show()
errors = []
for i in range(0, len(y_test)):
    if y_test.values[i] - y_pred[i] != 0:
        errors += [i]
for i in errors:
    plt.imshow(X_test[i].reshape(28,28))
    plt.xlabel("Prediction : {} / Real_value : {}".format(y_pred[i], y_test.values[i]))
    plt.show()
test = pd.read_csv("../input/test.csv")
test = test / 255.0
test = sc_X.fit_transform(test)
prediction = classifier.predict(test)
ImageId = pd.Series(np.arange(0, len(prediction)), name="ImageId")
submission = pd.DataFrame(data = [ImageId, prediction]).transpose()
submission = submission.rename(columns = {"Unnamed 0":"Label"})
submission.to_csv("submission.csv")
