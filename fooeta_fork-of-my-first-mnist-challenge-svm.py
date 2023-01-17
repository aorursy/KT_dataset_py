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
train_path = os.path.join("../input", "train.csv")
digits_train = pd.read_csv(train_path)
digits_train.head(5)
X = digits_train.drop('label', axis=1)
y = digits_train['label'].copy()

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


print("Train set shape: {}".format(X_train.shape))
svm = SVC(kernel='poly')
svm.fit(X_train, y_train)
print("Train set score: {:.2f}".format(svm.score(X_train, y_train)))
y_train_pred = svm.predict(X_train)
from sklearn.metrics import confusion_matrix
conf_mx = confusion_matrix(y_train, y_train_pred)
import matplotlib.pyplot as plt
%matplotlib inline

plt.matshow(conf_mx, cmap=plt.cm.gray)
print("Test set score: {:.2f}".format(svm.score(X_test, y_test)))
test_path = os.path.join("../input", "test.csv")
digit_test = pd.read_csv(test_path)
digit_test.head(5)
digit_test.shape
digit_pred = svm.predict(digit_test)
digit_pred
import csv

try:
    os.remove("result.csv")
except FileNotFoundError:
    print("result.csv not found.")

csvfile = open("result.csv", "w", encoding="utf-8")

writer = csv.writer(csvfile)

writer.writerow(["ImageId","Label"])

for i, digits in enumerate(digit_pred):
    writer.writerow([i+1, digits])
    
csvfile.close()