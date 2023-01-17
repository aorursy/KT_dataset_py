# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing,CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import svm
from sklearn import datasets as ds
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
%matplotlib inline
data=ds.load_digits()
print(data.DESCR)
data.images[0]
plt.imshow(data.images[0])
print(data.target)
print(data.target.size)
svc = svm.SVC(gamma=0.001, C=100.)
#1st training
X_train=data.data[0:1791]
Y_train=data.target[0:1791]
svc.fit(X_train,Y_train)
#1st prediction
X_test=data.data[1791:]
Y_test=data.target[1791:]
Y_pred=svc.predict(X_test)
for i in range(6):
    plt.subplot(321+i)
    plt.imshow(data.images[1791+i])
    print("The digit is {}".format(Y_test[i]))
accuracy_score(Y_test, Y_pred)
#2nd training
X_train=data.data[6:]
Y_train=data.target[6:]
svc.fit(X_train,Y_train)
#2nd prediction
X_test=data.data[:6]
Y_test=data.target[:6]
Y_pred=svc.predict(X_test)
for i in range(6):
    plt.subplot(321+i)
    plt.imshow(data.images[i])
    print("The digit is {}".format(Y_test[i]))
accuracy_score(Y_test, Y_pred)
#3rd training
X_train=data.data[:1785]
Y_train=data.target[:1785]
svc.fit(X_train,Y_train)
#3rd prediction
X_test= data.data[1785:1791]
Y_test= data.target[1785:1791]
Y_pred= svc.predict(X_test)
for i in range(6):
    plt.subplot(321+i)
    plt.imshow(data.images[1785+i])
    print("The Digit is {}".format(Y_pred[i]))
accuracy_score(Y_test,Y_pred)