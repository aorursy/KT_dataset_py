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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

clf = DecisionTreeClassifier()
xtrain = train.iloc[0:42000,1:]
train_label = train.iloc[0:42000,0]
clf.fit(xtrain, train_label)
xtest = test.iloc[0:28000,:]
p = clf.predict(xtest)
X_train = (train.iloc[:,1:]).astype('float32') # all pixel values
y_train = train.iloc[:,0].astype('int32') # only labels i.e targets digits
X_test = test.astype('float32')
X_train = X_train.values.reshape(X_train.shape[0], 28, 28)
j=0
for i in range(30000, 30003):
    plt.subplot(330 + (j+1))
    j+=1
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);
X_test = X_test.values.reshape(X_test.shape[0], 28, 28)
j=0
for i in range(1004, 1007):
    plt.subplot(330 + (j+1))
    j+=1
    plt.imshow(X_test[i], cmap=plt.get_cmap('gray'))
    plt.title(p[i]);
