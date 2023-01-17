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
#Loading datset
train=pd.read_csv('../input/mnist_test.csv')
test=pd.read_csv('../input/mnist_train.csv')
X_train=train.drop('label',axis=1)
y_train=train['label']
X_test=test.drop('label',axis=1)
y_test=test['label']

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
accuracy
