# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



#load the iris dataset

iris = datasets.load_iris()

X = iris.data[:,[0,2]]

y = iris.target



print('Class labels:', np.unique(y))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)



sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)





ppn = Perceptron(max_iter = 40, eta0=0.1, random_state = 1)

ppn.fit(X_train_std, y_train)



y_pred = ppn.predict(X_test_std)

print('Accuracy: %2f' %accuracy_score(y_test, y_pred))
