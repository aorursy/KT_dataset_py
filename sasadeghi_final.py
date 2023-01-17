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
import pandas as pd
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
X_train = train.iloc[:, 2:-1].values
Y_train = train.iloc[:, 21].values
X_test = test.iloc[:, 2:].values
X_all = np.concatenate((X_train, X_test), axis=0)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_all)
X_all = imputer.transform(X_all)
X_train = X_all[0:900,:]
X_test = X_all[900:1340,:]
X = X_train
Y = Y_train
from sklearn.preprocessing import QuantileTransformer
sc = QuantileTransformer()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)
from sklearn.kernel_approximation import AdditiveChi2Sampler
sc = AdditiveChi2Sampler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators = 60 , learning_rate = 0.3)
clf.fit(X, Y)

Y_pred = clf.predict(X)
from sklearn.metrics import accuracy_score
accuracy_score(Y, Y_pred)
Y_test_pred = clf.predict(X_test)
cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': Y_test_pred }
submission = pd.DataFrame(cols)
print(submission)

submission.to_csv("submission.csv", index=False)
