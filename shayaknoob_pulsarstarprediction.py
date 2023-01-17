# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/pulsar_stars.csv")

print(df.head())

values = df.values

print(values.shape)

X = values[:, 0:7]

Y = values[:, 8]
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=33)

scaler = StandardScaler().fit(X_train)

s_X = scaler.transform(X_train)

s_X_test = scaler.transform(X_test)
svc = SVC(kernel='rbf')

svc.fit(s_X, y_train)
y_pred = svc.predict(s_X_test)

accuracy_score(y_test, y_pred)