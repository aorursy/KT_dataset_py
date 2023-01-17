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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
datatrain = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

datatest = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
x = datatrain.iloc[:, 2:].values

y = datatrain.target.values

x_test = datatest.iloc[:, 1:].values
x_train = x

y_train = y
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()





x_train[:,0] = le.fit_transform(x_train[:,0])

x_test[:,0] = le.fit_transform(x_test[:,0])

knn = KNeighborsClassifier(11)
knn.fit(x_train, y_train)
y_preds = knn.predict(x_test)
y_preds
pd.concat([datatest.ID_code, pd.Series(y_preds).rename('target')], axis = 1).to_csv('submission_knn.csv', index =False)