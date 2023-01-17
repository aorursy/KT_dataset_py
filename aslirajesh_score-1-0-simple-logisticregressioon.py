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
dataset = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
dataset.head()
dataset.shape
X = pd.get_dummies(dataset.iloc[0:,1:],drop_first=True)

dataset["class"] = dataset["class"].map({"e":1,"p":0})

Y = dataset['class']
X.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33)
from sklearn.linear_model import LogisticRegression

l_regressor = LogisticRegression(random_state = 0)

l_regressor.fit(X_train, y_train)
y_pred = l_regressor.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)