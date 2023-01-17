# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dt = pd.read_csv("../input/heart.csv")
dt.head()
X_train, X_test, y_train, y_test = train_test_split(dt.drop('target', 1), dt['target'], test_size = 35, random_state=129)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = np.mean(pred == y_test)

print('accuracy: ', accuracy*100, '%')
model = RandomForestClassifier(random_state=39, n_estimators=137)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = np.mean(pred == y_test)

print('accuracy: ', accuracy*100, '%')