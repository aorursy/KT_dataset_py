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
from PIL import Image, ImageDraw

from sklearn import datasets
test = pd.read_csv('/kaggle/input/digit-recog/test.csv')
train= pd.read_csv('/kaggle/input/digit-recog/train.csv')
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score
print("Loading dataset...")

clf = KNeighborsClassifier(n_neighbors=5)

train_x = test[:10000]

train_y = train[:10000]

print("Train model")

clf.fit(train_x, train_y)

test_x = test[10000:10100]

expected = train[10000:10100].values.tolist()

print("Compute predictions")

predicted = clf.predict(test_x)

print(predicted)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(expected,predicted)/100)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
clfs = RandomForestClassifier(n_estimators=5)
train_x = test[:10000]

train_y = train[:10000]
print("Train model")

clfs.fit(train_x, train_y)
test_x = test[10000:11000]

expected = train[10000:11000].values.tolist()
print("Compute predictions")

predicted = clf.predict(test_x)
print(mean_squared_error(expected,predicted)/100)