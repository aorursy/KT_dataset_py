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
from sklearn.datasets import load_iris

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.dummy import DummyClassifier



from sklearn.model_selection import train_test_split
iri = pd.read_csv('../input/Iris.csv')
iri.info()
feats = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']



x=iri[feats].values

y=iri['Species'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = DecisionTreeClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test,y_pred))
model = RandomForestClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test,y_pred))