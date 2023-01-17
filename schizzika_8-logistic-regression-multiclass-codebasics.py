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
# import load_iris function from datasets module

# convention is to import modules instead of sklearn as a whole

from sklearn.datasets import load_iris
# save "bunch" object containing iris dataset and its attributes

# the data type is "bunch"

iris = load_iris()

type(iris)
iris
iris['target_names']
# print the iris data

# same data as shown previously

# each row represents each sample

# each column represents the features

print(iris.data)
# print the names of the four features

print(iris.feature_names)

# print integers representing the species of each observation

# 0, 1, and 2 represent different species

print(iris.target)
# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica

print(iris.target_names)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target, test_size=0.2)



model.fit(X_train, y_train)
model.score(X_test, y_test)
model.predict(X_test)
model.score(X_train, y_train)
y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)

cm
import seaborn as sn

import matplotlib.pyplot as plt

plt.figure(figsize = (10,7))

sn.heatmap(cm, annot=True)

plt.xlabel('Predicted')

plt.ylabel('Truth')