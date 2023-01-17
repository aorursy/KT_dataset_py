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
from seaborn import pairplot 
import pandas as pd
data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.drop("species", axis=1), data["species"],test_size=0.3)
print(x_train.shape)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf = clf.fit(x_train, y_train)

clf
hasil_prediksi = clf.predict(x_test)

hasil_prediksi