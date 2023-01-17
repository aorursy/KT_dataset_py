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



import sklearn.datasets as dt

# Any results you write to the current directory are saved as output.
dic = dt.load_digits()
dic.keys()
dic.images.shape
import matplotlib.pyplot as plt

plt.imshow(dic.images[200])
dic.data
X=dic.data

Y=dic.target
X.shape

Y.shape
dy = pd.DataFrame(Y)

dy.nunique()

dy.columns
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

modelo = knn.fit(X_train,Y_train)

y_pred = modelo.predict(X_test)

y_score = modelo.score(X_test,Y_test)

y_score