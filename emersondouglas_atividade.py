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
import sklearn.datasets as dt
dic = dt.load_digits()

dic.keys()
import matplotlib.pyplot as plt

plt.imshow(dic.images[200])
X = dic.data

y = dic.target

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier

Knn= KNeighborsClassifier(n_neighbors=3)

modelo=Knn.fit(X_train,y_train)

y_pred=modelo.predict(X_test)

y_score=modelo.score(X_test,y_test)

print(y_score)
