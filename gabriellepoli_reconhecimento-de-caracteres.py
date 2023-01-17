

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sklearn.datasets as dt
dic = dt.load_digits()

dic.keys()
dic.data
dic.data.shape
dic.images.shape
import matplotlib.pyplot as plt

plt.imshow(dic.images[200])
dic.target_names
X = dic.data

y = dic.target
X.shape
y.shape
dy = pd.DataFrame(y)

dy.nunique()

dy.columns
dy[0].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)

modelo = knn.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

y_score = modelo.score(X_test, y_test)

y_score
compara = pd.DataFrame(y_test)

compara['pred'] = y_pred

compara.head(100)