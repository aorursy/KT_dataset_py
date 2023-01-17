



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import sklearn.datasets as dt

dic = dt.load_digits()

dic.keys
dic.data.shape
dic.images.shape
import matplotlib.pyplot as plt

plt.imshow(dic.images[200])
X=dic.data

y=dic.target



dy =pd.DataFrame(y)

dy.nunique()

dy.columns
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Xss = ss.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xss,y,test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)

modelo = knn.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

y_score = modelo.score(X_test, y_test)

y_score