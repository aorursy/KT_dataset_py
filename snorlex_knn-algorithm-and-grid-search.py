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
train_path = "../input/fashion-mnist_train.csv"

train_df = pd.read_csv(train_path)

print(train_df.shape)
target = train_df["label"].values.reshape(train_df.shape[0], 1)

train_df.drop("label", axis=1, inplace=True)

train = train_df.values

print(train.shape)
train = train.astype("float32") / 255
from sklearn.preprocessing import OneHotEncoder



enc = OneHotEncoder()

enc.fit(target)

target1 = enc.transform(target).toarray()

print(target1[:5])
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(train, target1, test_size=0.2, random_state=25)
from time import time

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



params = {"n_neighbors":[1, 3, 5, 7], "metric":["euclidean", "manhattan", "chebyshev"]}

acc = {}

i=0



for m in params["metric"]:

    acc[m] = []

    for k in params["n_neighbors"]:

        print("Model_{} metric: {}, n_neighbors: {}".format(i, m, k))

        i += 1

        t = time()

        knn = KNeighborsClassifier(n_neighbors=k, metric=m)

        knn.fit(train_x, train_y)

        pred = knn.predict(val_x)

        print("Time: ", time() - t)

        acc[m].append(accuracy_score(val_y, pred))

        print("Acc: ", acc[m][-1])
import matplotlib.pyplot as plt



c = ["r", "b", "g"]

for i, m in enumerate(params["metric"]):

    plt.plot(range(1, 9, 2), acc[m], c=c[i])

plt.show()