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
from sklearn.datasets import load_iris

iris = load_iris()



dir(iris)
dir(iris)


df = pd.DataFrame(iris.data, columns = iris.feature_names)



df.head()
df.drop(["sepal length (cm)", "sepal width (cm)"], axis = 'columns', inplace = True)
from matplotlib import pyplot as plt
plt.scatter(df["petal length (cm)"], df["petal width (cm)"])

plt.xlabel("petal length (cm)")

plt.ylabel("petal width (cm)")
from sklearn.cluster import KMeans

km = KMeans(n_clusters = 3)
km
y_predicted = km.fit_predict(df[["petal length (cm)","petal width (cm)"]])

y_predicted
df['cluster'] = y_predicted
df
df1 = df[df.cluster == 0]

df2 = df[df.cluster == 1]

df3 = df[df.cluster == 2]

plt.scatter(df1["petal length (cm)"], df1["petal width (cm)"], color = 'green')

plt.scatter(df2["petal length (cm)"], df2["petal width (cm)"], color = 'orange')

plt.scatter(df3["petal length (cm)"], df3["petal width (cm)"], color = 'pink')

plt.xlabel("Petal length(cm)")

plt.ylabel("Petal width(cm)")

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color = 'purple', marker = '+', label = 'Centroid')

plt.legend()
k_rng = range(1, 10)

sse = []

for k in k_rng:

    km = KMeans(n_clusters = k)

    km.fit(df[["petal length (cm)", "petal width (cm)"]])

    sse.append(km.inertia_)

plt.xlabel('K')

plt.ylabel('Sum of Squared Errors')

plt.plot(k_rng, sse)
