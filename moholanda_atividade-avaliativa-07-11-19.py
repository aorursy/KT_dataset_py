import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sn 

sn.set()
import pandas as pd

cars = pd.read_csv("../input/carsdata/cars.csv", na_values=' ')
cars.head()
cars.columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60',

       'year', 'brand']
cars= cars.dropna()

cars['cubicinches'] = cars['cubicinches'].astype(int)

cars['weightlbs'] = cars['weightlbs'].astype(int)
x = cars.iloc[:,0:7]

x.head()
x.describe()
x_array = x.values 

x_array
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)

    kmeans.fit(x_array)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)

plt.title('Metodo Cotuvelo - Elbow Method')

plt.xlabel('Número de Clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)

cars['clusters'] = kmeans.fit_predict(x_array)

cars.head()

cars.groupby("clusters").agg('mean').plot.bar(figsize=(10,7.5))

plt.title("Gastos por Cluster")