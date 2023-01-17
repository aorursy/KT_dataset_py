# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.impute import SimpleImputer

from matplotlib import pyplot as plt

from matplotlib.pyplot import figure

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/ccdata/CC GENERAL.csv")
data.head()
imputer = SimpleImputer(missing_values=np.nan)

imputed_data = imputer.fit_transform(data[['MINIMUM_PAYMENTS']])

data['MINIMUM_PAYMENTS'] = imputed_data
data.head()
data = data.dropna()

data = data.drop('CUST_ID',axis=1)
columns=list(data.columns.values)
data.shape
inertia = []

i_s = [] 

for i in np.arange(1,10):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(data[['BALANCE_FREQUENCY','PURCHASES_FREQUENCY']])

    inertia.append(kmeans.inertia_)

    i_s.append(i)

plt.plot(i_s, inertia)
k_means = KMeans(n_clusters=3)

k_means.fit(data[['BALANCE_FREQUENCY','PURCHASES_FREQUENCY']])
k_means.cluster_centers_

xs = []

ys = []

for x in k_means.cluster_centers_:

    xs.append(x[0])

for y in k_means.cluster_centers_:

    ys.append(y[1])
k_means.labels_
plt.scatter(data.BALANCE_FREQUENCY, data.PURCHASES_FREQUENCY, c=k_means.labels_, s=10)

plt.scatter(xs, ys, marker='x', c='r')

plt.xlabel("BALANCE FREQ")

plt.ylabel("PURCHASE FREQ")
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

scaler = StandardScaler()

standard_data = scaler.fit_transform(data)

kmeans = KMeans(n_clusters = 6)

kmeans.fit(standard_data)

colors = ['red', 'blue', 'yellow', 'k','orange','green']

dist = 1-cosine_similarity(standard_data)



z = [colors[i] for i in kmeans.labels_]



pca = PCA(n_components=2, copy=True, random_state=0)

transformed_data = pca.fit_transform(dist)

x,y = transformed_data[:,0] , transformed_data[:,1]



plt.scatter(x,y,c=z, s=10)