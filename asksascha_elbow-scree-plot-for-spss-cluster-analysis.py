import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv("../input/analysecluster2/canalyse2.csv", sep=";")



df.head(4)
data = df.copy()

categorical_features = ['Beruf']

continuous_features = ['Einkommen', 'Marke']

data[continuous_features].describe()
for col in categorical_features:

    dummies = pd.get_dummies(data[col], prefix=col)

    data = pd.concat([data, dummies], axis=1)

    data.drop(col, axis=1, inplace=True)

data.head()
mms = MinMaxScaler()

mms.fit(data)

data_transformed = mms.transform(data)
Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(data_transformed)

    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()