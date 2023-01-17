import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from time import time

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.cluster import KMeans

from sklearn.datasets import load_digits

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

import matplotlib.pyplot as plt

import pandas

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets.samples_generator import (make_blobs,

                                                make_circles,

                                                make_moons)

from sklearn.cluster import KMeans, SpectralClustering

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_samples, silhouette_score
data = pd.read_csv("../input/heart.csv")
data.head()
data.isna().sum()
data['cp'].value_counts()
data['thal'].value_counts()
data = pd.get_dummies(data=data, columns=['cp', 'thal'])
data.head()
x = data.drop(['target'], axis=1)

y = data['target']
scaler = MinMaxScaler(feature_range=[0,1])

x = scaler.fit_transform(x)
sse = {}

data = pd.DataFrame()

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(x)

    data["clusters"] = kmeans.labels_

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
from keras.layers import Input, Dense

from keras.models import Model

from keras import regularizers

from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(x, test_size=0.1)
x_train.shape, x_test.shape
input_data = Input(shape=(19,))

encoded = Dense(128, activation='relu')(input_data)

encoded = Dense(64, activation='relu')(encoded)

encoded = Dense(16, activation='relu')(encoded)



encoded = Dense(2, activation='relu')(encoded)



decoded = Dense(16, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(decoded)

decoded = Dense(128, activation='relu')(decoded)

decoded = Dense(19, activation='sigmoid')(decoded)
autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



autoencoder.fit(x_train, x_train,

                epochs=10,

                batch_size=128,

                shuffle=True,

                validation_data=(x_test, x_test))
encoder = Model(input_data, encoded)
reduced_x_train = encoder.predict(x_train)

reduced_x_test = encoder.predict(x_test)
reduced_x_train.shape, reduced_x_test.shape
dummy = pd.DataFrame()



sse = {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(reduced_x_train)

    dummy["auto_clusters"] = kmeans.labels_

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()

#Using softmax to form clusters



input_data = Input(shape=(19,))

encoded = Dense(128, activation='relu')(input_data)

encoded = Dense(64, activation='relu')(encoded)

encoded = Dense(16, activation='relu')(encoded)



encoded = Dense(2, activation='softmax')(encoded)



decoded = Dense(16, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(decoded)

decoded = Dense(128, activation='relu')(decoded)

decoded = Dense(19, activation='sigmoid')(decoded)



autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



autoencoder.fit(x_train, x_train,

                epochs=10,

                batch_size=128,

                shuffle=True,

                validation_data=(x_test, x_test))
encoder = Model(input_data, encoded)

reduced_x_train = encoder.predict(x_train)

reduced_x_test = encoder.predict(x_test)
predict_clusters = np.argmax(reduced_x_test, axis=1)
predict_clusters
reduced_x_test