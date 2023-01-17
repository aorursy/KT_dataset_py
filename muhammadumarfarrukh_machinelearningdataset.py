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

from keras.layers import Input, Dense

from keras.models import Model

from keras import regularizers

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.metrics.cluster import completeness_score

from sklearn.metrics import confusion_matrix
data = pd.read_csv("../input/ML dataset.csv")
data.isna().sum()
# fig_size = plt.rcParams["figure.figsize"]

# fig_size[0] = 35

# fig_size[1] = 18

# data.hist(bins=20)
x = data.drop(['Class'], axis=1)

y = data['Class']
scaler = MinMaxScaler(feature_range=[0,1])

x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3)

x_test.shape, x_train.shape,y_test.shape, y_train.shape
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42,max_iter = 1000)

    kmeans.fit(x_train)

    # inertia method returns wcss for that model

    wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10,5))



plt.figure(figsize=(10,5))

sns.lineplot(range(1, 11), wcss,marker='o',color='red')

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('SSE')

plt.show()
from scipy.spatial.distance import cdist

distortions = []

K = range(1,10)

for k in K:

    kmeanModel = KMeans(n_clusters = k, init = 'k-means++', random_state = 42,max_iter = 1000).fit(x_train)

    kmeanModel.fit(x_train)

    distortions.append(sum(np.min(cdist(x_train, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / x_train.shape[0])



# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
for i, k in enumerate([2, 3, 4]):

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(18, 7)

    

    # Run the Kmeans algorithm

    km = KMeans(n_clusters=k, max_iter = 1000)

    labels = km.fit_predict(x_train)

    y_label_test = km.predict(x_test)

    centroids = km.cluster_centers_



    # Get silhouette samples

    silhouette_vals = silhouette_samples(x_train, labels)

    

    

    

    # Silhouette plot

    y_ticks = []

    y_lower, y_upper = 0, 0

    for i, cluster in enumerate(np.unique(labels)):

        cluster_silhouette_vals = silhouette_vals[labels == cluster]

        cluster_silhouette_vals.sort()

        y_upper += len(cluster_silhouette_vals)

        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)

        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))

        y_lower += len(cluster_silhouette_vals)



    # Get the average silhouette score and plot it

    avg_score = np.mean(silhouette_vals)

    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')

    ax1.set_yticks([])

    ax1.set_xlim([-0.1, 1])

    ax1.set_xlabel('Silhouette coefficient values')

    ax1.set_ylabel('Cluster labels')

    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    

    # Scatter plot of data colored with labels

    ax2.scatter(x_train[:, 0], x_train[:, 1], c=labels)

    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)

    ax2.set_xlim([-2, 2])

    ax2.set_xlim([-2, 2])

    ax2.set_xlabel('Eruption time in mins')

    ax2.set_ylabel('Waiting time to next eruption')

    ax2.set_title('Visualization of clustered data', y=1.02)

    ax2.set_aspect('equal')

    plt.tight_layout()

    plt.suptitle(f'Silhouette analysis using k = {k}',

                 fontsize=16, fontweight='semibold', y=1.05);

   

    silhouette_avg = silhouette_score(x_train, labels)

    print("For n_clusters =", k,

          "The average silhouette_score is :", silhouette_avg)

    #y_label_train = kmeans.labels_

    accurcyscore = accuracy_score(y_test,y_label_test)

    completenessscore =completeness_score(y_test, y_label_test)

    print("Accuracy Score ",accurcyscore)

    print("Completeness Score ",completenessscore)
input_data = Input(shape=(14,))

encoded = Dense(14, activation='relu')(input_data)

encoded = Dense(10, activation='relu')(encoded)

encoded = Dense(5, activation='relu')(encoded)



encoded = Dense(2, activation='relu')(encoded)



decoded = Dense(5, activation='relu')(encoded)

decoded = Dense(10, activation='relu')(decoded)



decoded = Dense(14, activation='sigmoid')(decoded)
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
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(encoder).create(prog='dot', format='svg'))
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42,max_iter = 1000)

    kmeans.fit(reduced_x_train)

    # inertia method returns wcss for that model

    wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10,5))



plt.figure(figsize=(10,5))

sns.lineplot(range(1, 11), wcss,marker='o',color='red')

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('SSE')

plt.show()
from scipy.spatial.distance import cdist

distortions = []

K = range(1,10)

for k in K:

    kmeanModel = KMeans(n_clusters = k, init = 'k-means++', random_state = 42,max_iter = 1000).fit(reduced_x_train)

    kmeanModel.fit(reduced_x_train)

    distortions.append(sum(np.min(cdist(reduced_x_train, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / reduced_x_train.shape[0])



# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
for i, k in enumerate([2, 3, 4]):

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(18, 7)

    

    # Run the Kmeans algorithm

    km = KMeans(n_clusters=k, max_iter = 1000)

    labels = km.fit_predict(reduced_x_train)

    y_label_test = km.predict(reduced_x_test)

    centroids = km.cluster_centers_



    # Get silhouette samples

    silhouette_vals = silhouette_samples(reduced_x_train, labels)

    

    

    

    # Silhouette plot

    y_ticks = []

    y_lower, y_upper = 0, 0

    for i, cluster in enumerate(np.unique(labels)):

        cluster_silhouette_vals = silhouette_vals[labels == cluster]

        cluster_silhouette_vals.sort()

        y_upper += len(cluster_silhouette_vals)

        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)

        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))

        y_lower += len(cluster_silhouette_vals)



    # Get the average silhouette score and plot it

    avg_score = np.mean(silhouette_vals)

    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')

    ax1.set_yticks([])

    ax1.set_xlim([-0.1, 1])

    ax1.set_xlabel('Silhouette coefficient values')

    ax1.set_ylabel('Cluster labels')

    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    

    # Scatter plot of data colored with labels

    ax2.scatter(reduced_x_train[:, 0], reduced_x_train[:, 1], c=labels)

    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)

    ax2.set_xlim([-2, 2])

    ax2.set_xlim([-2, 2])

    ax2.set_xlabel('Eruption time in mins')

    ax2.set_ylabel('Waiting time to next eruption')

    ax2.set_title('Visualization of clustered data', y=1.02)

    ax2.set_aspect('equal')

    plt.tight_layout()

    plt.suptitle(f'Silhouette analysis using k = {k}',

                 fontsize=16, fontweight='semibold', y=1.05);

   

    silhouette_avg = silhouette_score(reduced_x_train, labels)

    print("For n_clusters =", k,

          "The average silhouette_score is :", silhouette_avg)

    #y_label_train = kmeans.labels_

    accurcyscore = accuracy_score(y_test,y_label_test)

    completenessscore =completeness_score(y_test, y_label_test)

    print("Accuracy Score ",accurcyscore)

    print("Completeness Score ",completenessscore)
input_data = Input(shape=(14,))

encoded = Dense(14, activation='relu')(input_data)

encoded = Dense(10, activation='relu')(encoded)

encoded = Dense(5, activation='relu')(encoded)



encoded = Dense(2, activation='softmax')(encoded)



decoded = Dense(5, activation='relu')(encoded)

decoded = Dense(10, activation='relu')(decoded)



decoded = Dense(14, activation='sigmoid')(decoded)







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
predict_clusters = np.argmax(reduced_x_test, axis=1)
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(encoder).create(prog='dot', format='svg'))
sse = []

list_k = list(range(1, 10))



for k in list_k:

    km = KMeans(n_clusters=k)

    km.fit(reduced_x_train)

    sse.append(km.inertia_)



# Plot sse against k

plt.figure(figsize=(6, 6))

plt.plot(list_k, sse, '-o')

plt.xlabel(r'Number of clusters *k*')

plt.ylabel('Sum of squared distance');
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42,max_iter = 1000)

    kmeans.fit(reduced_x_train)

    # inertia method returns wcss for that model

    wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10,5))



plt.figure(figsize=(10,5))

sns.lineplot(range(1, 11), wcss,marker='o',color='red')

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('SSE')

plt.show()
from scipy.spatial.distance import cdist

distortions = []

K = range(1,10)

for k in K:

    kmeanModel = KMeans(n_clusters = k, init = 'k-means++', random_state = 42,max_iter = 1000).fit(reduced_x_train)

    kmeanModel.fit(reduced_x_train)

    distortions.append(sum(np.min(cdist(reduced_x_train, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / reduced_x_train.shape[0])



# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
for i, k in enumerate([2, 3, 4]):

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(18, 7)

    

    # Run the Kmeans algorithm

    km = KMeans(n_clusters=k, max_iter = 1000)

    labels = km.fit_predict(reduced_x_train)

    y_label_test = km.predict(reduced_x_test)

    centroids = km.cluster_centers_



    # Get silhouette samples

    silhouette_vals = silhouette_samples(reduced_x_train, labels)

    

    

    

    # Silhouette plot

    y_ticks = []

    y_lower, y_upper = 0, 0

    for i, cluster in enumerate(np.unique(labels)):

        cluster_silhouette_vals = silhouette_vals[labels == cluster]

        cluster_silhouette_vals.sort()

        y_upper += len(cluster_silhouette_vals)

        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)

        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))

        y_lower += len(cluster_silhouette_vals)



    # Get the average silhouette score and plot it

    avg_score = np.mean(silhouette_vals)

    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')

    ax1.set_yticks([])

    ax1.set_xlim([-0.1, 1])

    ax1.set_xlabel('Silhouette coefficient values')

    ax1.set_ylabel('Cluster labels')

    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    

    # Scatter plot of data colored with labels

    ax2.scatter(reduced_x_train[:, 0], reduced_x_train[:, 1], c=labels)

    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)

    ax2.set_xlim([-2, 2])

    ax2.set_xlim([-2, 2])

    ax2.set_xlabel('Eruption time in mins')

    ax2.set_ylabel('Waiting time to next eruption')

    ax2.set_title('Visualization of clustered data', y=1.02)

    ax2.set_aspect('equal')

    plt.tight_layout()

    plt.suptitle(f'Silhouette analysis using k = {k}',

                 fontsize=16, fontweight='semibold', y=1.05);

   

    silhouette_avg = silhouette_score(reduced_x_train, labels)

    print("For n_clusters =", k,

          "The average silhouette_score is :", silhouette_avg)

    #y_label_train = kmeans.labels_

    accurcyscore = accuracy_score(y_test,y_label_test)

    completenessscore =completeness_score(y_test, y_label_test)

    print("Accuracy Score ",accurcyscore)

    print("Completeness Score ",completenessscore)