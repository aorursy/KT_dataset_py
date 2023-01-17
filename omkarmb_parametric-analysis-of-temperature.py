!pip install tslearn
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
seed = 0
numpy.random.seed(seed)
# read data
df = pd.read_csv('/kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv')
df

X_train=df['ambient'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")

X_train=df['coolant'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
X_train=df['u_d'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
X_train=df['u_q'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
X_train=df['motor_speed'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
X_train=df['torque'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
X_train=df['i_d'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
X_train=df['i_q'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
X_train=df['pm'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
X_train=df['stator_yoke'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
X_train=df['stator_tooth'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
X_train=df['stator_winding'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")

X_train=df['profile_id'].values
X_train=X_train[:998000]
X_train=np.reshape(X_train, (998, 1000))
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=1000).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=36, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

#plt.figure()
plt.figure(figsize=(50,25))
for yi in range(36):

    plt.subplot(6, 6, yi + 1)
    
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")

# DBA-k-means
print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=3,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
y_pred = dba_km.fit_predict(X_train)

for yi in range(3):
    plt.subplot(3, 3, 4 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("DBA $k$-means")
# Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=3,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
y_pred = sdtw_km.fit_predict(X_train)

for yi in range(3):
    plt.subplot(3, 3, 7 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()