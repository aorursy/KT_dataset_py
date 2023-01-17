from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import *

from sklearn import datasets
from sklearn.datasets import load_digits

digits = load_digits()



sampleDigits = digits.data.reshape((-1,8,8))[:10]

fig = plt.figure(figsize=(8, 3))

plt.title("Original digits data")

for i in range(10):

    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])

    ax.imshow(sampleDigits[i,:,:])

plt.show()
est = KMeans(n_clusters=10)

clusters = est.fit_predict(digits.data)

# est.cluster_centers_.shape



fig = plt.figure(figsize=(8, 3))

for i in range(10):

    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])

    ax.imshow(est.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)



#print cluster centers:

plt.show()
# Compute clustering accuracy

from scipy.stats import mode

from sklearn.metrics import accuracy_score



labels = np.zeros_like(clusters) #create array with zeros. Len=num of instances.

for i in range(10): #for each digit

    mask = (clusters == i) #filter on digit i

    labels[mask] = mode(digits.target[mask])[0] # use the mode as the target attribute. Note that we are using the target to create a classifier!

    

print("Digits clustering Accuracy: %0.3f"%(accuracy_score(digits.target, labels)))
# Plot clustering results

from sklearn.decomposition import PCA



X = PCA(2).fit_transform(digits.data)



kwargs = dict(cmap = plt.cm.get_cmap('rainbow', 10),

              edgecolor='none', alpha=0.6)

fig, ax = plt.subplots(1, 2, figsize=(15, 10))

ax[1].scatter(X[:, 0], X[:, 1], c=labels, **kwargs)

ax[1].set_title('learned cluster labels')



ax[0].scatter(X[:, 0], X[:, 1], c=digits.target, **kwargs)

ax[0].set_title('true labels')

plt.show()
from sklearn.metrics import confusion_matrix

# print(confusion_matrix(digits.target, labels))



fig = plt.figure(figsize=(12,8))

plt.imshow(confusion_matrix(digits.target, labels),

           cmap='Blues', interpolation='nearest')

plt.colorbar()

plt.grid(False)

plt.ylabel('true')

plt.xlabel('predicted')

plt.show()
from sklearn.datasets import load_sample_image

china = load_sample_image("china.jpg")

fig = plt.figure(figsize=(12,10))

plt.imshow(china)

plt.grid(False)

plt.show()
print(china.shape)

# normalize to standard vector

X = (china / 255.0).reshape(-1, 3)

print(X.shape)
def plot_pixels(data, title, colors=None, N=10000):

    if colors is None:

        colors = data

    

    # choose a random subset

    rng = np.random.RandomState(0)

    i = rng.permutation(data.shape[0])[:N]

    colors = colors[i]

    R, G, B = data[i].T

    

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].scatter(R, G, color=colors, marker='.')

    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))



    ax[1].scatter(R, B, color=colors, marker='.')

    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))



    fig.suptitle(title, size=20);

    plt.show()

plot_pixels(X, title='Input color space: 16 million possible colors')
# from sklearn.cluster import MiniBatchKMeans

numColors = 16

model = KMeans(numColors)

model.fit(X)

new_colors = model.cluster_centers_[model.predict(X)]



plot_pixels(X, colors=new_colors, title="Reduced color space: {} colors".format(numColors))
# Compress 256^3 colors 

from sklearn.cluster import MiniBatchKMeans

import seaborn as sns



# model = MiniBatchKMeans(numColors)

labels = model.fit_predict(X)

colors = model.cluster_centers_

new_image = colors[labels].reshape(china.shape)

new_image = (255 * new_image).astype(np.uint8)



# create and plot the new image

fig, ax = plt.subplots(1, 2, figsize=(25, 25))

with sns.axes_style('white'):

    ax[0].imshow(china)

    ax[0].set_title('input: 16 million colors')



    ax[1].imshow(new_image)

    ax[1].set_title('{0} colors'.format(numColors))

plt.show()