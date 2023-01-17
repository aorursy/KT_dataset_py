# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.decomposition import NMF

train = pd.read_csv('../input/mnist_train.csv')

train = train.drop("label",axis=1)   #axis=1 for dropping column

train.head()


pca = PCA(n_components=30).fit(train.values)

eigenvalues = pca.components_
row = 5

col = 6



plt.figure(figsize=(13,12))

for i in list(range(row * col)):

    plt.subplot(row, col, i + 1)

    plt.imshow(eigenvalues[i].reshape(28,28), cmap='gray')

    plt.xticks(())

    plt.yticks(())

plt.show()
model = NMF(n_components=30,random_state=0)

train_nmf_weight=model.fit_transform(train.values)

print(train_nmf_weight.shape, model.components_.shape)
row = 5

col = 6



plt.figure(figsize=(13,12))

for i in list(range(row * col)):

    plt.subplot(row, col, i + 1)

    plt.imshow(model.components_[i].reshape(28,28), cmap='gray')



plt.show()