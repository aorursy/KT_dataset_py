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
# Read MNIST data

mnist_train_csv = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

mnist_test_csv = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")



# Load MNIST data and labels

mnist_train_data = mnist_train_csv.iloc[:,1:]

mnist_test_data = mnist_test_csv.iloc[:,1:]

mnist_train_label = mnist_train_csv['label']

mnist_test_label = mnist_test_csv['label']

mnist_all_data = pd.concat([mnist_train_data, mnist_test_data])

mnist_all_label = pd.concat([mnist_train_label, mnist_test_label])

mnist_inverted_train_data = mnist_train_data.copy()

mnist_inverted_test_data = mnist_test_data.copy()

mnist_inverted_all_data = 255 - mnist_all_data.copy()

mnist_inverted_train_data = 255 - mnist_inverted_train_data

mnist_inverted_test_data = 255 - mnist_inverted_test_data

mnist_train_data /= 255.

mnist_test_data /= 255.

mnist_all_data /= 255.

mnist_inverted_train_data /= 255.

mnist_inverted_test_data /= 255.

mnist_inverted_all_data /= 255.



mnist_inverted_train_label = mnist_train_label.copy()

mnist_inverted_test_label = mnist_test_label.copy()

mnist_inverted_all_label = mnist_all_label.copy()



print('number of MNIST train data: %d' % len(mnist_train_data))

print('number of MNIST test data: %d' % len(mnist_test_data))

print('number of all MNIST data: %d' % len(mnist_all_data))
from sklearn import datasets

digits = datasets.load_digits()

# Take the first 500 data points: it's hard to see 1500 points

X = digits.data[:500]

y = digits.target[:500]



from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)



X_2d = tsne.fit_transform(X)



target_ids = range(len(digits.target_names))



from matplotlib import pyplot as plt

plt.figure(figsize=(6, 5))

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'

for i, c, label in zip(target_ids, colors, digits.target_names):

    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)

plt.legend()

plt.show()
from sklearn import datasets

digits = datasets.load_digits()

# Take the first 500 data points: it's hard to see 1500 points

X = mnist_all_data[:1000]

y = mnist_all_label[:1000]



from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)



X_2d = tsne.fit_transform(X)



target_ids = range(len(digits.target_names))



from matplotlib import pyplot as plt

plt.figure(figsize=(6, 5))

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'

labels = [i for i in range(10)]

for i, c, label in zip(y, colors, labels):

    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)

plt.legend()

plt.show()
# Take the first 500 data points.

X = mnist_inverted_all_data[:1000]

y = mnist_inverted_all_label[:1000]



from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)



X_2d = tsne.fit_transform(X)



target_ids = range(len(digits.target_names))



from matplotlib import pyplot as plt

plt.figure(figsize=(6, 5))

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'

for i, c, label in zip(target_ids, colors, digits.target_names):

    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)

plt.legend()

plt.show()
# Perform the necessary imports

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE



model = TSNE(learning_rate=100)

transformed = model.fit_transform(mnist_test_data[:1000])



xs = transformed[:,0]

ys = transformed[:,1]

plt.scatter(xs,ys,c=mnist_test_label[:1000])

plt.legend([i for i in range(10)])



plt.show()
# Perform the necessary imports

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE



model = TSNE(learning_rate=100)

transformed = model.fit_transform(mnist_inverted_test_data[:1000])



xs = transformed[:,0]

ys = transformed[:,1]

plt.scatter(xs,ys,c=mnist_inverted_test_label[:1000])

plt.legend([i for i in range(10)])



plt.show()
# Calculate the mean and standard deviation

mnist_train_mean, mnist_train_std = mnist_train_data.to_numpy().mean(), mnist_train_data.to_numpy().flatten().std()

mnist_test_mean, mnist_test_std = mnist_test_data.to_numpy().flatten().mean(), mnist_test_data.to_numpy().flatten().std()

mnist_all_mean, mnist_all_std = mnist_all_data.to_numpy().flatten().mean(), mnist_all_data.to_numpy().flatten().std()



mnist_inverted_train_mean, mnist_inverted_train_std = mnist_inverted_train_data.to_numpy().flatten().mean(), mnist_inverted_train_data.to_numpy().flatten().std()

mnist_inverted_test_mean, mnist_inverted_test_std = mnist_inverted_test_data.to_numpy().flatten().mean(), mnist_inverted_test_data.to_numpy().flatten().std()

mnist_inverted_all_mean, mnist_inverted_all_std = mnist_inverted_all_data.to_numpy().flatten().mean(), mnist_inverted_all_data.to_numpy().flatten().std()
print('MNIST train mean: {}, std: {}'.format(mnist_train_mean, mnist_train_std))

print('MNIST test mean: {}, std: {}'.format(mnist_test_mean, mnist_test_std))

print('MNIST all mean: {}, std: {}'.format(mnist_all_mean, mnist_all_std))



print('inverted MNIST train mean: {}, std: {}'.format(mnist_inverted_train_mean, mnist_inverted_train_std))

print('inverted MNIST test mean: {}, std: {}'.format(mnist_inverted_test_mean, mnist_inverted_test_std))

print('inverted MNIST all mean: {}, std: {}'.format(mnist_inverted_all_mean, mnist_inverted_all_std))
# Visualize the MNIST data

import matplotlib.pyplot as plt

tmp_idx = np.random.randint(0,60000)

plt.imshow(mnist_train_data.iloc[tmp_idx,:].to_numpy().reshape(28,28),cmap=plt.cm.binary)

plt.title(mnist_train_label[tmp_idx])

plt.show()

plt.imshow(mnist_inverted_train_data.iloc[tmp_idx,:].to_numpy().reshape(28,28),cmap=plt.cm.binary)

plt.title(mnist_inverted_train_label[tmp_idx])

plt.show()