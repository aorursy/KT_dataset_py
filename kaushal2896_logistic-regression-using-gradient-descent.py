# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



from sklearn.datasets import make_blobs

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# generate linearly separable data

X, y = make_blobs(n_samples = 100, centers = 2, cluster_std=0.5, n_features = 1, random_state = 666)

np.random.seed(666)

# randomize the value of y as it's only generating 0 or 1

y = y*np.random.rand(100) + np.random.rand(100, )

X = X.ravel()
plt.scatter(X, y)

plt.show()