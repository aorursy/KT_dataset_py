!pip install ipympl
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import itertools

import ipympl

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from mpl_toolkits.mplot3d import Axes3D



from sklearn import datasets

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris = datasets.load_iris()
print(iris.feature_names)



X = iris.data[iris.target < 2][:, [0,1]]

y = iris.target[iris.target<2]



print(X.shape, y.shape)



clf = MLPClassifier(solver='lbfgs', alpha=1e-5,

                     hidden_layer_sizes=(1,), random_state=42)



clf.fit(X, y)

print(clf.coefs_)

print(clf.n_outputs_)

print(clf.n_layers_)

# add step 5 for less memory-hungry but faster version:

# (warning: it will become edgier)

w1 = list(range(-300, 300))

w2 = list(range(-300, 300))



def loss(w1, w2):

    clf.coefs_ = [np.array([[w1,],[w2,]], ), clf.coefs_[1]]

    y_pred = clf.predict(X)

    return np.sqrt(sum((y-y_pred)**2))

    

xy = np.array(list(itertools.product(w1, w2)))

z = np.array([loss(x, y) for (x, y) in xy])

print(xy.shape, z.shape)
%matplotlib inline

#uncomment to get the interactive version (not working in kaggle, download and run locally)

#%matplotlib widget



fig = plt.figure(figsize=(20, 20));

# ax = fig.gca(projection='3d')

# plt.show()



ax = fig.add_subplot(111, projection='3d');



# load some test data for demonstration and plot a wireframe

ax.plot_trisurf(xy[:, 1], xy[:, 0], z, cmap=plt.cm.viridis, linewidth=0.2);



#you can rotate the plot by changing arguments:

ax.view_init(30, 100);



plt.draw();
