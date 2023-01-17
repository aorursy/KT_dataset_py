import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Install MiniSom Package

!pip install MiniSom
## Importing the dataset

dataset = pd.read_csv('/kaggle/input/credit-card-applications/Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values 

y = dataset.iloc[:, -1].values
## Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

X = sc.fit_transform(X)
## Training the SOM

from minisom import MiniSom

som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)

som.random_weights_init(X)

som.train_random(data = X, num_iteration = 100)
## Visualizing the results

from pylab import bone, pcolor, colorbar, plot, show

bone()

pcolor(som.distance_map().T)

colorbar()

markers = ['o', 's']

colors = ['r', 'g']

for i, x in enumerate(X):

    w = som.winner(x)

    plot(w[0] + 0.5,

         w[1] + 0.5,

         markers[y[i]],

         markeredgecolor = colors[y[i]],

         markerfacecolor = 'None',

         markersize = 10,

         markeredgewidth = 2)

show()