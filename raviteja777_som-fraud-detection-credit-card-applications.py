# Self Organizing Maps



# Importing Required Libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing dataset

dataset = pd.read_csv('/kaggle/input/Credit_Card_Applications.csv')

dataset.head(10)
!pip install MiniSom
dataset.columns
# X is all columns except last one class

# Y is Class - we use it to verify after clustering is done 

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values
# Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))

X = sc.fit_transform(X)
# Training the SOM

#using MiniSom library

from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len=15)
som.random_weights_init(X)

som.train_random(data = X, num_iteration = 100)
dataset.info()
som.distance_map()
# visualizing the results 

from pylab import bone, colorbar, pcolor, plot, show
bone()

pcolor(som.distance_map().T)

colorbar()
# use markers o for class 0 in y 

# use square (s) for class 1 in y

bone()

pcolor(som.distance_map().T)

colorbar()

markers = ['o', 's']

colors = ['r', 'g']

for i,x in enumerate(X):

    w = som.winner(x)    

    plot(w[0]+0.5,

         w[1]+0.5,

         markers[y[i]], 

         markeredgecolor=colors[y[i]], 

         markerfacecolor='None', 

         markersize=10, 

         markeredgewidth=2)



show()
# Find frauds

mappings = som.win_map(X)

mappings
# from fig squares (8,6) , (2,6) seem like potential fraudulent customers

frauds = np.concatenate((mappings[(1,7)],mappings[(8,8)]),axis=0)
frauds




frauds = sc.inverse_transform(frauds)

frauds