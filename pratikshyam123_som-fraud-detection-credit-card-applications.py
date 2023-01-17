##Install MiniSom Package
!pip install MiniSom
# Self Organizing Maps

# Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing dataset
dataset = pd.read_csv('/kaggle/input/Credit_Card_Applications.csv')
dataset.head(10)
dataset.dtypes
# X is all columns except last one class
# Y is Class - we use it to verify after clustering (SOM) is done 
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
som.distance_map().shape
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
markers = ['o', 's'] #red circles -> customers who didn't get approval
colors = ['r', 'g'] # green square -> customers who got  approval
for i, x in enumerate(X):  # loop over customer database , for each customer vector
    w = som.winner(x)  # getting the winning node for the particular customer
    plot(w[0] + 0.5, # x coordinate of winning node = w[0]
         w[1] + 0.5, # y coordinate of the winning node = w[1], adding 0.5 to put marker in middle of square
         markers[y[i]],# association between customer approval and markers
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
# Finding the frauds
mappings = som.win_map(X) # dictionary of mappings from winning node coordinates to customers
mappings.keys() #keys are the coordinates in the plot
# from fig squares (7,5) seem like potential fraudulent customers - white color - maximum mean interneuron distance
frauds = np.concatenate((mappings[(6,4)],mappings[(6,8)]),axis=0)
#frauds = mappings[(7,5)]
np.asarray(frauds).shape
frauds = sc.inverse_transform(frauds)  #inverse feature scaled to the original values
np.asarray(frauds).shape
print('Fraud Customer IDs')
for i in frauds[:, 0]:
    print(int(i))