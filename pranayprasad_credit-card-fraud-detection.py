import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/credit-card-applications/Credit_Card_Applications.csv')
dataset.head()
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values  #We won't be using this value in our training as SOM is an Unsupervised Learning algorithm
X.shape
y.shape
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1)) 
X = sc.fit_transform(X)
! pip install minisom
from minisom import MiniSom
som = MiniSom(x=20 ,y=20 ,sigma=1.0 ,learning_rate=0.5 ,input_len=15)
som.random_weights_init(X)
som.train_random(data=X ,num_iteration=100)
from pylab import bone, pcolor, colorbar, plot, show

plt.figure(figsize=(17, 10), dpi= 80, facecolor='w', edgecolor='k') # To make the fig bigger 

pcolor(som.distance_map().T) # This line finds out the mean inter neuron distance and makes a map based on these distances.
                             # It makes clusters based on the colours based on the distances. The darker the colour the closer the neurons is to it's neighbourhood.
                             # The lighter neurons are the outliers and if customers are present in it that means they are fradulent.
colorbar() # This is the legend of the map
plt.figure(figsize=(17, 10), dpi= 80, facecolor='w', edgecolor='k') # To make the fig bigger 
pcolor(som.distance_map().T) 

colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):  # Looping through X such as x loops through the rows and i loops through columns of that customer (each attribute of a customer).
    w = som.winner(x)      # Finding out the winner node of each customer
    plot(w[0] + 0.5,       # w[0]- x coordinate , w[1] - y coordinate. We are placing the markers at the center of each node/neuron. That's the whole code inside the plot().
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
mappings = som.win_map(X)
mappings.keys()
frauds = np.concatenate( (mappings[(16,3)],mappings[(18,3)]) , axis = 0 ) 
frauds = sc.inverse_transform(frauds)
CustID = frauds[:,0]
CustID