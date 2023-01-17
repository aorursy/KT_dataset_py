import numpy as np
import pandas as pd
import matplotlib.pylab as plt

data = pd.read_csv('../input/CreditCard.csv')
# checking the head
data.head()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values  # although the dependent variable is stored here, we won't be using it in the model
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)
!pip install minisom
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)
# the library used for this specific purpose is 'pylab'
from pylab import bone, pcolor, colorbar, plot, show
plt.figure(figsize = (20,10))
# Step 4.1 - initializing the figure (the window having the map)
bone()

# Step 4.2 - putting different WNs on the map 
# this is done by using different colors corresponding to the different range of values of the MIDs
# i.e we add the info of the MIDs for all the WNs that the SOM identified 
# done by the distance_map() method which returns the matrix of all the MIDs in it for all the WNs
pcolor(som.distance_map().T)  # .T means taking transpose of this matrix to get the values in order for this function
# we need to run all the lines corresponding to the visualization together hence we write the lines again
plt.figure(figsize = (20,10))
bone()
pcolor(som.distance_map().T)
colorbar()  # legend
plt.figure(figsize = (20,10))
bone()
pcolor(som.distance_map().T)
colorbar() 

# 'red circle' marker will represent customer who didn't get approval and 'green square' marker will represnt who did
markers = ['o', 's']
colors = ['red', 'green']

# creating the loop to apply the logic above
for i, j in enumerate(X):          
    wn = som.winner(j)             
    plot(wn[0] + 0.5, wn[1] + 0.5, markers[y[i]], markeredgecolor = colors[y[i]], markerfacecolor = 'None', markersize=15,
                markeredgewidth = 2)  
    
show()
mappings = som.win_map(X) 
frauds = mappings[(7,6)]
frauds = sc.inverse_transform(frauds)
df_frauds = pd.DataFrame(frauds)
df_frauds
