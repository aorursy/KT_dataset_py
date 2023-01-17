# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # to plot the data for the visualization of SOM



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
! pip install MiniSom
"""## Importing the dataset"""



dataset = pd.read_csv('../input/credit-card-applications/Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values 

X
y = dataset.iloc[:, -1].values

y
"""## Feature Scaling"""



from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

X = sc.fit_transform(X)
"""##Training the SOM"""



from minisom import MiniSom

som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)

som.random_weights_init(X)

# And this above method is random weights underscore in it.So that's the method that will initialize the weights.

# And inside this method, we just need to input the data.That is x, our data on which the model will be trained.

som.train_random(data = X, num_iteration = 100)
"""##Visualizing the results"""



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



"""## Finding the frauds"""



mappings = som.win_map(X)

frauds = np.concatenate((mappings[(5,3)], mappings[(1,8)]), axis = 0)

# this above line corresponds to the white boxes in our map which shows the value between them i.r MID to be

# 1 which is definately are the customers with the fraud of credit card.

frauds = sc.inverse_transform(frauds)

# the inverse transfoem is applied because we have scaled our dataset and when we see the frauds here you will

# get the scaled customer id which is actually is not true one so inverse transform is applied.
print(frauds)
"""##Printing the Fraunch Clients"""



print('Fraud Customer IDs')

for i in frauds[:, 0]:

  print(int(i))