"""

Created on Fri Oct  5 15:03:37 2018



@author: alex

"""

from keras import losses, models, optimizers

from keras.models import Sequential

from keras.layers import (Dense, Dropout, Activation, Flatten) 

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.datasets import load_boston 

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import ElasticNet, Lasso, Ridge

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd
boston = load_boston()



# Plotting all of the features

  

for i in range(np.shape(boston.data)[1]):

    # Creating subpots

    fig=plt.figure()

    ax1=fig.add_subplot(1,1,1)

    ax1.scatter(boston.data[:,i], boston.target)

    # Figure name

    ax1.set_title(boston.feature_names[i])

    # Axis name

    ax1.set_xlabel(boston.feature_names[i])

    ax1.set_ylabel('House price')

    # Legend add 

    ax1.legend(loc='best')



df = pd.DataFrame(boston.data[:, 12])      # Create DataFrame using only the LSAT feature

df.columns = ['LSTAT']

df['MEDV'] = boston.target                 # Create new column with the target MEDV

df.head()
from sklearn.tree import DecisionTreeRegressor    # Import decision tree regression model



X = df[['LSTAT']].values                          # Assign matrix X

y = df['MEDV'].values                             # Assign vector y



sort_idx = X.flatten().argsort()                  # Sort X and y by ascending values of X

X = X[sort_idx]

y = y[sort_idx]



tree = DecisionTreeRegressor(criterion='mse',     # Initialize and fit regressor

                             max_depth=3)         

tree.fit(X, y)
plt.figure(figsize=(16, 8))

plt.scatter(X, y, c='steelblue',                  # Plot actual target against features

            edgecolor='white', s=70)

plt.plot(X, tree.predict(X),                      # Plot predicted target against features

         color='black', lw=2)

plt.xlabel('% lower status of the population [LSTAT]')

plt.ylabel('Price in $1000s [MEDV]')

plt.show()