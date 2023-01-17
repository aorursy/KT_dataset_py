import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl
train_df = pd.read_csv('../input/conways-reverse-game-of-life-2020/train.csv')

test_df = pd.read_csv("../input/conways-reverse-game-of-life-2020/test.csv")

print(train_df.shape)

print(test_df.shape)
train_df.head()
colors = ['white','black'] 

bounds = [0,1] 

cmap = mpl.colors.ListedColormap(colors)

norm = mpl.colors.BoundaryNorm(bounds,cmap.N)

for i in range(3):

    # convert to matrix

    start = np.asarray(train_df.iloc[i][2:2+625])

    start_mat = start.reshape(25,25)



    # extract stop values

    stop = np.asarray(train_df.iloc[i][2+625:])

    stop_mat = stop.reshape(25,25)

    

    plt.rcParams['figure.figsize'] = (15.0, 7.0) #  Set the figsize

    plt.figure()

    plt.subplot(1,2,1)

    plt.xlabel('start_map')

    plt.imshow(start_mat,cmap = cmap)

    plt.colorbar()



    plt.subplot(1,2,2)

    plt.xlabel('stop_map')

    plt.imshow(stop_mat,cmap = cmap)

    plt.colorbar()

    plt.show()

colors = ['black','white','yellow','blue'] 

bounds = [-1,0,1,2] 

cmap = mpl.colors.ListedColormap(colors)

norm = mpl.colors.BoundaryNorm(bounds,cmap.N)

for i in range(3):

    # convert to matrix

    start = np.asarray(train_df.iloc[i][2:2+625])

    start_mat = start.reshape(25,25)

    # extract stop values

    stop = np.asarray(train_df.iloc[i][2+625:])

    stop_mat = stop.reshape(25,25)

    

    plt.rcParams['figure.figsize'] = (15.0, 7.0) #  Set the figsize

    plt.figure()

    plt.subplot(1,2,1)

    plt.xlabel('add')

    plt.imshow(start_mat+stop_mat,cmap = cmap,norm = norm)

    plt.colorbar()



    plt.subplot(1,2,2)

    plt.xlabel('minus')

    plt.imshow(start_mat-stop_mat,cmap = cmap,norm = norm)

    plt.colorbar()

    plt.show()

    