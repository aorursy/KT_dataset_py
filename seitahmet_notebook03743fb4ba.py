import os
import numpy as np

import pandas as pd
from pandas.plotting import andrews_curves
from pandas.plotting import parallel_coordinates

from sklearn.cluster import AgglomerativeClustering

import seaborn as sns

#matplotlib and related imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import Patch
import matplotlib.patches as patches

def print_files():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
#load dataset
df = pd.read_csv('/kaggle/input/video-game-length-times/games.csv') 
#first 5 rows of the dataset
df.head()
#last 5 rows of the dataset
df.tail()
#dataset info
df.info()
#description of dataset
df.describe()
#checking null values
df.isnull().sum()
#shape of dataset
df.shape
#ratings higher than 70
df_ratings = df[df.rating > 70].rating
print(df_ratings)
#making a correlation matrix
corr = df.corr()
corr
# create a mask to pass it to seaborn and only show half of the cells 
# because corr between x and y is the same as the y and x
# it's only for estetic reasons
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig = plt.figure(figsize = (10, 5))

# plot the data using seaborn
ax = sns.heatmap(corr, 
                 mask = mask, 
                 vmax = 0.3, 
                 square = True,  
                 cmap = "viridis")
# set the title for the figure
ax.set_title("Heatmap using seaborn");
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot()

# plot using matplotlib
ax.imshow(df.corr(), cmap = 'viridis', interpolation = 'nearest')
# set the title for the figure
ax.set_title("Heatmap using matplotlib");
