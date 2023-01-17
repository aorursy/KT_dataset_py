#initialize
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
%matplotlib inline
# load csv data
pacman = pd.read_csv("../input/pacman_2.csv")
pacman.info()
# convert target from string to int
pacman['move'] = pacman['move-direction'].astype('category')
pacman['target'] = pd.factorize(pacman['move-direction'])[0]
pacman['target'].value_counts()
# scatter plot using pandas

ax = pacman[pacman.move=='up'].plot.scatter(x='range_x', y='range_y', 
                                                    color='red', label='up')
pacman[pacman.move=='down'].plot.scatter(x='range_x', y='range_y', 
                                                color='green', label='down', ax=ax)
pacman[pacman.move=='left'].plot.scatter(x='range_x', y='range_y', 
                                                color='blue', label='left', ax=ax)
pacman[pacman.move=='right'].plot.scatter(x='range_x', y='range_y', 
                                                color='yellow', label='right', ax=ax)
ax.set_title("scatter")