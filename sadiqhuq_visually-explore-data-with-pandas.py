"""
Explore the data to gain priliminary insight
Simple statistics and visual representation
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

filename = '../input/Iris.csv'

df       = pd.read_csv(filename,sep=',')

# Attribute Information:
# 5. class: 'Species'
#    0 - Iris Setosa
#    1 - Iris Versicolour
#    2 - Iris Virginica

# statistical summary of the Iris dataset
df.describe()
# Scatter Matrix

# Map label to integer
labelcolors  = pd.factorize(df.Species)[0]
speciescolors = ['goldenrod','brown','orange']

fig =   pd.plotting.scatter_matrix(df,
        c=labelcolors, cmap=matplotlib.colors.ListedColormap(speciescolors),
        figsize=(10,10), marker='o',
        hist_kwds={'bins': 20}, s=60, alpha=.8)
# Parallel Co-ordinates
# https://en.wikipedia.org/wiki/Parallel_coordinates

fig = pd.plotting.parallel_coordinates( df.drop(['Id'], axis=1),'Species',
                                      color=speciescolors)
# Andrews Curves
# https://en.wikipedia.org/wiki/Andrews_plot
fig = pd.plotting.andrews_curves( df.drop(['Id'], axis=1),'Species',
                                color=speciescolors)