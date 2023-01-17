# 4 More Quick and Easy Data Visualizations in Python with code - TowardsDataScience


# Heat Map

# Importing libraries

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creating a random dataset

data = pd.DataFrame(np.random.random((10,6)),columns=["Iron Man","Captain America","Black Widow","Thor","Hulk","Hawkeye"])

print(data)

# Plotting of heat map

heatmap_plot = sns.heatmap(data,center=0,cmap='gist_ncar')
plt.show()
# 2D Density Plot

from scipy.stats import skewnorm

# Creating data
speed = skewnorm.rvs(4,size=50)
size = skewnorm.rvs(4,size=50)

# Create and shor the 2D density plot
ax = sns.kdeplot(speed,size,cmap="Reds",shade=False,bw=.15,cbar=True)
ax.set(xlabel='speed',ylabel='size')
plt.show()
# Tree Diagram

from scipy.cluster import hierarchy

# Reading the dataset

df = pd.read_csv("../input/pokemon/Pokemon.csv")
df = df.set_index('Name')
del df.index.name
df = df.drop(["Type 1","Type 2","Legendary"],axis=1)
df = df.head(n=20)

# Calculating the distance between each sample

Z = hierarchy.linkage(df,'ward')

# Orientation of the tree

hierarchy.dendrogram(Z,orientation="left",labels=df.index)

plt.show()
