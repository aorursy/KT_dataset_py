# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

auto_mpg = pd.read_csv("../input/autompg-dataset/auto-mpg.csv")

auto_mpg
import matplotlib.pyplot as plt

plt.hist(auto_mpg.mpg)

plt.title("MPG")

plt.show()
auto_mpg.origin.unique()
auto_mpg_1 = auto_mpg[auto_mpg.origin==1]

auto_mpg_1.head()
auto_mpg_2 = auto_mpg[auto_mpg.origin==2]

auto_mpg_3 = auto_mpg[auto_mpg.origin==3]

# Define how many subplots in each row and column

fig, axes = plt.subplots(nrows=3, ncols=1)



# Assign data to each subplot



ax0, ax1, ax2 = axes.flat



ax0.hist(auto_mpg_1.mpg)

ax0.set_title('Region 1')



ax1.hist(auto_mpg_2.mpg)

ax1.set_title('Region 2')



ax2.hist(auto_mpg_3.mpg)

ax2.set_title('Region 3')



plt.tight_layout()



plt.show();
auto_mpg.cylinders.unique()

import seaborn as sns

g = sns.FacetGrid(auto_mpg, col="cylinders")

g.map(plt.hist, "mpg");
g = sns.FacetGrid(auto_mpg, col="origin")

g.map(plt.hist, "mpg");
auto_mpg.mpg.plot.hist(grid=True, bins=15, rwidth=0.9,

                      color='blue')

plt.title('Miles Per Gallon')

plt.xlabel('mpg')

plt.ylabel('No. of Vehicles')
fig,ax = plt.subplots()

auto_mpg[['mpg', 'horsepower']].plot.kde(ax=ax, legend=False, title='Histogram MPG vs. HorsePower')

auto_mpg[['mpg', 'horsepower']].plot.hist(density=True, ax=ax)

ax.set_ylabel('HorsePower')

ax.grid(axis='y')
arr = np.random.random((16,16))

plt.imshow(arr, cmap='hot', interpolation='nearest')

plt.xticks([0,2,4,6,8,10,12,14])

plt.colorbar()

plt.show()



# Generate Data

x= np.random.randn(4096)

y= np.random.randn(4096)



# Create Heat Map

hmap, xedges, yedges = np.histogram2d(x, y, bins=(64, 64))

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]



# Plot the heatmap

plt.clf()

plt.ylabel('Y')

plt.xlabel('X')

plt.title('HeatMap')

plt.xticks(range(-3, 4))

plt.imshow(hmap, extent=extent)

plt.show()

# Basic Box Plot

plt.boxplot(auto_mpg.mpg)

plt.show()
# notched plot

plt.figure()

plt.boxplot(auto_mpg["mpg"], 1)

plt.show()
auto_mpg.boxplot(column='mpg', by='origin')

plt.title('')
auto_mpg.boxplot(column='mpg', by='cylinders', notch=True)

auto_mpg.boxplot(column='mpg', by='cylinders', notch=None, sym='')

auto_mpg.boxplot(column='mpg', by='cylinders', notch=None, sym='', vert=False)

auto_mpg.boxplot(column='mpg', by='cylinders', notch=None, sym='', vert=False, whis=2)

auto_mpg.head()
cylinder_counts = auto_mpg.cylinders.value_counts()

cylinder_counts
cylinders= cylinder_counts.keys().tolist()

cylinders1 = auto_mpg.cylinders.unique()

cylinders, cylinders1
cars_by_cylinders = cylinder_counts.values.tolist()

cars_by_cylinders
# Plotting Pie Chart

plt.pie(cars_by_cylinders, labels=cylinders, startangle=90, autopct='%.1f%%')

plt.title('')

plt.show()
fig, ax = plt.subplots()

ax.pie(cars_by_cylinders, labels=cylinders, autopct='%1.1f%%', startangle=90)



# Draw the circle

cc_circle = plt.Circle((0,0), 0.7, fc='yellow')

fig = plt.gcf()

fig.gca().add_artist(cc_circle)



# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()

help(auto_mpg.boxplot())
dir(plt)