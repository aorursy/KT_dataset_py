import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
melting = pd.read_csv('/kaggle/input/tempurature-data-elements/Elements melting point .csv', index_col='Atomic number')

boiling = pd.read_csv('/kaggle/input/tempurature-data-elements/Elements boiling points.csv', index_col='Atomic number')

boiling.drop(['Name chemical element', 'Symbol'], axis=1, inplace=True)

mass = pd.read_csv('/kaggle/input/tempurature-data-elements/Atomic mass.csv', index_col='Atomic number')

mass.drop(['Name chemical element', 'Symbol'], axis=1, inplace=True)
melting = melting.merge(boiling, left_index=True, right_index=True)

periodic = melting.merge(mass, left_index=True, right_index=True)

periodic.columns

periodic = periodic[[ 'Name chemical element', 'Symbol','Melting- point (C)', 

       'Boiling- point', 'Atomic Mass']]
#removes an error in the dataset, will have it updated out later

periodic = periodic.loc[periodic['Boiling- point'] != 760]
# Below we have the Correlation coeficient of Atomic Mass compaired to the other two features

periodic.corr()['Atomic Mass']
import seaborn as sns

import matplotlib.pyplot as plt
fig, ax =plt.subplots(1,2, figsize=(20, 10))

sns.scatterplot(x=periodic['Boiling- point'], y=periodic.index, ax=ax[0])

sns.scatterplot(y=periodic['Boiling- point'], x=periodic['Atomic Mass'], ax=ax[1]);
print('Boiling point by Melting point, color indicating atomic mass')

fig, ax =plt.subplots(1,2, figsize=(20, 10))

sns.scatterplot(periodic['Boiling- point'], periodic['Melting- point (C)'],hue=periodic.index, ax=ax[0])

sns.scatterplot(periodic['Boiling- point'], periodic['Melting- point (C)'],hue=periodic['Atomic Mass'], ax=ax[1]);
# %matplotlib notebook



import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d



import seaborn as sns, numpy as np, pandas as pd, random

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# sns.set_style("whitegrid", {'axes.grid' : False})



fig = plt.figure(figsize=(10,10))



ax = Axes3D(fig)







x = periodic['Boiling- point']

y = periodic['Melting- point (C)']

z = hue=periodic['Atomic Mass']





g = ax.scatter(x, y, z, c=x, marker='o', depthshade=False, cmap='Paired')

ax.set_xlabel('Boiling point')

ax.set_ylabel('Melting point')

ax.set_zlabel('Atomic Mass')



# produce a legend with the unique colors from the scatter

legend = ax.legend(*g.legend_elements(), loc="lower center", title="Boiling Point", borderaxespad=-10, ncol=4)

ax.add_artist(legend)



# plt.show()



from matplotlib import animation



def rotate(angle):

     ax.view_init(azim=angle)



angle = 1

ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=1)

ani.save('rotating_chem_plot.gif', writer=animation.PillowWriter(fps=25))
import os

from IPython.display import Image

Image(filename="./rotating_chem_plot.gif")