import pandas as pd



import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model, preprocessing, svm

from sklearn.preprocessing import StandardScaler, Normalizer

import math

import matplotlib

import seaborn as sns
df = pd.read_csv('../input/database.csv', sep=',', header=0)

#df = pd.read_csv('../input/autos.csv', sep=',', header=0, encoding='cp1252')
df.sample(5)
df.info()
plt.subplot(331)

plt.title("Depth Error plot")

df['Depth Error'].plot()



plt.subplot(332)

plt.title('Magnitude Error plot')

df['Magnitude Error'].plot()



plt.subplot(333)

plt.title('Azimuthal Gap plot')

df['Azimuthal Gap'].plot()



plt.subplot(334)

plt.title('Horizontal Distance plot')

df['Horizontal Distance'].plot()



plt.subplot(335)

plt.title('Horizontal Error plot')

df['Horizontal Error'].plot()



plt.subplot(336)

plt.title('Root Mean Square plot')

df['Root Mean Square'].plot()



plt.tight_layout()
#from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap 

#import matplotlib.pyplot as plt

import numpy as np

import string

import matplotlib.cm as cm



areas = [

    { 'label': 'Italy',

      'llcrnrlat': 35.57580,

      'llcrnrlon': 6.67969,

      'urcrnrlat': 47.55336,

      'urcrnrlon': 19.33594},

    { 'label': 'Greece',

      'llcrnrlat': 33.62262,

      'llcrnrlon': 18.01758,

      'urcrnrlat': 42.33317,

      'urcrnrlon': 29.17969},

    { 'label': 'Japan',

      'llcrnrlat': 29.65822,

      'llcrnrlon': 127.79297,

      'urcrnrlat': 46.41419,

      'urcrnrlon': 151.08398},

    { 'label': 'South-east Asia',

      'llcrnrlat': -11.90095,

      'llcrnrlon': 92.02148,

      'urcrnrlat': 19.02967,

      'urcrnrlon': 130.51758},

]



fig = plt.figure()

fig.set_figheight(15)

fig.set_figwidth(15)



for i, a in enumerate(areas):

    print(i, a)

    ax = fig.add_subplot(100*len(areas) + 20 + i+1)

    m = Basemap(projection='cyl',

                llcrnrlat=a['llcrnrlat'],

                llcrnrlon=a['llcrnrlon'],

                urcrnrlat=a['urcrnrlat'],

                urcrnrlon=a['urcrnrlon'],

                resolution='l')

    m.drawcountries()

    m.drawcoastlines()

    m.shadedrelief()



    m.scatter(df['Longitude'].values

              ,df['Latitude'].values

              ,s=df['Magnitude'].values*1

              ,marker="o"

              ,cmap=cm.seismic

              ,alpha=.5

              ,latlon=True)



    plt.title("Seismic events in %s area" % a['label'])

#plt.tight_layout()



plt.show()
import seaborn as sns

sns.set(style="white")



corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,

            square=True, #xticklabels=5, yticklabels=5,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)