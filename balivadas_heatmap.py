# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

from pandas import DataFrame

import numpy as np

import matplotlib.pyplot as plt

import scipy.interpolate as interpolate

from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

import numpy as np

#import Tkinter, tkFileDialog, tkMessageBox

from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import griddata

df = pd.read_csv('../input/soil-map2/123.csv',encoding="ISO-8859-1", error_bad_lines = False)

df= DataFrame(df, columns=['Lat','Long','Volumetric Water Content'])

df = df.dropna()



Y = df['Lat']

X = df['Long']

X = df['Long']*-1

Z = df['Volumetric Water Content']





X = np.array(X)

Y = np.array(Y)

Z = np.array(Z)





yi = np.linspace(41.8571498, 41.86144356, 550)

xi = np.linspace(-88.22001742, -88.22921114, 550)

xi, yi = np.meshgrid(xi, yi)

zi = interpolate.griddata( (X, Y), Z, (xi, yi),method='cubic')

Z_range = (0.2, 0.45)



plt.figure()

plt.subplots_adjust(left=0,right=1,bottom=0,top=1)

CS = plt.contourf(xi,yi,zi,50, vmin=0.2, vmax=0.45, cmap='viridis')   #jet for normal, jet_r for reverse

plt.colorbar()



Y_range = plt.gca().get_ylim()                         # x min value and x max value

X_range = plt.gca().get_xlim()

# data points 

plt.gca().set_ylim(Y_range[0], Y_range[1])

plt.gca().set_xlim(X_range[0], X_range[1])

plt.axis('off')  

plt.savefig( "contour_plot.png" , transparent=True)  #"Documents/Rajmundry_Project/RIVER_DATA/" + 
import folium

from matplotlib.pyplot import imread

from folium import plugins

min_lon = X.min()

max_lon =X.max()

min_lat = Y.min()

max_lat = Y.max()



# create the map

map_ = folium.Map(location=[Y.mean(), X.mean()],

                  tiles='openstreetmap', zoom_start = 13)



# read in png file to numpy array

data = imread('../input/contour-map/contour_plot (2).png')



# Overlay the image

map_.add_children(folium.raster_layers.ImageOverlay(data, opacity=0.8, \

        bounds =[[min_lat, min_lon], [max_lat, max_lon]]))

map_