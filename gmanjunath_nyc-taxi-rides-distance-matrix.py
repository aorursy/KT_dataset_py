# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/new-york-city-taxi-trip-distance-matrix/train_distance_matrix.csv')

data_train.tail(5)
colors = ['darkred','steelblue']

plt.scatter(data_train['trip_duration'], data_train['google_duration'],s=3, c=data_train['vendor_id'], cmap=matplotlib.colors.ListedColormap(colors) )



plt.xlabel("Trip_Duration")

plt.ylabel("Google_Duration")

plt.xlim(-100,8000)



plt.show()
fig = plt.figure()

ax = fig.gca(projection='3d')



X = np.array(data_train['pickup_latitude'])

Y = np.array(data_train['pickup_longitude'])

Z = np.array(data_train['trip_duration'])



colors = ['darkred','steelblue']



surf = ax.scatter(X,Y,Z, c= data_train['vendor_id'], cmap=matplotlib.colors.ListedColormap(colors), s=3, linewidth=0.1, antialiased=True)



ax.set_xlim(40.6,40.9)

ax.set_ylim(-73.7,-74.1)

ax.set_zlim(0, 4500)



plt.show()
