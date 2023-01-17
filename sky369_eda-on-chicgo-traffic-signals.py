# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd

cam_loc = pd.read_csv('../input/red-light-camera-locations.csv', delimiter=',')
cam_vio = pd.read_csv('../input/red-light-camera-violations.csv', delimiter=',')
speed_loc = pd.read_csv('../input/speed-camera-locations.csv', delimiter=',')
speed_vio = pd.read_csv('../input/speed-camera-violations.csv', delimiter=',')

#cam_loc.head(5)
#cam_vio.head(5)
#speed_loc.head(5)
#speed_vio.head(5)

# let us see how many redlight cameras were there in Chicago city
print ("length of red-light-camera-locations.csv  is  " + str(len(cam_loc)))

# and of course redlight violations
print ("length of red-light-camera-violations.csv  is  " + str(len(cam_vio)))

# we are intresetd number of speed camera locations as well
print ("length of speed-camera-locations.csv  is  " + str(len(speed_loc)))

# let us check speed camera violations.
print ("length of speed-camera-violations.csv  is  " + str(len(speed_vio)))

## It seems like there are more red light camera vilations than speed camera violations.

## we need to find where people are violating red signals.to do that, let us take a look at the red-light-camera-violations.csv

cam_vio.head(5)

# from VIOLATIONS column find top 10  Intersections where there are more violations.



##cam_loc.head(5)

## find what type of violations are more
##print (str(pd.sum(cam_vio.VIOLATIONS)))

#cam_vio['VIOLATIONS'].sum()
#speed_vio['VIOLATIONS'].sum()

x = cam_vio['VIOLATIONS'].sum()
y = speed_vio['VIOLATIONS'].sum()

plt.plot(x,y)





