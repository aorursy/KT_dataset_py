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
import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(-5,5,100)

y = 2*x+1

plt.plot(x, y, '-r', label='y=2x+1')

plt.title('Graph of y=2x+1')

plt.xlabel('x', color='#1C2833')

plt.ylabel('y', color='#1C2833')

plt.legend(loc='upper left')

plt.grid()

plt.show()
import matplotlib.pyplot as plt

import numpy as np



delta = 0.025

xrange = np.arange(-2, 2, delta)

yrange = np.arange(-2, 2, delta)

X, Y = np.meshgrid(xrange,yrange)



# F is one side of the equation, G is the other

F = X**2

G = 1- (5*Y/4 - np.sqrt(np.abs(X)))**2

plt.contour((F - G), [0])

plt.show()
#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pyplot as plt

import numpy as np



def heart_3d(x,y,z):

   return (x**2+(9/4)*y**2+z**2-1)**3-x**2*z**3-(9/80)*y**2*z**3





def plot_implicit(fn, bbox=(-1.5, 1.5)):

    ''' create a plot of an implicit function

    fn  ...implicit function (plot where fn==0)

    bbox ..the x,y,and z limits of plotted interval'''

    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    A = np.linspace(xmin, xmax, 100) # resolution of the contour

    B = np.linspace(xmin, xmax, 40) # number of slices

    A1, A2 = np.meshgrid(A, A) # grid on which the contour is plotted



    for z in B: # plot contours in the XY plane

        X, Y = A1, A2

        Z = fn(X, Y, z)

        cset = ax.contour(X, Y, Z+z, [z], zdir='z', colors=('r',))

        # [z] defines the only level to plot

        # for this contour for this value of z



    for y in B:  # plot contours in the XZ plane

        X, Z = A1, A2

        Y = fn(X, y, Z)

        cset = ax.contour(X, Y+y, Z, [y], zdir='y', colors=('red',))



    for x in B: # plot contours in the YZ plane

        Y, Z = A1, A2

        X = fn(x, Y, Z)

        cset = ax.contour(X+x, Y, Z, [x], zdir='x',colors=('red',))



    # must set plot limits because the contour will likely extend

    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits

    # to encompass all values in the contour.

    ax.set_zlim3d(zmin, zmax)

    ax.set_xlim3d(xmin, xmax)

    ax.set_ylim3d(ymin, ymax)



    plt.show()



if __name__ == '__main__':

    plot_implicit(heart_3d)
