# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits import mplot3d

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams["figure.figsize"] = 12.8, 9.6

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fig = plt.figure()

ax = plt.axes(projection='3d')
ax = plt.axes(projection='3d')



# Data for a three-dimensional line

zline = np.linspace(0, 15, 1000)

xline = np.sin(zline)

yline = np.cos(zline)

ax.plot3D(xline, yline, zline, 'gray')



# Data for three-dimensional scattered points

zdata = 15 * np.random.random(100)

xdata = np.sin(zdata) + 0.1 * np.random.randn(100)

ydata = np.cos(zdata) + 0.1 * np.random.randn(100)

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
def f(x, y):

    return np.sin(np.sqrt(x ** 2 + y ** 2))



x = np.linspace(-6, 6, 30)

y = np.linspace(-6, 6, 30)



X, Y = np.meshgrid(x, y)

Z = f(X, Y)
fig = plt.figure()

ax = plt.axes(projection='3d')

ax.contour3D(X, Y, Z, 50, cmap='binary')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z');
ax.view_init(60, 35)

fig
fig = plt.figure()

ax = plt.axes(projection='3d')

ax.plot_wireframe(X, Y, Z, color='black')

ax.set_title('wireframe');
def LoG(x, y, sigma):

    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)

    return -1 / (np.pi * sigma ** 4) * (1 - temp) * np.exp(-temp)





N = 49

half_N = N // 2

X2, Y2 = np.meshgrid(range(N), range(N))

Z2 = -LoG(X2 - half_N, Y2 - half_N, sigma=8)

X1 = np.reshape(X2, -1)

Y1 = np.reshape(Y2, -1)

Z1 = np.reshape(Z2, -1)

plt.figure(figsize=(10,10))

ax = plt.axes(projection='3d')

ax.plot_wireframe(X2, Y2, Z2, color='r')

ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,

                cmap='viridis', edgecolor='none')

ax.set_title('surface');
r = np.linspace(0, 6, 20)

theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)

r, theta = np.meshgrid(r, theta)



X = r * np.sin(theta)

Y = r * np.cos(theta)

Z = f(X, Y)



ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,

                cmap='viridis', edgecolor='none');
plt.figure(figsize=(10,10))

ax = plt.axes(projection='3d')

ax.plot_surface(X2, Y2, Z2, cmap='jet')

plt.show()
theta = 2 * np.pi * np.random.random(1000)

r = 6 * np.random.random(1000)

x = np.ravel(r * np.sin(theta))

y = np.ravel(r * np.cos(theta))

z = f(x, y)
ax = plt.axes(projection='3d')

ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
plt.figure(figsize=(10,8))

ax = plt.axes(projection='3d')

ax.scatter(X1, Y1, Z1, c=Z1, cmap='BrBG', linewidth=1)
ax = plt.axes(projection='3d')

ax.plot_trisurf(x, y, z,

                cmap='viridis', edgecolor='none');
plt.figure(figsize=(10,10))

ax = plt.axes(projection='3d')

ax.plot_trisurf(X1, Y1, Z1, cmap='twilight_shifted')
n_radii = 8

n_angles = 36



# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).

radii = np.linspace(0.125, 1.0, n_radii)

angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)



# Repeat all angles for each radius.

angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)



# Convert polar (radii, angles) coords to cartesian (x, y) coords.

# (0, 0) is manually added at this stage,  so there will be no duplicate

# points in the (x, y) plane.

x = np.append(0, (radii*np.cos(angles)).flatten())

y = np.append(0, (radii*np.sin(angles)).flatten())



# Compute z to make the pringle surface.

z = np.sin(-x*y)



fig = plt.figure()

ax = fig.gca(projection='3d')



ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)



plt.show()
from matplotlib.collections import PolyCollection

from matplotlib import colors as mcolors



fig = plt.figure(figsize=(8,8))

ax = fig.gca(projection='3d')





def cc(arg):

    return mcolors.to_rgba(arg, alpha=0.6)



xs = np.arange(0, 10, 0.4)

verts = []

zs = [0.0, 1.0, 2.0, 3.0]

for z in zs:

    ys = np.random.rand(len(xs))

    ys[0], ys[-1] = 0, 0

    verts.append(list(zip(xs, ys)))



poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),

                                         cc('y')])

poly.set_alpha(0.7)

ax.add_collection3d(poly, zs=zs, zdir='y')



ax.set_xlabel('X')

ax.set_xlim3d(0, 10)

ax.set_ylabel('Y')

ax.set_ylim3d(-1, 4)

ax.set_zlabel('Z')

ax.set_zlim3d(0, 1)



plt.show()

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, projection='3d')

for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):

    xs = np.arange(20)

    ys = np.random.rand(20)



    # You can provide either a single color or an array. To demonstrate this,

    # the first bar of each set will be colored cyan.

    cs = [c] * len(xs)

    cs[0] = 'c'

    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)



ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_zlabel('Z')



plt.show()
fig = plt.figure()

ax = fig.gca(projection='3d')



# Make the grid

x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),

                      np.arange(-0.8, 1, 0.2),

                      np.arange(-0.8, 1, 0.8))



# Make the direction data for the arrows

u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)

v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)

w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *

     np.sin(np.pi * z))



ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)



plt.show()
fig = plt.figure()

ax = fig.gca(projection='3d')



# Demo 1: zdir

zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))

xs = (1, 4, 4, 9, 4, 1)

ys = (2, 5, 8, 10, 1, 2)

zs = (10, 3, 8, 9, 1, 8)



for zdir, x, y, z in zip(zdirs, xs, ys, zs):

    label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)

    ax.text(x, y, z, label, zdir)



# Demo 2: color

ax.text(9, 0, 0, "red", color='red')



# Demo 3: text2D

# Placement 0, 0 would be the bottom left, 1, 1 would be the top right.

ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)



# Tweaking display region and labels

ax.set_xlim(0, 10)

ax.set_ylim(0, 10)

ax.set_zlim(0, 10)

ax.set_xlabel('X axis')

ax.set_ylabel('Y axis')

ax.set_zlabel('Z axis')



plt.show()

from mpl_toolkits.mplot3d.axes3d import  get_test_data

from matplotlib import cm



# set up a figure twice as wide as it is tall

fig = plt.figure(figsize=plt.figaspect(0.5))



#===============

#  First subplot

#===============

# set up the axes for the first plot

ax = fig.add_subplot(1, 2, 1, projection='3d')



# plot a 3D surface like in the example mplot3d/surface3d_demo

X = np.arange(-5, 5, 0.25)

Y = np.arange(-5, 5, 0.25)

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)

Z = np.sin(R)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,

                       linewidth=0, antialiased=False)

ax.set_zlim(-1.01, 1.01)

fig.colorbar(surf, shrink=0.5, aspect=10)



#===============

# Second subplot

#===============

# set up the axes for the second plot

ax = fig.add_subplot(1, 2, 2, projection='3d')



# plot a 3D wireframe like in the example mplot3d/wire3d_demo

X, Y, Z = get_test_data(0.05)

ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)



plt.show()
