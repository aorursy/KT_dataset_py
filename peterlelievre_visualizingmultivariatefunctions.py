%matplotlib notebook

# Import required packages:

import numpy as np

import matplotlib.pyplot as plt

import math

from mpl_toolkits.mplot3d import Axes3D
# Define the bivariate function to be plotted:

def f(x,y):

    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

# (change the line immediately above if you want to change the function)
# Define the x and y data points at which to plot the function:

ds = 0.2 # point spacing used for surface plots

dp = 0.05 # point spacing used for profile and contour plots

xmin = -2 # minimum x value

xmax =  2 # maximum x value

ymin = -2 # minimum y value

ymax =  2 # maximum y value
# Calculate the x and y data points at which to plot the function:

xs = np.arange(xmin,xmax+ds,ds)

ys = np.arange(ymin,ymax+ds,ds)

xp = np.arange(xmin,xmax+dp,dp)

yp = np.arange(ymin,ymax+dp,dp)

# ( x=np.arange(A,B,D) defines the x data points as points spaced D apart from A and B )

# Define the x and y profile values to plot:

xk = np.linspace(-2,2,5) # the x profile values to plot

yk = np.linspace(-2,2,5) # the y profile values to plot

# Set how many contours to show:

nc = 8

# ( x=np.linspace(A,B,N) defines the x data points as N points equally spaced between A and B )

# "Vectorize" the function:

f2 = np.vectorize(f)

# Calculate the function values on a 2D grid:

Xs,Ys = np.meshgrid(xs,ys)

Xp,Yp = np.meshgrid(xp,yp)

Zs = f2(Xs,Ys)

Zp = f2(Xp,Yp)
# Plot a 3D surface in a single colour:

fig = plt.figure()

ax = Axes3D(fig)

ax.plot_surface(Xs,Ys,Zs, rstride=1, cstride=1, color='green')

# Change view angle (defaults are 30,-60):

ax.view_init(elev=30, azim=-60)

# Add labels on the axes and color them differently:

ax.set_xlabel('x',color='black')

ax.set_ylabel('y',color='blue')

ax.set_zlabel('f(x,y)',color='green')

ax.w_xaxis.line.set_color('black')

ax.w_yaxis.line.set_color('blue')

ax.w_zaxis.line.set_color('green')

ax.tick_params(axis='x', colors='black')

ax.tick_params(axis='y', colors='blue')

ax.tick_params(axis='z', colors='green')

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')
# Plot a 3D surface using a colour map:

fig = plt.figure()

ax = Axes3D(fig)

s = ax.plot_surface(Xs,Ys,Zs, rstride=1, cstride=1, cmap='viridis')

# Change view angle (defaults are 30,-60):

ax.view_init(elev=30, azim=-60)

# Add labels on the axes and color them differently:

ax.set_xlabel('x',color='black')

ax.set_ylabel('y',color='blue')

ax.set_zlabel('f(x,y)',color='green')

ax.w_xaxis.line.set_color('black')

ax.w_yaxis.line.set_color('blue')

ax.w_zaxis.line.set_color('green')

ax.tick_params(axis='x', colors='black')

ax.tick_params(axis='y', colors='blue')

ax.tick_params(axis='z', colors='green')

# Add a color bar (for spacing only):

fig.colorbar(s,shrink=0.9)

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')
# Plot the function as a map:

fig = plt.figure()

plt.imshow(Zp, interpolation='nearest', cmap='viridis', origin='lower', extent=[xmin,xmax,ymin,ymax])

# Add labels on the axes:

plt.xlabel('x')

plt.ylabel('y')

# Add a color bar:

plt.colorbar()

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')
# Plot a 3D transparent coloured surface and profiles:

fig = plt.figure()

ax = Axes3D(fig)

s = ax.plot_surface(Xs,Ys,Zs, rstride=1, cstride=1, cmap='viridis', alpha=0.5)

# Plot profiles on the surface:

for y in yk:

   yt = np.tile(y,[xp.size,1])

   ax.plot(xp,yt,f2(xp,y), lw=2, color='black')

# Change view angle (defaults are 30,-60):

ax.view_init(elev=30, azim=-60)

# Add labels on the axes and color them differently:

ax.set_xlabel('x',color='black')

ax.set_ylabel('y',color='blue')

ax.set_zlabel('f(x,y)',color='green')

ax.w_xaxis.line.set_color('black')

ax.w_yaxis.line.set_color('blue')

ax.w_zaxis.line.set_color('green')

ax.tick_params(axis='x', colors='black')

ax.tick_params(axis='y', colors='blue')

ax.tick_params(axis='z', colors='green')

# Add a color bar:

fig.colorbar(s,shrink=0.9)

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')
# Plot coloured profiles at specific x or y values:

fig = plt.figure()

#cs = plt.contour(Yp,Zp,Xp, levels=xk, cmap='viridis') # use this to plot profiles at specific x values

cs = plt.contour(Xp,Zp,Yp, levels=yk, cmap='viridis') # use this to plot profiles at specific y values

plt.clabel(cs, inline=1, fontsize=10)

# Add labels on the axes:

plt.xlabel('x')

plt.ylabel('f(x,y)')

# Add a color bar:

plt.colorbar()

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')
# Plot black profiles at specific x or y values:

fig = plt.figure()

#cs = plt.contour(Yp,Zp,Xp, levels=xk, colors='black') # use this to plot profiles at specific x values

cs = plt.contour(Xp,Zp,Yp, levels=yk, colors='black') # use this to plot profiles at specific y values

plt.clabel(cs, inline=1, fontsize=10)

# Add labels on the axes:

plt.xlabel('x')

plt.ylabel('f(x,y)')

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')
# Plot a 3D transparent coloured surface and contours:

fig = plt.figure()

ax = Axes3D(fig)

s = ax.plot_surface(Xs,Ys,Zs, rstride=1, cstride=1, cmap='viridis', alpha=0.5)

# Plot black contours on the surface:

ax.contour3D(Xp,Yp,Zp, nc, linewidths=2, colors='black') # black

#ax.contour3D(Xp,Yp,Zp, nc, linewidths=2, cmap='viridis') # colored

# Change view angle (defaults are 30,-60):

ax.view_init(elev=30, azim=-60)

# Add labels on the axes and color them differently:

ax.set_xlabel('x',color='black')

ax.set_ylabel('y',color='blue')

ax.set_zlabel('f(x,y)',color='green')

ax.w_xaxis.line.set_color('black')

ax.w_yaxis.line.set_color('blue')

ax.w_zaxis.line.set_color('green')

ax.tick_params(axis='x', colors='black')

ax.tick_params(axis='y', colors='blue')

ax.tick_params(axis='z', colors='green')

# Add a color bar:

fig.colorbar(s,shrink=0.9)

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')
# Plot black contours:

fig = plt.figure()

cb = plt.contour(Xp,Yp,Zp, nc, colors='black')

plt.clabel(cb, inline=1, fontsize=10)

# Add labels on the axes:

plt.xlabel('x')

plt.ylabel('y')

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')
# Plot coloured contours:

fig = plt.figure()

cb = plt.contour(Xp,Yp,Zp, nc, cmap='viridis')

plt.clabel(cb, inline=1, fontsize=10)

# Add labels on the axes:

plt.xlabel('x')

plt.ylabel('y')

# Add a color bar:

plt.colorbar()

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')
# Plot black contours and fill with colour:

fig = plt.figure()

plt.contourf(Xp,Yp,Zp, nc, alpha=.75, cmap='viridis')

# Add a color bar:

plt.colorbar()

# Plot black contours:

cb = plt.contour(Xp,Yp,Zp, nc, colors='black')

plt.clabel(cb, inline=1, fontsize=10)

# Add labels on the axes:

plt.xlabel('x')

plt.ylabel('y')

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')
# Plot 3D transparent coloured surface and contours on x-y plane:

fig = plt.figure()

ax = Axes3D(fig)

# Plot a 3D coloured and transparent surface:

s = ax.plot_surface(Xs,Ys,Zs, rstride=1, cstride=1, cmap='viridis', alpha=0.5)

# Plot contours beneath:

z1 = -1 # z value at which to plot the contours

z2 = 1 # maximum z value to plot

ax.contour(Xp,Yp,Zp, nc, offset=z1, cmap='viridis')

ax.set_zlim(z1,z2)

# Change view angle (defaults are 30,-60):

ax.view_init(elev=30, azim=-60)

# Add labels on the axes and color them differently:

ax.set_xlabel('x',color='black')

ax.set_ylabel('y',color='blue')

ax.set_zlabel('f(x,y)',color='green')

ax.w_xaxis.line.set_color('black')

ax.w_yaxis.line.set_color('blue')

ax.w_zaxis.line.set_color('green')

ax.tick_params(axis='x', colors='black')

ax.tick_params(axis='y', colors='blue')

ax.tick_params(axis='z', colors='green')

# Add a color bar:

fig.colorbar(s,shrink=0.9)

# Set figure background color white:

fig = plt.gcf()

fig.patch.set_facecolor('white')