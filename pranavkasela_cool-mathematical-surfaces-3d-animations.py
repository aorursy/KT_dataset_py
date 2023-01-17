import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



from matplotlib import cm

from matplotlib.ticker import LinearLocator

import matplotlib.animation as animation



from IPython.display import HTML
def f(x,y):

    "the parabolic surface function"

    return x**2+y**2



X = np.arange(-5, 5, 0.1)

Y = np.arange(-5, 5, 0.1)

# Create a mesh grid to get a 2d arrays from the two 1d arrays

X, Y = np.meshgrid(X, Y)



# Z from the X,Y meshgrid (which is also a 2d array)

Z = f(X,Y)
fig = plt.figure()

# we say that the plot is 3d with the projection option

# and we set the plot angle in view_init

ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=20, azim=30)



#plot the surface with plot_surface

ax.plot_surface(X,Y,Z)



#the points can be plotted with scatter.

ax.scatter(0,0,f(0,0),s=50,c="red",depthshade=False)



#set the axis off (if you don't want the axis obviously :D)

ax.set_axis_off()



plt.title("f(x,y) = x^2+y^2")

plt.show()
def f(x,y):

    "the parabolic surface function"

    return x**2+y**2



X = np.arange(-5, 5, 0.25)

Y = np.arange(-5, 5, 0.25)

# Create a mesh grid to get a 2d arrays from the two 1d arrays

X, Y = np.meshgrid(X, Y)



# Z from the X,Y meshgrid (which is also a 2d array)

Z = f(X,Y)
# simple gradient descent

FRAMES = 60

points = [(5,5)]

LR = 0.05



def gradient(x,y):

    return (2*x,2*y) 



for i in range(FRAMES):

    grad = gradient(points[i][0],points[i][1])

    new_p = (points[i][0] - LR*grad[0], 

             points[i][1] - LR*grad[1])

    points.append(new_p)
fig = plt.figure()

ax = Axes3D(fig)



ax.plot_wireframe(X, Y, Z,color='blue',

                  linewidth=1)



t1 = ax.text(-2,0,f(5,5), f"current = {f(5,5)}°", fontsize=16)



# Do a normal plot in which we will update the point with set_data_3d

scatter, = ax.plot([5],[5],[f(5,5)], "o", markersize=12, color="red")



def init():

    "First view of the plot"

    

    ax.view_init(elev=40, azim=-90)

    

    ax.set_xlim([-5,5])

    ax.set_ylim([-5,5])

    return fig,



def animate(frame_n):

    """

    Change the plot point value on each iteration!

    """

    x = points[frame_n][0]

    y = points[frame_n][1]

    t1.set_text(f"value = {round(f(x,y),2)}")

    

    scatter.set_data([x],[y])

    scatter.set_3d_properties([f(x,y)])

    return fig,



# Animate

ani = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=FRAMES, interval=150, blit=True)



plt.close()



# show the animation as a javascript animation (so you can control it interactively)

HTML(ani.to_jshtml())
from math import pi

from numpy import sin, cos, sqrt



v = np.arange(-pi/2, pi/2, 0.001)

u = np.arange(0, pi, 0.001)



u, v = np.meshgrid(u, v)





X = 2 / 3. * (cos(u) * cos(2 * v)

        + sqrt(2) * sin(u) * cos(v)) * cos(u) / (sqrt(2) -

                                                 sin(2 * u) * sin(3 * v))

Y = 2 / 3. * (cos(u) * sin(2 * v) -

        sqrt(2) * sin(u) * sin(v)) * cos(u) / (sqrt(2)

        - sin(2 * u) * sin(3 * v))

Z = -sqrt(2) * cos(u) * cos(u) / (sqrt(2) - sin(2 * u) * sin(3 * v))
fig = plt.figure()

ax = Axes3D(fig)



def init():

    "First view of the plot"

    ax.plot_surface(X, Y, Z, rstride=10, cstride=10,

                    linewidth=0, antialiased=False)

    ax.set_axis_off()

    ax.view_init(elev=0, azim=0)

    ax.set_xlim([-1.5,1.5])

    ax.set_ylim([-1.5,1.5])

    ax.set_zlim([-2,0])

    return fig,



def animate(frame_n):

    """

    A loop which returns on the initial point!

    """

    if frame_n < 30:

        elev = frame_n*6 

        azim = frame_n*12

    else:

        elev = 30*6 - (frame_n - 30)*6

        azim = 30*12 - (frame_n - 30)*12

    ax.view_init(elev=elev, azim=azim)

    print(f"done frame number: {frame_n}", end="\r")

    return fig,



# Animate

ani = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=60, interval=120, blit=True)



plt.close()



# show the animation as a javascript animation (so you can control it interactively)

HTML(ani.to_jshtml())
# Taken From stack :D

def plot_implicit(fn, bbox=(-2.5,2.5)):

    ''' create a plot of an implicit function

    fn  ...implicit function (plot where fn==0)

    bbox ..the x,y,and z limits of plotted interval'''

    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3

    fig = plt.figure()

    ax = Axes3D(fig)

    A = np.linspace(xmin, xmax, 150) # resolution of the contour

    B = np.linspace(xmin, xmax, 75) # number of slices

    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted



    for z in B: # plot contours in the XY plane

        X,Y = A1,A2

        Z = fn(X,Y,z)

#         col = [plt.cm.Greens(z)]

        cset = ax.contour(X, Y, Z+z, [z], zdir='z', alpha=0.7)

        # [z] defines the only level to plot for this contour for this value of z



    for y in B: # plot contours in the XZ plane

        X,Z = A1,A2

        Y = fn(X,y,Z)

#         col = [plt.cm.Blues(z)]

        cset = ax.contour(X, Y+y, Z, [y], zdir='y', alpha=0.7)



    for x in B: # plot contours in the YZ plane

        Y,Z = A1,A2

        X = fn(x,Y,Z)

#         col = [plt.cm.Reds(z)]

        cset = ax.contour(X+x, Y, Z, [x], zdir='x', alpha=0.7)



    # must set plot limits because the contour will likely extend

    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits

    # to encompass all values in the contour.

    ax.set_zlim(zmin,zmax)

    ax.set_xlim(xmin,xmax)

    ax.set_ylim(ymin,ymax)

    return fig,ax
def fun(x,y,z):

    return  x**4+2*x**2*y**2+2*x**2*z**2+y**4+2*y**2*z**2+z**4+8*x*y*z-10*x**2-10*y**2-10*z**2+20



fig, ax = plot_implicit(fun, bbox=(-3.5,3.5))

def init():

    "First view of the plot"

    ax.set_axis_off()

    ax.view_init(elev=0, azim=0)

    return fig,



def animate(frame_n):

    """

    A loop which returns on the initial point!

    """

    if frame_n < 30:

        elev = frame_n*6 

        azim = frame_n*12

    else:

        elev = 30*6 - (frame_n - 30)*6

        azim = 30*12 - (frame_n - 30)*12

    ax.view_init(elev=elev, azim=azim)

    return fig,



# Animate

ani = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=60, interval=120, blit=True)



plt.close()



# show the animation as a javascript animation (so you can control it interactively)

HTML(ani.to_jshtml())