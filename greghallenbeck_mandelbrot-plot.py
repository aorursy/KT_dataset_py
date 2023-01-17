import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from matplotlib.pyplot import imshow

from PIL import Image

import colorsys

import matplotlib.cm as cm
# Inputs are the parameter c = a + bi and the maximum number of iterations

def mandelbrot(a, b, maxiter):

    iterations = 0 

    

    #z0 = c = a + bi

    x = a

    y = b

    

    while (iterations < maxiter):

        

        # Check exit condition.

        # If |z_i| > 2 then it will escape to infinity

        if (x**2 + y**2) > 4:

            return iterations

        

        # The next value of z_(i+1) = z_i**2 + c

        # If z_i = x + yi

        # Then z_i**2 = x**2 - y**2 + 2xyi

        # So z_(i+1) = (x**2 - y**2 + a) + (2xy + b)i

        

        # Do in two steps so that we don't overwrite the variables

        nextreal = x**2 - y**2 + a

        nextimag = 2*x*y + b

        

        x = nextreal

        y = nextimag

       

        

        iterations += 1

    

    return iterations
# Parameters of window.

# Make dx larger to run quickly (0.1 is a good choice for moving around the limits...0.0001 takes a long time to render but is pretty)

# Change cm.twilight_shifted (in the next cell) to another colormap to change coloring

minx = -1.0

maxx = -0.5

miny = 0.1

maxy =  0.6

dx     = 0.0001



maxiter = 50



xlist = np.arange(minx, maxx, dx)

ylist = np.arange(miny, maxy, dx)



xwidth  = len(xlist)

yheight = len(ylist)



mbrot = np.zeros((xwidth, yheight), dtype=int)



for ix in range(xwidth):

    for iy in range(yheight):

        mbrot[ix, iy] = mandelbrot(xlist[ix], ylist[iy], maxiter)
# Convert the mbrot array to an image.

im = Image.fromarray(np.uint8(cm.twilight_shifted(1-np.rot90(mbrot)/maxiter)*255))
plt.figure(figsize=(20,20))

imshow(im)

plt.axis('off')

plt.show()