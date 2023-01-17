!pip install celluloid
# I have found tha using celluloid is the most intuitive way of drawing plots in matplotlib



# imports

%matplotlib inline

import numpy as np

from IPython.display import HTML

from matplotlib import pyplot as plt

from celluloid import Camera

from random import randrange



# set a parameter to control the max lim

lim = 20



# instanciate the figure and set some x and y limits

fig = plt.figure()

ax = fig.add_subplot()



# to prevent resize of the plot

ax.set(xlim = [0,lim], ylim = [0,lim])



# no ticks and no spines

ax.set_xticks([])

ax.set_yticks([])

ax.spines["top"].set_color("None")

ax.spines["bottom"].set_color("None")

ax.spines["right"].set_color("None")

ax.spines["left"].set_color("None")



# instanciate the camera class

camera = Camera(fig)



def draw_explosion(ax):

    '''

    Generates some random points and draws them on the plot.

    '''

    

    # generate 250 points between 0 and 20 for x and y

    x = [np.random.random()*lim for i in range(250)]

    y = [np.random.random()*lim for i in range(250)]

    

    # plot those points to simulate and "explosion"

    ax.scatter(x, y, color = "orange", marker = (5, 2), alpha = .5)



# iterate at a given range

for i in range(15):

    if i < 10:

        # draw some bubbles that are getting closer

        ax.scatter(i, 10, s = (500*i) + 1, color = "red", alpha = .5)

        ax.scatter(lim - i, 10, s = (500*i) + 1, color = "blue", alpha = .5)

    elif i >= 10:

        # once the colission has happened, draw random points

        draw_explosion(ax)

        

    # okay, so it seems that what camera does, is captures all the frames

    # and then creates a video out of them

    # this is very cool, since we can draw as many things as we want

    # using simple matplotlib

    # and everything will be added to the video

    camera.snap()

    

# renders the video in jupyter notebooks

animation = camera.animate()

HTML(animation.to_html5_video())