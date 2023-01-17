# default library for data handling

import numpy as np 

import pandas as pd 



# matplotlib series

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib import animation



# HTML 

from IPython.display import HTML
print(animation.writers.list())
TWOPI = 2*np.pi



fig, ax = plt.subplots(1, 1)



t = np.arange(0.0, TWOPI, 0.001)

s = np.sin(t)

l = plt.plot(t, s)



ax = plt.axis([0,TWOPI,-1,1])



redDot, = plt.plot([0], [np.sin(0)], 'ro')



def animate(i):

    redDot.set_data(i, np.sin(i))

    return redDot,



myAnimation = animation.FuncAnimation(fig, animate, 

                                      frames=np.arange(0.0, TWOPI, 0.1), 

                                      interval=10, blit=True, 

                                      repeat=False)



plt.show()
%time myAnimation.save('myAnimation1.gif', writer='imagemagick', fps=30)



%time myAnimation.save('myAnimation2.gif', writer='pillow', fps=30)
HTML('<img src="./myAnimation1.gif" />')