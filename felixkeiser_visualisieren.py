# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



!pip install ffmpeg

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from IPython.core.display import display, HTML

from matplotlib import pyplot as plt

from matplotlib import animation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def planck(x,T):

    h = 6.62607015e-34

    c = 299792458

    k = 1.380649e-23

    U = 2 * math.pi * h * c * c /((x ** 5) * (math.exp(h * c/(x * k * T)) - 1))

    return U



def curve(T, xrange, dx):

    x = np.arange(dx, xrange, dx)

    y = np.copy(x)

    for i in range(x.shape[0]):

        y[i] = planck(x[i], T)

    return x,y
xrange = 1.5e-5

dx = 1e-7



fig = plt.figure()

ax = plt.axes(xlim=(0, xrange * 1e9), ylim=(0, 1.2))

line, = ax.plot([], [], lw=2)

textvar = ax.text(0, 1.1, '$T = 250 K$' , size= 15)

ax.set_xlabel('Wellenlänge in nm')



def init():

    line.set_data([], [])

    return line, textvar





def animate(i):

    T = 250 + 20 * i

    x, y = curve(T, xrange, dx)

    x = x * 1e9

    y_max = np.amax(y)

    y = y / y_max

    line.set_data(x, y)

    textvar.set_text('$T = $' + str(T) + 'K')

    ax.set_ylabel('U/U_max (U_max = ' + str(int(y_max)) + ' J/m^3 nm)')

    return line, textvar





anim = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=300, interval=40, blit=True)



plt.close()

display(HTML(anim.to_jshtml()))
anim.save('myAnimation.gif', writer='imagemagick', fps=30)