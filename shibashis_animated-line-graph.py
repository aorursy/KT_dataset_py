# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as animation



x = np.linspace(0, 10, 100)

y = np.sin(x)



fig, ax = plt.subplots()

line, = ax.plot(x, y, color='k')



def update(num, x, y, line):

    line.set_data(x[:num], y[:num])

    line.axes.axis([0, 10, 0, 1])

    return line,



ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line],

                              interval=25, blit=True)

ani.save('test.gif')

plt.show()
#download the generated animation

from IPython.display import FileLink

FileLink(r'test.gif')