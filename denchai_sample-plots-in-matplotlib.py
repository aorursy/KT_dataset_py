import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib

import matplotlib.pyplot as plt

#Data for plotting

t = np.arange(0.0, 2.0, 0.01)

s = np.sin(2 * np.pi * t)



fig, ax = plt.subplots()

ax.plot(t, s)



ax.set(xlabel='time (s)', ylabel='voltage (mV)',

       title='Simple sinwave')

ax.grid()

#This saves to /kaggle/working/test.png

fig.savefig("test.png")

plt.show()