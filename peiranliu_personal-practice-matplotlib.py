# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np

x = np.arange(0,100)

y = x*2

z = x**2
#Create a figure object and put two axes on it, ax1 and ax2. Located at [0,0,1,1] and [0.2,0.5,.2,.2] respectively
fig=plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x,y)

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_title('x,y')
#Now plot (x,y) on both axes. And call your figure object to show it
fig=plt.figure()

ax1 = fig.add_axes([0,0,1,1])

ax2 = fig.add_axes([0.2,0.5,0.4,0.4])

ax1.plot(x,y,color='red')

ax2.plot(y,x)
#use x,y, and z arrays to recreate the plot below. Notice the xlimits and y limits on the inserted plot
fig=plt.figure()

ax=fig.add_axes([0,0,1,1])

ax2=fig.add_axes([0.2,0.5,0.4,0.4])

ax.plot(x,z)

ax.set_xlabel("x")

ax.set_ylabel("z")



ax2.plot(x,y)

ax2.set_title('zoom')

ax2.set_xlabel("x")

ax2.set_ylabel("y")

ax2.set_xlim(20,22)

ax2.set_ylim(30,50)
#Use plt.subplots(nrows=1, ncols=2) to create the plot below
fig,axes=plt.subplots(1,2)

axes[0].plot(x,y)

axes[1].plot(x,z,color='blue',lw=3,ls='--')
fig,axes=plt.subplots(1,2,figsize=(12,2))

axes[0].plot(x,y)

axes[1].plot(x,z,color='blue',lw=3,ls='--')