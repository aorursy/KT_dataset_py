# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the necessary packages and modules

import matplotlib.pyplot as plt

import numpy as np



# Prepare the data

x = np.linspace(0, 10, 100)



# Plot the data

plt.plot(x, x, label='linear')



# Add a legend

plt.legend()



# Show the plot

plt.show()
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)

ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')

ax.set_xlim(0.5, 4.5)

plt.show()
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)

plt.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')

plt.xlim(0.5, 4.5)

plt.show()
# Import `pyplot`

import matplotlib.pyplot as plt



# Initialize a Figure 

fig = plt.figure()



# Add Axes to the Figure

fig.add_axes([0,0,1,1])
# Import the necessary packages and modules

import matplotlib.pyplot as plt

import numpy as np



# Create a Figure

fig = plt.figure()



# Set up Axes

ax = fig.add_subplot(111)



# Scatter the data

ax.scatter(np.linspace(0, 1, 5), np.linspace(0, 5, 5))



# Show the plot

plt.show()
# Import `pyplot` from `matplotlib`

import matplotlib.pyplot as plt



# Initialize the plot

fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



# or replace the three lines of code above by the following line: 

#fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))



# Plot the data

ax1.bar([1,2,3],[3,4,5])

ax2.barh([0.5,1,2.5],[0,1,2])



# Show the plot

plt.show()
# Import `pyplot` from `matplotlib`

import matplotlib.pyplot as plt

import numpy as np



# Initialize the plot

fig = plt.figure()

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132)

ax3 = fig.add_subplot(133)



# Plot the data

ax1.bar([1,2,3],[3,4,5])

ax2.barh([0.5,1,2.5],[0,1,2])

ax2.axhline(0.45)

ax1.axvline(0.65)

ax3.scatter(np.linspace(0, 1, 5), np.linspace(0, 5, 5))



# Delete `ax3`

fig.delaxes(ax3)



# Show the plot

plt.show()
# Import the necessary packages and modules

import matplotlib.pyplot as plt

import numpy as np



# Prepare the data

x = np.linspace(0, 10, 100)



# Plot the data

plt.plot(x, x, label='linear')



# Add a legend

plt.legend()



# Show the plot

plt.show()
# Save Figure

plt.savefig("foo.png")



# Save Transparent Figure

plt.savefig("foo.png", transparent=True)


# importing matplotlib module  

from matplotlib import pyplot as plt 

  

# x-axis values 

x = [5, 2, 9, 4, 7] 

  

# Y-axis values 

y = [10, 5, 8, 4, 2] 

  

# Function to plot 

plt.plot(x,y) 

  

# function to show the plot 

plt.show() 
# importing matplotlib module  

from matplotlib import pyplot as plt 

  

# x-axis values 

x = [5, 2, 9, 4, 7] 

  

# Y-axis values 

y = [10, 5, 8, 4, 2] 

  

# Function to plot the bar 

plt.bar(x,y) 

  

# function to show the plot 

plt.show()
# importing matplotlib module  

from matplotlib import pyplot as plt 

  

# Y-axis values 

y = [10, 5, 8, 4, 2] 

  

# Function to plot histogram 

plt.hist(y) 

  

# Function to show the plot 

plt.show() 
# importing matplotlib module  

from matplotlib import pyplot as plt 

  

# x-axis values 

x = [5, 2, 9, 4, 7] 

  

# Y-axis values 

y = [10, 5, 8, 4, 2] 

  

# Function to plot scatter 

plt.scatter(x, y) 

  

# function to show the plot 

plt.show() 




# importing required modules

import matplotlib.pyplot as plt

 

# values of x and y axes

x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

y = [1, 4, 3, 2, 7, 6, 9, 8, 10, 5]

 

plt.plot(x, y)

plt.xlabel('x')

plt.ylabel('y')

 

plt.show()
# importing libraries

import random

import matplotlib.pyplot as plt

 

fig = plt.figure()

 

# function to get random values for graph

def get_graphs():

    xs =[]

    ys =[]

    for i in range(10):

        xs.append(i)

        ys.append(random.randrange(10))

    return xs, ys

 

# defining subplots

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)

 

# hiding the marker on axis

x, y = get_graphs()

ax1.plot(x, y)

ax1.tick_params(axis ='both', which ='both', length = 0)

 

# One can also change marker length

# by setting (length = any float value)

 

# hiding the ticks and markers

x, y = get_graphs()

ax2.plot(x, y)

ax2.axes.get_xaxis().set_visible(False)

ax2.axes.get_yaxis().set_visible(False)

 

# hiding the values and displaying the marker

x, y = get_graphs()

ax3.plot(x, y)

ax3.yaxis.set_major_formatter(plt.NullFormatter())

ax3.xaxis.set_major_formatter(plt.NullFormatter())

 

# tilting the ticks (usually needed when

# the ticks are densely populated)

x, y = get_graphs()

ax4.plot(x, y)

ax4.tick_params(axis ='x', rotation = 45)

ax4.tick_params(axis ='y', rotation =-45)

     

plt.show()
# importing libraries

import matplotlib.pyplot as plt

import numpy as np

 

# values of x and y axes

x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

y = [1, 4, 3, 2, 7, 6, 9, 8, 10, 5]

 

plt.plot(x, y, 'b')

plt.xlabel('x')

plt.ylabel('y')

 

# 0 is the initial value, 51 is the final value

# (last value is not taken) and 5 is the difference

# of values between two consecutive ticks

plt.xticks(np.arange(0, 51, 5))

plt.yticks(np.arange(0, 11, 1))

plt.show()