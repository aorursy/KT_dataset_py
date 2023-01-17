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
#Object Oriented plotting follows 4 steps:

#1. Create a blank figure

#2. Add Axes

#3. Generate the plot(s)

#4. Specify params for plots
#Subplots contain more than one plot and are generated using the subplots() method using matplotlib
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from numpy.random import randn

from pandas import Series, DataFrame

from matplotlib import rcParams
%matplotlib inline

rcParams['figure.figsize']=5,4
#Defining axes

x=range(1,10)

y=[1,2,3,4,0,4,3,2,1]

fig=plt.figure()

ax=fig.add_axes([.1,.1,1,1])

ax.plot(x,y)
#How to set an axis limit

#We need to recreate the figure

#We use the set_xlim() to set the x axis limit and set_ylim() for y axis

#We can also add/hide the number on the axis using the set_xticks() method and setytocks() method. 
fig=plt.figure()

ax=fig.add_axes([0.1,0.1,1,1])

ax.set_xlim([1,9])

ax.set_ylim([0,5])

ax.set_xticks([1,2,3,4,5,6,7,8,9])

ax.set_yticks([0,1,2,3,4,5])

ax.plot(x,y)
#We can also add a grid using .grid()

fig1=plt.figure()

ax1=fig1.add_axes([.1,.1,1,1])

ax1.set_xlim([1,10])

ax1.set_ylim([0,5])

ax1.grid()

ax1.plot(x,y)
#Subplots can be used to create multiple plots in the same figure using the .sublots() function. 

fig=plt.figure()

fig, (sub1, sub2)=plt.subplots(1,2)

sub1.plot(x)

sub2.plot(x,y)