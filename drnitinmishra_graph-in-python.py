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
# importing the required module 
import matplotlib.pyplot as plt 

# x axis values 
x = [1,2,3] 
# corresponding y axis values 
y = [2,4,1] 

# plotting the points 
plt.plot(x, y) 

# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 

# giving a title to my graph 
plt.title('My first graph!') 

# function to show the plot 
plt.show() 

import matplotlib.pyplot as plt 

# line 1 points 
x1 = [1,2,3] 
y1 = [2,4,1] 
# plotting the line 1 points 
plt.plot(x1, y1, label = "line 1") 

# line 2 points 
x2 = [1,2,3] 
y2 = [4,1,3] 
# plotting the line 2 points 
plt.plot(x2, y2, label = "line 2") 

# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
# giving a title to my graph 
plt.title('Two lines on same graph!') 

# show a legend on the plot 
plt.legend() 

# function to show the plot 
plt.show() 

import matplotlib.pyplot as plt 

# x axis values 
x = [1,2,3,4,5,6] 
# corresponding y axis values 
y = [2,4,1,5,2,6] 

# plotting the points 
plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, 
		marker='o', markerfacecolor='blue', markersize=12) 

# setting x and y axis range 
plt.ylim(1,8) 
plt.xlim(1,8) 

# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 

# giving a title to my graph 
plt.title('Some cool customizations!') 

# function to show the plot 
plt.show() 

import matplotlib.pyplot as plt 

# x-coordinates of left sides of bars 
left = [1, 2, 3, 4, 5] 

# heights of bars 
height = [10, 24, 36, 40, 5] 

# labels for bars 
tick_label = ['one', 'two', 'three', 'four', 'five'] 

# plotting a bar chart 
plt.bar(left, height, tick_label = tick_label, 
		width = 0.8, color = ['red', 'green','blue','black']) 

# naming the x-axis 
plt.xlabel('x - axis') 
# naming the y-axis 
plt.ylabel('y - axis') 
# plot title 
plt.title('My bar chart!') 

# function to show the plot 
plt.show() 
