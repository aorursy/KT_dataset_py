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

import numpy as np
year=[1951,1961,1971,1981,1991,2001,2011]

population=[36.1,43.9,54.8,68.3,84.6,102.8,121.1]

plt.plot(year,population)

plt.xlabel('Year')

plt.ylabel('Population')

plt.title('Population of India in crores')

plt.legend()

plt.ioff() 

plt.show()
x = np.arange(0.0, 2, 0.01)

y1 = np.cos(2 * np.pi * x)

y2 = 1.2 * np.cos(4 * np.pi * x)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True,figsize=(10, 7))



ax1.fill_between(x, 0, y1,color='y')

ax1.set_ylabel('between y1 and 0')



ax2.fill_between(x, y1, 1,color='r')

ax2.set_ylabel('between y1 and 1')



ax3.fill_between(x, y1, y2,color='b')

ax3.set_ylabel('between y1 and y2')

ax3.set_xlabel('x')

plt.ioff()

# Defining the first data set 

x=[1,2,4]  

y=[5,7,1]  



#Defining the second data set

x1=[1,2,4]

y1=[15,17,11]



#Plotting the graphs

plt.plot(x,y,label='First Plot')

plt.plot(x1,y1,label='Second Plot')



#Defining the X,Y labels and title 

plt.xlabel('X Values')

plt.ylabel('Y Values')

plt.title('Simple X,Y Plot')

plt.legend()

plt.ioff()
# Defining the first data set 

x1=[1,3,5,7,9]

y1=[50,23,50,81,39]



# Defining the second data set 

x2=[2,4,6,8,10]

y2=[30,29,67,11,91]



#Plotting the bar charts with labels and colors 

plt.bar(x1,y1,label='Bar chart 1',color='y')

plt.bar(x2,y2,label='Bar chart 1',color='r')



#Defining the x,y labels,title and legend 

plt.xlabel('X Values')

plt.ylabel('Y Values')

plt.title('Simple X,Y Plot')

plt.legend()

plt.ioff() 
# Defining the first data set 

x1=[1,3,5,7,9]

y1=[50,23,50,81,39]



# Defining the second data set 

x2=[2,4,6,8,10]

y2=[30,29,67,11,91]



#Plotting the bar charts with labels and colors 

plt.barh(x1,y1,label='Bar chart 1',color='y')

plt.barh(x2,y2,label='Bar chart 1',color='r')



#Defining the x,y labels,title and legend 

plt.xlabel('X Values')

plt.ylabel('Y Values')

plt.title('Simple X,Y Plot')

plt.legend()

plt.ioff() 
import matplotlib.pyplot as plt



fig, ax = plt.subplots()

ax.broken_barh([(40, 30), (150, 10)], (10, 9), facecolors='blue')

ax.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9),

               facecolors=('red', 'yellow', 'green'))

ax.set_ylim(5, 35)

ax.set_xlim(0, 200)

ax.set_xlabel('seconds since start')

ax.set_yticks([15, 25])

ax.set_yticklabels(['Bill', 'Jim'])

ax.grid(True)

ax.annotate('race interrupted', (61, 25),

            xytext=(0.8, 0.9), textcoords='axes fraction',

            arrowprops=dict(facecolor='black', shrink=0.05),

            fontsize=16,

            horizontalalignment='right', verticalalignment='top')



plt.show()

# Plotting Age distribution using bar chart

population=[12,22,32,39,28,45,51,67,55,88,76,25,91,102,59,43,34,71]

per=[x for x in range(len(population))]

plt.bar(per,population)

plt.xlabel('Person')

plt.ylabel('Age')

plt.show()

plt.ioff()

bins=[0.10,20,30,40,50,60,70,80,90,100,110]

plt.hist(population,bins,histtype='bar',rwidth=0.8)

plt.xlabel('Age group')

plt.ylabel('No of people')

plt.title('Histogram of population')

plt.show()

plt.ioff()
import matplotlib.pyplot as plt



Men = [5., 30., 45., 22.]

Women = [5., 25., 50., 20.]



X = range(4)



plt.bar(X, Men, color = 'b')

plt.bar(X, Women, color = 'r',bottom=Men)

plt.show()

plt.ioff()
#Data set for scatter plot



x=[1,2,3,4,5,6,7,8,9,10]

y=[11,23,46,56,76,87,56,22,99,76]

plt.scatter(x,y,label='Scatter',color='k',marker='o',s=300) #k represents black color

plt.xlabel('X Value')

plt.ylabel('Y Value')

plt.show()

plt.ioff() 
import numpy as np

days=[1,2,3,4,5]

sleep=[7,8,6,11,7]

food=[2,3,4,3,2]

work=[7,8,7,2,2]

play=[8,5,7,8,13]

lab = ["Sleep ", "Food", "Work","Play"]

fig, ax = plt.subplots()

ax.stackplot(days, sleep, food, work,play, labels=lab,colors=['m','c','r','g'])

ax.legend(loc='upper right')

plt.xlabel('Day of Week')

plt.ylabel('Time in hours')

plt.title('How time is spend during weekdays')

plt.ioff()

plt.show()
hours=[8,2,8,2,4]

activity=['Sleep','Eat','Work','Play','Others']

col=['c','b','m','r','y']

plt.pie(hours,labels=activity,colors=col,shadow=True,startangle=0,explode=(0.2,0.2,0.2,0.2,0.2),autopct='%1.1f%%')

plt.title('How time is in a Day')

plt.ioff()
import matplotlib.pyplot as plt

# Pie chart

labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']

sizes = [15, 30, 45, 10]

#colors

colors = ['r','b','g','y']

 

fig1, ax1 = plt.subplots()

ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=45)

#draw circle

centre_circle = plt.Circle((0,0),.5,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()

plt.ioff() 
import matplotlib.pyplot as plt

w = 4

h = 3

d = 150

plt.figure(figsize=(w, h), dpi=d)

theta = [0, 0.785, 2.5,3]

r = [0, 0.1, 0.2,.4]

ax = plt.subplot(111, projection='polar')

ax.plot(theta, r)

plt.savefig("out.png")

plt.ioff() 
data = {'pineapple': 10, 'grapes': 15, 'gauva': 5, 'watermelon': 20}

names = list(data.keys())

values = list(data.values())



fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

axs[0].bar(names, values)

axs[1].scatter(names, values)

axs[2].plot(names, values)

fig.suptitle('Categorical Plotting')

plt.ioff() 
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x=[1,2,6,7,3,7,8,9,10,19]

y=[33,1,45,67,30,78,98,51,45,10]

z=[11,38,65,23,66,78,88,45,61,18]

ax.scatter(x, y, z)

ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')

plt.show()

plt.ioff() 
