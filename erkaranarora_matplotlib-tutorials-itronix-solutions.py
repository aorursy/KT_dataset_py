#Importing pyplot

'''

from matplotlib import pyplot as plt

or

import matplotlib.pyplot as plt 

'''



import matplotlib.pyplot as plt

a=[1,9,3,5]



#Plotting to our canvas

plt.plot(a)

#Showing what we plotted

plt.show()
import matplotlib.pyplot as plt

a=[1,2,3]

b=[4,5,1]

plt.plot(a,b)

plt.show()
import matplotlib.pyplot as plt

a=[1,9,3,5]

b=[2,45,33,6]

plt.title('Info',fontsize=25,fontname='Times New Roman',color='blue',weight='bold')

plt.ylabel('Y axis',fontsize=25,fontname='Times New Roman',color='blue',weight='bold')

plt.xlabel('X axis',fontsize=25,fontname='Times New Roman',color='blue',weight='bold')

plt.plot(a,b)

plt.show()
import matplotlib.pyplot as plt

a=[1,9,3,5]

b=[2,45,33,6]

plt.xlabel('Time')

plt.ylabel('Sensor')

plt.title("Title - Itronix Solutions")

# Limit the range of the plot to only where the data is.    

# Avoid unnecessary whitespace.    

plt.ylim(0, 90)    

plt.xlim(0, 10)  

plt.text(3,60, "Text : Itronix Solutions", fontsize=17)    

plt.plot(a,b,"#34A300",linewidth=5)

plt.show()

import matplotlib.pyplot as plt

a=[1,9,3,5]

b=[2,45,33,6]

plt.xlabel('Time')

plt.ylabel('Sensor')

plt.title("Itronix Solutions")

plt.axis([0, 20, 0, 50])

plt.plot(a,b,'k^')

plt.show()

import matplotlib.pyplot as plt

a=[1,9,3,5]

b=[2,45,33,6]

plt.xlabel('Time')

plt.ylabel('Sensor')

plt.title("Itronix Solutions")

plt.axis([0, 20, 0, 50])

#plt.plot(a,b,color='k',linewidth=5)

plt.plot(a, b, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12) 

plt.show()

import numpy as np

x = np.arange(1,15) 

print(x)

y = 3 * x + 7 # y=mx+c

print(y)

#plt.plot(x,y)

plt.plot(x,y,"ob")

plt.plot(y,x,"^y")



plt.show()
import numpy as np

x = np.arange(1,15) 

print(x)

y = 3 * x + 7 # y=mx+c

print(y)

plt.bar(x,y)

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

        width = 0.8, color = ['red', 'green']) 

  

# naming the x-axis 

plt.xlabel('x - axis') 

# naming the y-axis 

plt.ylabel('y - axis') 

# plot title 

plt.title('My bar chart!') 

  

# function to show the plot 

plt.show() 

import numpy as np

x1=[2,3,6]

y1=[6,3,4]



x2=[3,5,7]

y2=[7,3,6]

plt.bar(x1,y1,width=0.5)

plt.bar(x2,y2,width=0.5,alpha=.4) # alpha -> float (0.0 transparent through 1.0 opaque)

plt.show()
import numpy as np



N = 5

men = (20, 35, 30, 35, 27)

women = (25, 32, 34, 20, 25)



index = np.arange(N) 

width = 0.35       

plt.bar(index, men, width, label='Men')

plt.bar(index + width, women, width,label='Women')



plt.ylabel('Scores')

plt.title('Scores by group and gender')



plt.xticks(index + width / 2, ('G1', 'G2', 'G3', 'G4', 'G5'))

plt.legend(loc='best')

plt.show()
import matplotlib.pyplot as plt

import numpy as np

x=[1,2,3,4,5]

y=[10,45,33,35,68]

a=plt.bar(x,y)

for rect in a:

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()/2., 1.04*height,'%.2f' % float(height),ha='center', va='bottom')

plt.title('title')

plt.xlabel('x')

plt.ylabel('y')      

plt.show()
from matplotlib import style

print(plt.style.available)

style.use('ggplot')



x1=[2,3,6]

y1=[6,3,4]



x2=[3,5,7]

y2=[7,3,3]

plt.title("Suicide case study")

plt.xlabel("Year")

plt.ylabel("Total Suicide")

plt.bar(x1,y1,width=0.5,align='center',label='Punjab')

plt.bar(np.array(x2)+.5,y2,width=0.5,alpha=.4,align='center',label='Haryana')

plt.legend(loc='best') #by default

plt.grid(True,color='k')

#plt.savefig("Itronix.pdf")

#plt.savefig('foo.png', transparent=True)

plt.show()

import numpy as np

# evenly sampled time at 200ms intervals

t = np.arange(0., 5., 0.2)



# red dashes, blue squares and green triangles

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

plt.show()
x1=[2,3,6]

y1=[6,3,4]



x2=[3,5,7]

y2=[7,3,6]

plt.title("Suicide case study")

plt.xlabel("Year")

plt.ylabel("Total Suicide")

plt.scatter(x1,y1,label='Punjab')

plt.scatter(x2,y2,label='Haryana')

plt.legend(loc='best') #by default

plt.grid(True,color='k')

plt.show()

import requests

import json

import matplotlib.pyplot as plt

a=requests.get('https://www.quandl.com/api/v3/datasets/NSE/SBIN.json?api_key=x8VdDXzBty3iGsaRLTVs')

b=json.loads(a.text)

my_data=b['dataset']['data']

useful_data=[i[1] for i in reversed(my_data)]

plt.title('SBI Stock Prices')

plt.xlabel('Time')

plt.ylabel('Stock Price')

plt.plot(useful_data)

plt.savefig('figure.png')

plt.show()



#Below you can see that X axis are not correct
import requests

import json

import matplotlib.pyplot as plt

import pandas as pd

a=requests.get('https://www.quandl.com/api/v3/datasets/NSE/SBIN.json?api_key=x8VdDXzBty3iGsaRLTVs')

b=json.loads(a.text)

data=b['dataset']['data']

df=pd.DataFrame(data)

df[0] = pd.to_datetime(df[0])

df=df.set_index(df.columns[0])

my_data=df[1]

plt.plot(my_data)

plt.title('SBI Stock Prices')

plt.xlabel('Time')

plt.ylabel('Stock Price')

plt.savefig('SBI.pdf')

plt.show()
#!pip install quandl

import quandl

import matplotlib.pyplot as plt

dataset1=quandl.get("NSE/SBIN", authtoken="x8VdDXzBty3iGsaRLTVs")

my_data=dataset1['Open']

print(my_data.index)

plt.plot(my_data)

plt.title('SBI Stock Prices')

plt.xlabel('Time')

plt.ylabel('Stock Price')

plt.savefig('SBI.pdf')

plt.show()
import numpy as np

import matplotlib.pyplot as plt



x = [0,5,9,10,15]

y = [0,1,2,3,4]

plt.plot(x,y)

plt.xticks(np.arange(min(x), max(x)+1, 1.0))

plt.yticks(np.arange(min(y), max(y)+1, 1.0))



#plt.xticks(np.arange(0, 19, step=1.5))

#plt.xticks(np.arange(5), ('A', 'B', 'C', 'D', 'E'))

plt.show()
x = [0,1,2,3,4,5]

y = [10,20,15,18,7,19]

xlabels = ['jan','feb','mar','apr','may','jun']

plt.plot(x,y)



plt.xticks(range(0,len(x)),xlabels,rotation=45)

plt.show()
plt.title("Itronix Solutions : Pie Chart")

slices=[40,10,10,10,30]

channels=['9X','Sony','Star Plus','Colors','Cartoon Network']

cols=['red','green','b','cyan','y']

plt.pie(slices,labels=channels,colors=cols,explode=(0.1,0,0,0,0),autopct='%.2f%%')

plt.legend(loc="upper right",fontsize=12,bbox_to_anchor=(1.65,1.1))

plt.show()
from pylab import *

t = arange(0.0, 20.0, 1)

s = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

 

subplot(1,2,1)

title('subplot(2,1,1)')

plot(t,s)

 

subplot(1,2,2)

title('subplot(2,1,2)')

plot(t,s,'r-')

show()
import matplotlib.pyplot as plt

import numpy as np

t = np.arange(0.0, 20.0, 1)

s = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

 

plt.subplot(2,2,1)

plt.title('Subplot 1')

plt.plot(t,s)

 

plt.subplot(2,2,2)

plt.title('Subplot 2')

plt.plot(t,s,'r-')



plt.subplot(2,2,3)

plt.title('Subplot 3')

plt.plot(t,s)

 

plt.subplot(2,2,4)

plt.title('Subplot 4')

plt.plot(t,s,'r-')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,wspace=0.35)

plt.show()
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)



plt.subplot(grid[0, 0])

plt.subplot(grid[0, 1:])

plt.subplot(grid[1, :2])

plt.subplot(grid[1, 2])



plt.show()
import matplotlib.pyplot as plt

from matplotlib import gridspec



fig = plt.figure()

gs = gridspec.GridSpec(3, 3)



ax1 = fig.add_subplot(gs[0,:])

ax1.plot([1,2,3,4,5], [10,5,10,5,10], 'r-')



ax2 = fig.add_subplot(gs[1,:-1])

ax2.plot([1,2,3,4], [1,4,9,16], 'k-')



ax3 = fig.add_subplot(gs[1:, 2])

ax3.plot([1,2,3,4], [1,10,100,1000], 'b-')



ax4 = fig.add_subplot(gs[2,0])

ax4.plot([1,2,3,4], [0,0,1,1], 'g-')



ax5 = fig.add_subplot(gs[2,1])

ax5.plot([1,2,3,4], [1,0,0,1], 'c-')



gs.update(wspace=0.5, hspace=0.5)

import matplotlib.pyplot as plt

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

ax1.plot(range(10), 'r')

ax2.plot(range(10), 'b')

ax3.plot(range(10), 'g')

ax4.plot(range(10), 'k')

plt.show()
m = [

[1,0,2,0,0],

[1,1,1,2,0],

[0,4,1,0,0],

[0,4,4,1,2],

[1,3,0,0,1],

    

]

 

plt.matshow(m)
# importing matplotlib module  

from matplotlib import pyplot as plt 

# Y-axis values 

y = [10, 5, 8, 4, 2] 

# Function to plot histogram 

plt.hist(y) 

# Function to show the plot 

plt.show() 



# importing required modules 

from mpl_toolkits.mplot3d import axes3d 

import matplotlib.pyplot as plt 

from matplotlib import style 

import numpy as np 

  

# setting a custom style to use 

style.use('ggplot') 

  

# create a new figure for plotting 

fig = plt.figure() 

  

# create a new subplot on our figure 

ax1 = fig.add_subplot(111, projection='3d') 

  

# defining x, y, z co-ordinates for bar position 

x = [1,2,3,4,5,6,7,8,9,10] 

y = [4,3,1,6,5,3,7,5,3,7] 

z = np.zeros(10) 

  

# size of bars 

dx = np.ones(10)              # length along x-axis 

dy = np.ones(10)              # length along y-axs 

dz = [1,3,4,2,6,7,5,5,10,9]   # height of bar 

  

# setting color scheme 

color = [] 

for h in dz: 

    if h > 5: 

        color.append('r') 

    else: 

        color.append('b') 

  

# plotting the bars 

ax1.bar3d(x, y, z, dx, dy, dz, color = color) 

  

# setting axes labels 

ax1.set_xlabel('x-axis') 

ax1.set_ylabel('y-axis') 

ax1.set_zlabel('z-axis') 

  

plt.show()





'''

x, y, z, dx, dy, dz are lists. They represent the x and y , z positions

of each bar and dx, dy, dz represent the depth, 

width and height (dimensions in x, y and z) of the bars.

'''