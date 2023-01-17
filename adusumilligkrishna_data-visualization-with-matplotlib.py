#Importing the matplotlib package



import matplotlib.pyplot as plt



import numpy as np
x = np.linspace(0, 10, 100)

fig = plt.figure()



plt.plot(x, np.sin(x))

plt.plot(x, np.cos(x))
#variation in the curve lines

plt.plot(x, np.sin(x), '_')

plt.plot(x, np.cos(x), '--')

#Plotting title

plt.title("X-Y sine Curve")

plt.xlabel("X curve")

plt.ylabel("Y Curve")

plt.plot(x, np.sin(x), '_')

plt.plot(x, np.cos(x), '--')
#Saving a figure

fig.savefig("figure.jpg")



#Check the different file formats

fig.canvas.get_supported_filetypes()
#Grid plot



plt.style.use('seaborn-whitegrid')

fig = plt.figure()

ax = plt.axes()
#let us plot the graph on the sine

x = np.linspace(0,10,1000)

ax.plot(x,np.sin(x))

plt.plot(x,np.sin(x))

plt.title("X-Y Curve")

plt.xlabel("X Curve")

plt.ylabel("Y Curve")
#changing the color

plt.plot(x, np.sin(x), color = 'red')
plt.style.use('seaborn-white')

data = np.random.randn(100)



#Histogram



plt.hist(data)
##random labels & Sizes



labels = 'Apple', 'banana', 'cherry', 'Guava'

sizes = [15, 30, 45,10]



#Highlight particular part of the graph

#Second label "banana" will be highlighted

explode = (0, 0.1, 0, 0)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode= explode, labels=labels, autopct='%1.1f%%', startangle = 90)

ax1.axis('equal')

plt.show()
x = [10,20,30,40,50]

y = [45,55,65,80,90]



plt.bar(x,y)

plt.title("Bar graph")

plt.xlabel("X-axis")



plt.ylabel("Y-axis")

plt.show()
#vertical

x = ['aj', 'Tom', 'Vicky', 'anusha','mike']

y = [40000, 50000, 60000, 70000,70000]



plt.bar(x,y, color = 'red')

plt.title('Employee salary')

plt.xlabel('Employee')

plt.ylabel('salary')
#Horizontal



x = ['aj', 'Tom', 'Vicky', 'anusha','mike']

y = [40000, 50000, 60000, 70000,70000]

#for horizontal graph chanege to "barh"

plt.barh(x,y, color = 'red')

plt.title('Employee salary')

plt.xlabel('Employee')

plt.ylabel('salary')

#line graph

import pandas as pd



year = [2015,2016,2017,2018,2019]

Salary_hike = [2000,3000,4000,3500,8000]



# plot line chart

plt.plot(year, Salary_hike)



plt.title("Salary hike curve")

plt.xlabel("year")

plt.ylabel('Salary_hike')

plt.style.use('seaborn-white')
A = [15, 30, 45, 22,45,54]

B = [15,25,50,20,60,70]



z2 = range(6)

plt.bar(z2, A, color = 'b')

plt.bar(z2, B, color = 'r',bottom = A)

plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = Axes3D(fig)

X = np.arange(-4, 4, 0.25)

Y = np.arange(-4, 4, 0.25)

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)

Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
r = np.arange(0, 2, 0.01)

theta = 2 * np.pi * r



ax = plt.subplot(111, projection='polar')

ax.plot(theta, r)

ax.set_rmax(2)

ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks

ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line

ax.grid(True)



ax.set_title("A line plot on a polar axis", va='bottom')

plt.show()