import numpy as np

import matplotlib.pyplot as plt
fig = plt.figure(figsize = (4,3))

ax = fig.add_axes([0,0,1,1]) # dimensions [left, bottom, width, height] of the new axes.

fruits = ['Apple','Banana','Mango','Strawberry','Guava']

counts = [23,17,35,29,12]

ax.bar(fruits,counts)

plt.show()
fig,ax = plt.subplots(1,1)

a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])

ax.hist(a,bins=[0,25,50,75,100])

ax.set_title("histogram of result")

ax.set_xticks([0,25,50,75,100])

ax.set_xlabel('marks')

ax.set_ylabel('no. of students')

plt.show()
fig,a =  plt.subplots(2,2)

x = np.arange(1,5)

a[0][0].plot(x,x*x)

a[0][0].set_title('square')

a[0][1].plot(x,np.sqrt(x))

a[0][1].set_title('square root')

a[1][0].plot(x,np.exp(x))

a[1][0].set_title('exp')

a[1][1].plot(x,np.log10(x))

a[1][1].set_title('log')

plt.show()
fig=plt.figure()

ax=fig.add_axes([0,0,1,1])

ax.axis('equal')

fruits = ['Apple','Banana','Mango','Strawberry','Guava']

counts = [23,17,35,29,12]

ax.pie(counts, labels=fruits,autopct='%1.2f%%')

plt.show()
girls_grades = [89, 90, 70, 89, 100, 80, 90, 100, 80, 34]

boys_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]

grades_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

fig=plt.figure(figsize=(4, 3))

ax=fig.add_axes([0,0,1,1])

ax.scatter(grades_range, girls_grades, color='r')

ax.scatter(grades_range, boys_grades, color='b')

ax.set_xlabel('Grades Range')

ax.set_ylabel('Grades Scored')

ax.set_title('scatter plot')

plt.show()
#numpy.linspace(start, stop, num = 50, endpoint = True, retstep = False, dtype = None)

xlist = np.linspace(-3.0, 3.0, 100)

ylist = np.linspace(-3.0, 3.0, 100)

X, Y = np.meshgrid(xlist, ylist)

Z = np.sqrt(X**2 + Y**2)

fig,ax=plt.subplots(1,1)

cp = ax.contourf(X, Y, Z)

fig.colorbar(cp) # Add a color bar to a plot 

ax.set_title('Filled Contours Plot')

ax.set_xlabel('x (cm)')

ax.set_ylabel('y (cm)')

plt.show()
#%matplotlib notebook

#%matplotlib inline



# uncomment matplotlib notebook to view the 3D graph

from mpl_toolkits import mplot3d

fig = plt.figure()

ax = plt.axes(projection = '3d')

z = np.linspace(0, 1, 100)

x = z * np.sin(20 * z)

y = z * np.cos(20 * z)

ax.plot3D(x, y, z, 'gray')

ax.set_title('3D line plot')

plt.show()
#%matplotlib notebook

#%matplotlib inline



# uncomment matplotlib notebook to view the 3D graph

def f(x, y):

    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)

y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)

Z = f(X, Y)

fig = plt.figure()

ax = plt.axes(projection='3d')

ax.contour3D(X, Y, Z, 50, cmap='binary')

ax.set_xlabel('x'),ax.set_ylabel('y'), ax.set_zlabel('z')

ax.set_title('3D contour')

plt.show()
#%matplotlib notebook

#%matplotlib inline



# uncomment matplotlib notebook to view the 3D graph

x = np.outer(np.linspace(-2, 2, 30), np.ones(30))

y = x.copy().T # transpose 

z = np.cos(x ** 2 + y ** 2) 

fig = plt.figure() 

ax = plt.axes(projection='3d') 

ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none') 

ax.set_title('Surface plot') 

ax.set_xlabel('x'),ax.set_ylabel('y'),ax.set_zlabel('z')

plt.show()