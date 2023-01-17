import numpy as np

x = np.arange(0,100)

y = x*2

z = x**2



import matplotlib.pyplot as plt 

%matplotlib inline



#Creating a figure object called fig using plt.figure()

#Using add_axes to add an axis to the figure canvas at [0,0,1,1].naming this new axis ax.

#Plotting (x,y) on ax and setting the labels as x and y and titles:



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x,y,'b')

ax.set_xlabel('x axis')

ax.set_ylabel('y axis')

ax.set_title('Heading')



#Creating a figure object and putting two axes on it, ax1 and ax2.

#Located at [0,0,1,1] and [0.2,0.5,.2,.2] respectively.

#Now plotting (x,y) on both axes:



fig = plt.figure()

ax1 = fig.add_axes([0,0,1,1])

ax2 = fig.add_axes([0.2,0.5,.2,.2])

ax1.plot(x,y,'r')

ax2.plot(x,y,'r')



#Creating the plot by adding two axes to a figure object at [0,0,1,1] and [0.2,0.5,.4,.4]

# using xlimits and y limits on the inserted plot:



fig = plt.figure()

ax1 = fig.add_axes([0,0,1,1])

ax2 = fig.add_axes([0.2,0.5,.4,.4])

ax1.plot(x,z,'b')

ax2.plot(x,y,'b')

ax2.set_xlim([20,22])

ax2.set_ylim([30,50])



#Using plt.subplots(nrows=1, ncols=2)for creating plot with 1 row and 2 colums.

#Now plotting (x,y) and (x,z) on the axes and setting the linewidth and style:



fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].plot(x,y,color="blue", lw=3, ls='--')

axes[1].plot(x,z,color="red", lw=3, ls='-')



#resizing the plot by adding the figsize() argument in plt.subplots().



fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,2))

axes[0].plot(x,y,color="blue", lw=5,ls='--')

axes[0].set_xlabel('x')

axes[0].set_ylabel('y')

axes[1].plot(x,z,color="red", lw=3, ls='--')

axes[1].set_xlabel('x')

axes[1].set_ylabel('z')