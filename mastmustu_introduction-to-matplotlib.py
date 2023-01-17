# Import the necessary packages and modules

import matplotlib.pyplot as plt

import numpy as np



# Prepare the data

x = np.linspace(0, 10, 100)

x = np.round(x,2)

print(x)

y = 1.5*x

# Plot the data



plt.plot(x, y, label='linear plot')

plt.plot(x/2, y/2, label='linear plot 2')

# Add a legend

plt.legend()



# Show the plot

plt.show()
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot([1, 2, 3, 4], [10, 20, 25, 30], color='magenta', linewidth=3) # line chart 

ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='orange', marker='^') # plotting dots 

ax.set_xlim(0.5, 4.5)

plt.show()
# Import `pyplot`

import matplotlib.pyplot as plt



# Initialize a Figure 

fig = plt.figure()



# Add Axes to the Figure

fig.add_axes([0,0,3,2])
# Import the necessary packages and modules

import matplotlib.pyplot as plt

import numpy as np



# Create a Figure

fig = plt.figure()



# Add Axes to the Figure

fig.add_axes([0,0,1,1])



# Set up Axes

ax = fig.add_subplot(111)



# Scatter the data

ax.scatter(np.linspace(0, 1, 5), np.linspace(0, 5, 5))



# Show the plot

plt.show()
# Import the necessary packages and modules

import matplotlib.pyplot as plt

import numpy as np



# Create a Figure

fig = plt.figure()



# Add Axes to the Figure

fig.add_axes([0,0,1,1])



# Set up Axes

ax1 = fig.add_subplot(2,2,3)

ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,2,1)

ax4 = fig.add_subplot(2,2,4)

# Scatter the data

ax1.scatter(np.linspace(0, 1, 5), np.linspace(0, 5, 5))



ax2.scatter(np.linspace(0, 1, 5), np.linspace(0, 5, 5))

ax3.scatter(np.linspace(0, 1, 5), np.linspace(0, 5, 5))

ax4.scatter(np.linspace(0, 1, 5), np.linspace(0, 5, 5))



# Show the plot

plt.show()
# Import `pyplot` from `matplotlib`

import matplotlib.pyplot as plt



# Initialize the plot

fig = plt.figure(figsize=(20,15))

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



# Initialize the plot

fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132)

ax3 = fig.add_subplot(133)



# Plot the data

ax1.bar([1,2,3],[3,4,5])

ax2.barh([0.5,1,2.5],[0,1,2])

ax2.axhline(0.45) # horizontal line

ax1.axvline(0.65) # vertical line

ax3.scatter([1,2,3],[1,2,3])





# Show the plot

plt.show()
# Import the necessary packages and modules

import matplotlib.pyplot as plt

import numpy as np



# Prepare the data

x = np.linspace(0, 10, 100)



fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)





# Plot the data

ax.plot(x, x , label='sample legend' )





# Add a legend

ax.legend(bbox_to_anchor=(1.1, 1.1))

#ax.legend()

ax.set(title="A Scatter Plot", xlabel="x values", ylabel="y values")

          

# Show the plot

plt.show()

from matplotlib import pyplot as plt

import numpy as np

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

langs = ['C', 'C++', 'Java', 'Python', 'PHP']

students = [23,17,35,29,12]

ax.pie(students, labels = langs,autopct='%1.2f%%')

plt.show()
from matplotlib import pyplot as plt

import numpy as np

fig = plt.figure(figsize =(8,8))

ax = fig.add_axes([0,0,1,1])

a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])

ax.hist(a, bins = [0,25,50,75,100])

ax.set_title("histogram of result")

ax.set_xticks([0,25,50,75,100])

ax.set_xlabel('marks')

ax.set_ylabel('no. of students')

plt.show()