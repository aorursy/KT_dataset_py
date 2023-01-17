%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)



import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import pylab as plot

import numpy as np
x = np.array([1,2,3,4,5,6,7,8])

y = x



plt.figure()

plt.scatter(x, y) # similar to plt.plot(x, y, '.'), but the underlying child objects in the axes are not Line2D
x = np.array([1,2,3,4,5,6,7,8])

y = x



# create a list of colors for each point to have

# ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'red']

colors = ['green']*(len(x)-1)

colors.append('red')



plt.figure()



# plot the point with size 100 and chosen colors

plt.scatter(x, y, s=100, c=colors)
plt.figure()

# plot a data series 'Tall students' in red using the first two elements of x and y

plt.scatter(x[:2], y[:2], s=100, c='red', label='Tall students')

# plot a second data series 'Short students' in blue using the last three elements of x and y 

plt.scatter(x[2:], y[2:], s=100, c='blue', label='Short students')
linear_data = np.array([1,2,3,4,5,6,7,8])

exponential_data = linear_data**2



plt.figure()

# plot the linear data and the exponential data

plt.plot(linear_data, '-o', exponential_data, '-o')
# plot another series with a dashed red line

plt.plot([22,44,55], '--r')
plt.figure()

xvals = range(len(linear_data))

plt.bar(xvals, linear_data, width = 0.3)
new_xvals = []



# plot another set of bars, adjusting the new xvals to make up for the first set of bars plotted

for item in xvals:

    new_xvals.append(item+0.3)



plt.bar(new_xvals, exponential_data, width = 0.3 ,color='red')
plt.figure()

xvals = range(len(linear_data))

plt.bar(xvals, linear_data, width = 0.3, color='b')

plt.bar(xvals, exponential_data, width = 0.3, bottom=linear_data, color='r')

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
plt.figure()

Y = np.random.normal(loc=0.0, scale=1.0, size=10000)

X = np.random.random(size=10000)

plt.scatter(X,Y)
plt.figure()



Y = np.random.normal(loc=0.0, scale=1.0, size=10000)

X = np.random.random(size=10000)

_ = plt.hist2d(X, Y, bins=25)
data.describe()