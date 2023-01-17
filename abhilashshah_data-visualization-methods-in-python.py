# A Gentle Introduction to Data Visualization Methods in Python


import matplotlib.pyplot as plt


# Line Plot

from numpy import sin
# Consistent interval for x-axis
X = [X*0.1 for X in range(100)]
# Function of x for y-axis
y = sin(X)
# Create line plot 
plt.plot(X,y)
# show line plot
plt.show()
# Bar Chart

from random import seed
from random import randint

# seed the random number generator
seed(1)
# Names for categories
X = ['red','green','blue']
#quantities for each category
y = [randint(0,100),randint(0,100),randint(0,100)]
#create bar chart
plt.bar(X,y)
plt.show()
# Histogram plot 

from numpy.random import seed
from numpy.random import randn

seed(1)

x = randn(1000)
plt.hist(x)
plt.show()
# Box and Whisker Plot

seed(1)
x = [randn(1000),5*randn(1000),10*randn(1000)]

plt.boxplot(x)
plt.show()
# Scatter Plot

seed(1)

x = 20*randn(1000)+100
y = x + (10*randn(1000)+50)

plt.scatter(x,y)
plt.show()