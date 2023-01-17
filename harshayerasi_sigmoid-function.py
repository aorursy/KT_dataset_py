import numpy as np

import matplotlib.pyplot as plt
def sigmoid_function(x,w,b):

    return 1/(1+np.exp(-(x*w + b)))
sigmoid_function(2,0.6,1)
X = np.linspace(-10,10,100) # Dividing -10 to 10 into 100 equal parts
w = 0.9

b = 0.1
# Calculating sigmoid function for X, w, and b

Y = sigmoid_function(X, w, b)
plt.plot(X, Y)

plt.show()
# Lets invert the plot by changing the sign of "w".

w = -0.9

# Calculating sigmoid function for X, w, and b

Y = sigmoid_function(X, w, b)
plt.plot(X, Y)

plt.show()
# Shifting the plot towards positive direction is by increasing the "b" value.

b = b+3

# Calculating sigmoid function for X, w, and b

Y = sigmoid_function(X, w, b)
plt.plot(X, Y)

plt.show()
# Shifting the plot towards negative direction is by decreasing the "b" value.

b = b-6

# Calculating sigmoid function for X, w, and b

Y = sigmoid_function(X, w, b)
plt.plot(X, Y)

plt.show()
def sigmoid_function_2d(x1, x2, w1, w2, b):

    return 1/(1+np.exp(-((x1*w1 + x2*w2) + b)))
sigmoid_function_2d(2, 0, 0.6, 0, 1)
X1 = np.linspace(-10,10,100)

X2 = np.linspace(-10,10,100)



XX1, XX2 = np.meshgrid(X1, X2)
XX1
XX2
print(X1.shape, X2.shape, XX1.shape, XX2.shape)
w1 = 0.5

w2 = 0.5

b = 0

Y = sigmoid_function_2d(XX1, XX2, w1, w2, b)
from mpl_toolkits import mplot3d
fig = plt.figure()

ax = plt.axes(projection = '3d')

ax.contour3D(XX1, XX2, Y, 200)

ax.set_xlabel('XX1')

ax.set_ylabel('XX2')

ax.set_zlabel('Y')
fig = plt.figure()

ax = plt.axes(projection = '3d')

ax.contour3D(XX1, XX2, Y, 200, cmap = 'binary')

ax.set_xlabel('XX1')

ax.set_ylabel('XX2')

ax.set_zlabel('Y')
fig = plt.figure()

ax = plt.axes(projection = '3d')

ax.contour3D(XX1, XX2, Y, 200, cmap = 'viridis')

ax.set_xlabel('XX1')

ax.set_ylabel('XX2')

ax.set_zlabel('Y')
fig = plt.figure()

ax = plt.axes(projection = '3d')

ax.plot_surface(XX1, XX2, Y, cmap = 'viridis')

ax.set_xlabel('XX1')

ax.set_ylabel('XX2')

ax.set_zlabel('Y')
fig = plt.figure()

ax = plt.axes(projection = '3d')

ax.plot_surface(XX1, XX2, Y, cmap = 'viridis')

ax.set_xlabel('XX1')

ax.set_ylabel('XX2')

ax.set_zlabel('Y')

ax.view_init(30,270)

# 30 is the angle of height you look and 270 is the angle of X axis.
# Flipping the plot to XX2

fig = plt.figure()

ax = plt.axes(projection = '3d')

ax.plot_surface(XX1, XX2, Y, cmap = 'viridis')

ax.set_xlabel('XX1')

ax.set_ylabel('XX2')

ax.set_zlabel('Y')

ax.view_init(30,0)
w1 = 0.5

w2 = 2

b = 0

Y = sigmoid_function_2d(XX1, XX2, w1, w2, b)
# The weight of x2 is more now, The contribution of XX2 is more when compared to XX1.

fig = plt.figure()

ax = plt.axes(projection = '3d')

ax.plot_surface(XX1, XX2, Y, cmap = 'viridis')

ax.set_xlabel('XX1')

ax.set_ylabel('XX2')

ax.set_zlabel('Y')

ax.view_init(30,180)