import numpy as np # linear algebra
def function(x, y):

    return(-2*x + x**2 - 12 * y + y**3)
def get_gradient(x, y):

    dx = -2 + 2*x

    dy = -12 + 3*y**2

    return(np.array([dx, dy]))
def new_point(x, y, grad):

    a = np.array([x,y])

    b = a - step * grad

    return(b[0], b[1])
# Starting Point 

x_0, y_0 = 4, 4

# Stepsize

step = 0.10
# Initial Step

function(x_0, y_0)

grad_0 = get_gradient(x_0, y_0)

x_1, y_1 = new_point(x_0, y_0, grad_0)
# Second Step

function(x_1, y_1)

grad_1 = get_gradient(x_1, y_1)

x_2, y_2 = new_point(x_1, y_1, grad_1)
x, y = 4, 4



for i in range(1, 10):

     grad = get_gradient(x, y)

     x, y = new_point(x, y, grad)

     print('\n' + "x: ", x) 

     print("y: ", y)

     print("z: ", function(x, y))