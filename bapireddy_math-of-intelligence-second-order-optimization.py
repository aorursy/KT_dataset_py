# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Library used for writing mathematical expressions in python
from sympy import *
from sympy.parsing import sympy_parser as spp
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

init_printing(use_unicode=True)


# Plot range
plot_from, plot_to, plot_step = -7.0, 7.0, 0.1
# Precision for iterative methodsmp
target_precision = 0.3
m = Matrix(symbols('x1 x2'))

# Lets define the differential function 
def dfdx(x,g):
    return [float(g[i].subs(m[0],x[0]).subs(m[1],x[1])) for i in range(len(g))]

def sgd(alpha=0.001):
    """
    Steepest Descent - 1st order optimization
    """
    print("STEEPEST DESCENT: start")
    
    g = [diff(obj,i) for i in m]
    
    # Intialise x
    xs = [[0.0,0.0]]
    xs[0] = x_start
    
    # Get gradient at start location df/dx
    iter_count = 0
    while np.linalg.norm(xs[-1]-x_result) > target_precision:
        # compute the gradient 
        gs = dfdx(xs[iter_count], g)
        # calculate next point
        xs.append(xs[iter_count]-np.dot(alpha,gs))
        iter_count += 1
        if iter_count > 10000:
            break
            
    print("STEEPEST DESCENT: end with result distance: {} in {} iterations".format(np.linalg.norm(xs[-1] - x_result) , iter_count))
    xs = np.array(xs)
    plt.plot(xs[:,0],xs[:,1],'g-o')   
# Now  our newtons method for second order optimization
def nm():
    """
    Newton's method - 2nd order optimisation
    """
    
    print("NEWTON METHOD: start")
    #gradient
    g = [diff(obj,i) for i in m]
    #Hessian matrixr
    H = Matrix([[diff(g[j], m[i]) for i in range(len(m))] for j in range(len(g))])
    H_inv = H.inv()
    
    xn = [[0, 0]]
    xn[0] = x_start
    
    iter_count = 0
    while np.linalg.norm(np.array(xn[-1]).astype(float) - x_result) > target_precision:
        
        gn = Matrix(2,1,dfdx(xn[iter_count],g))
        delta_xn = -H_inv * gn
        delta_xn = delta_xn.subs(m[0], xn[iter_count][0]).subs(m[1], xn[iter_count][1])
        xn.append(Matrix(xn[iter_count]) + delta_xn)
        iter_count += 1
    print("NEWTONS METHOD: result distance: {} in {}  iterations".format(np.array(xn[-1]).astype(float) - x_result , iter_count))     
    
    xn = np.array(xn)
    plt.plot(xn[:,0], xn[:,1], 'k-o')
if __name__ == '__main__':
    ####################
    # Quadratic function
    ####################
    # Start location
    x_start = [-4.0, 6.0]

    # obj = spp.parse_expr('x1**2 - x2 * x1 - x1 + 4 * x2**2')
    # x_result = np.array([16/15, 2/15])
    obj = spp.parse_expr('x1**2 - 2 * x1 * x2 + 4 * x2**2')
    x_result = np.array([0, 0])

    # Design variables at mesh points
    i1 = np.arange(plot_from, plot_to, plot_step)
    i2 = np.arange(plot_from, plot_to, plot_step)
    x1_mesh, x2_mesh = np.meshgrid(i1, i2)
    f_str = obj.__str__().replace('x1', 'x1_mesh').replace('x2', 'x2_mesh')
    f_mesh = eval(f_str)

    # Create a contour plot
    plt.figure()

    plt.imshow(f_mesh, cmap='Paired', origin='lower',
               extent=[plot_from - 20, plot_to + 20, plot_from - 20, plot_to + 20])
    plt.colorbar()

    # Add some text to the plot
    plt.title('f(x) = ' + str(obj))
    plt.xlabel('x1')
    plt.ylabel('x2')
    nm()
    sgd(alpha=0.05)
    plt.show()

    #####################
    # Rosenbrock function
    #####################
    # Start location
    x_start = [-4.0, -5.0]

    obj = spp.parse_expr('(1 - x1)**2 + 100 * (x2 - x1**2)**2')
    x_result = np.array([1, 1])

    # Design variables at mesh points
    i1 = np.arange(plot_from, plot_to, plot_step)
    i2 = np.arange(plot_from, plot_to, plot_step)
    x1_mesh, x2_mesh = np.meshgrid(i1, i2)
    f_str = obj.__str__().replace('x1', 'x1_mesh').replace('x2', 'x2_mesh')
    f_mesh = eval(f_str)

    # Create a contour plot
    plt.figure()

    plt.imshow(f_mesh, cmap='Paired', origin='lower',
               extent=[plot_from - 20, plot_to + 20, plot_from - 20, plot_to + 20])
    plt.colorbar()

    # Add some text to the plot
    plt.title('f(x) = ' + str(obj))
    plt.xlabel('x1')
    plt.ylabel('x2')
    nm()
    sgd(alpha=0.0002)
    plt.show()
