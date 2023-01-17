import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

from matplotlib import pyplot as plt

plt.style.use('bmh')

font = {'family' : 'sans-serif',

        'weight' : 'normal',

        'size'   : 14}

mpl.rc('font', **font)
# Define a differentiable function and its derivative

f = lambda x: x**4

df = lambda x: 4 * x**3



# Let us plot the f(x)

fig, axes = plt.subplots(figsize=(3,4))

x = np.linspace(-4,4,num=100,endpoint=True) # define a range for x

plt.plot(x,f(x), linewidth=4,zorder=1, color='royalblue')

plt.xlabel('x')

plt.ylabel('f(x)')
xi = 2 # Initial guess

diff_xi_xi_plus1 = xi



gamma = 0.01 # Step size multiplier

delta_x = 0.02 #Precision



fig, axes = plt.subplots(figsize=(6,8))

n = 20

colors = mpl.cm.autumn(np.linspace(0,1,n))



# Let us plot the f(x)

x = np.linspace(-5,5,num=100,endpoint=True) # define a range for x

plt.plot(x,f(x), linewidth=4,zorder=1, color='royalblue') # plot f(x)



#---------------------------------------------------------

# Start the iteration and keep plotting during the process

#---------------------------------------------------------

i = 1

while diff_xi_xi_plus1 > delta_x:

    # plot the gradient point

    plt.scatter(xi, f(xi), color='k', s=100, zorder=2)

    plt.scatter(xi, f(xi), color='white', s=10, zorder=3)

    

    # plot the gradient line; (x,y) for y=m(x-x0)+y0

    df_line_xrange = np.linspace(xi-3,xi+3,5)

    plt.plot(df_line_xrange, (df(xi) * (df_line_xrange - xi)) + f(xi), color=colors[i], zorder=1)



    if i == 1: xi_plus1 = xi

    xi_plus1 += -gamma * df(xi)

    diff_xi_xi_plus1 = abs(xi - xi_plus1)

    print("%d- xi: %f, xi+1: %f, diff_xi_xi_plus1: %f" % (i, xi,xi_plus1,diff_xi_xi_plus1))

    xi = xi_plus1

    i += 1



plt.ylim([0,20])

plt.xlim([-3,3])

plt.xlabel('x')

plt.ylabel('f(x)')

plt.title('Finding the Minimum of a Differentiable Function \n by Gradient Descent')



print("Local Minimum: %f with threshold of: %f" % (xi, delta_x))