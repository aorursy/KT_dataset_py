y_ini = 0.01 #initial condition in the range (0,1)

x_min, x_max = [3.5, 4] #maximum range [0, 4]

y_min, y_max = [0, 1]

iters = 1000 #number of iterations, higher is more accurate and more computationally expensive
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

%matplotlib inline



last = round(0.8*iters)

n_pixels = 3000

Zx = np.linspace(x_min, x_max, n_pixels)

y = np.repeat(y_ini, n_pixels)

for i in range(iters):

    y = Zx*y*(1-y)

    if i >= (iters-last): 

        if i == (iters-last): 

            D = np.array([Zx, y]).T

        else: D = np.concatenate([D,np.array([Zx, y]).T], axis = 0)            

X = D[:,0]

Y = D[:,1]



fig, ax = plt.subplots(figsize=(10, 10), 

                       dpi=300, 

                       facecolor='steelblue', 

                       edgecolor='none'

                      )

plt.axis('off')

ax.set_xlim(x_min, x_max)

ax.set_ylim(y_min, y_max)

plt.gca().set_axis_off()

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)

plt.margins(0,0)

plt.gca().xaxis.set_major_locator(plt.NullLocator())

plt.gca().yaxis.set_major_locator(plt.NullLocator())



plt.scatter(X,Y

            ,s = (3*72/n_pixels)**2

            ,alpha = 0.8

            ,c = 'darkred'

            ,marker = '.'

           )