import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['figure.figsize'] = (10, 5)   # Change this if figures look ugly. from matplotlib import rcParams



# IPython libraries



from ipywidgets import interactive

from IPython.display import display

training_points = 250    #  Number of training points

noise = 0.1   # Noise level



def generate_linear_data(training_points,noise):

    # generate random data-set

    np.random.seed(0)

    x = np.random.rand(training_points, 1)

    m = 3   # Slope

    c = 1   # Intercept

    y = c + m * x +  np.random.rand(training_points,1) * noise    # y = mx + c + noise

    # plot

    plt.scatter(x,y,s=25, marker = "o")

    plt.xlabel('x')

    plt.ylabel('y')

    plt.title("Generated data")

    plt.show()

    return (x,y)





# This will call the interactive widget with the data generating function, which also plots the data real-time

l=interactive(generate_linear_data,training_points={'50 samples':50,'200 samples':200},noise =(0,1,0.2))

display(l)

x_min = -5

x_max = 5

noise = 0.1



def generate_poly_data(training_points,x_min,x_max,noise):

    x1 = np.linspace(x_min,x_max,training_points*5)

    x = np.random.choice(x1,size=training_points)

    y = np.sin(x) + noise*np.random.normal(size=training_points)

    plt.scatter(x,y,edgecolors='k',c='red',s=60)

    plt.grid(True)

    plt.show()

    return (x,y)



# This will call the interactive widget with the data generating function, which also plots the data real-time

p=interactive(generate_poly_data,training_points={'50 samples':50,'200 samples':200},noise =(0,1,0.2),x_min=(-5,0,1), x_max=(0,5,1))

display(p)