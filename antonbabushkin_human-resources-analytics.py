# Import matplotlib (plotting) and numpy (numerical arrays).

# This enables their use in the Notebook.

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np



# Import IPython's interact function which is used below to

# build the interactive widgets

from IPython.html.widgets import interact



def plot_sine(frequency=4.0, grid_points=12, plot_original=True):

    """

    Plot discrete samples of a sine wave on the interval ``[0, 1]``.

    """

    x = np.linspace(0, 1, grid_points + 2)

    y = np.sin(2 * frequency * np.pi * x)



    xf = np.linspace(0, 1, 1000)

    yf = np.sin(2 * frequency * np.pi * xf)



    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlabel('x')

    ax.set_ylabel('signal')

    ax.set_title('Aliasing in discretely sampled periodic signal')



    if plot_original:

        ax.plot(xf, yf, color='red', linestyle='solid', linewidth=2)



    ax.plot(x,  y,  marker='o', linewidth=2)



# The interact function automatically builds a user interface for exploring the

# plot_sine function.

interact(plot_sine, frequency=(1.0, 22.0, 0.5), grid_points=(10, 16, 1), plot_original=True);