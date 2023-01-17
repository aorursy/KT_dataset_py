import matplotlib.pyplot as plt
import numpy as np
""" This command may be helpful for Kaggle Kernel to visualize (Comment out in case of need) """

# %matplotlib inline  
x = np.arange(1, 6)

x
y = np.arange(2, 11, 2)

y
plt.plot(x, y, "red")

plt.show()
plt.subplot(2, 2, 1)  # For 2-2 matrix, works on 1st graphic
plt.plot(x, y, "blue")

plt.subplot(2, 2, 2)  # For 2-2 matrix, works on 2nd graphic
plt.plot(y, x, "yellow")

plt.subplot(2, 2, 3)  # For 2-2 matrix, works on 3rd graphic
plt.plot(x, y, "red")

plt.subplot(2, 2, 4)  # For 2-2 matrix, works on 4th graphic
plt.plot(x, x ** 2, "black")

plt.show()
fig = plt.figure() # create figure

""" Add axes to figure """

axes = fig.add_axes([
    0.1,    # margin from left
    0.2,    # margin from bottom
    0.4,    # length at X axis
    0.6     # length at Y axis
])

plt.show() # Show figure
fig = plt.figure() # Create figure

""" Set Outer Axes Properties """
outer_axes = fig.add_axes([
    0.1,    # margin from left
    0.2,    # margin from bottom
    0.8,    # length at X axis
    0.7     # length at Y axis
])

outer_axes.plot(y, x)
outer_axes.set_xlabel("Outer X")
outer_axes.set_ylabel("Outer Y")
outer_axes.set_title("Outer Graph")


""" Set Inner Axes Properties """
inner_axes = fig.add_axes([
    0.7,    # margin from left
    0.4,    # margin from bottom
    0.1,    # length at X axis
    0.2     # length at Y axis
])


inner_axes.plot(y, x, "red")
inner_axes.set_xlabel("Inner X")
inner_axes.set_ylabel("Inner Y")
inner_axes.set_title("Inner Graph")

plt.show()
flg, axes = plt.subplots(nrows = 2, ncols = 2) # Creates 4 figures at the same time

plt.show()
flg, axes = plt.subplots(nrows = 2, ncols = 2) # Creates 4 figures at the same time

plt.tight_layout() # To put spaces between sub plots

plt.show()
flg, axes = plt.subplots(nrows = 2, ncols = 1) # Creates 4 figures at the same time

print(axes) # Attention! Axes is an array with 4 elements
""" Drawing x, y Graphic for All Axes """

flg, axes = plt.subplots(nrows = 2, ncols = 1) # Creates 4 figures at the same time

for ax in axes:
    ax.plot(x, y)
    
plt.tight_layout() # To put spaces between sub plots

plt.show()
""" Drawing Axes with Different Functions """

flg, axes = plt.subplots(nrows = 2, ncols = 1) # Creates 4 figures at the same time

axes[0].plot(x, y)

axes[0].set_title("x / y")

axes[1].plot(x**3, y, "red")

axes[1].set_title("x3 / y")
    
plt.tight_layout() # To put spaces between sub plots

plt.show()
fig = plt.figure(figsize = (10, 3)) # Set figure size

axes = fig.add_axes([0, 0, 1, 1])

axes.plot(x, y, color="green")

plt.show()
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 6))

axes[0].plot(x, y, color = "red")

axes[0].set_title("First Axes")

axes[1].plot(x, x ** 0.5, color = "orange")

axes[1].set_title("Second Axes")

plt.tight_layout() # To put spaces between sub plots

plt.show()
fig.savefig("Figure1.png") # Saves figure to Figure1.png
fig.savefig("Figure1.pdf")
""" A code block to read kaggle directory  """

import os

for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
fig = plt.figure(figsize = (6, 4))

axes = fig.add_axes([0.0, 0.0, 1, 1])

axes.plot(x, x ** 0.5, color = "red", label = "square root")

axes.plot(x, x ** 2, color = "blue", label = "square")

axes.plot(x, x ** 3, color = "#c7ab00", label = "cube") # using hexadecimal color code

axes.legend() # To show mapping between lines and labels

plt.show()
fig = plt.figure()

axes = fig.add_axes([0, 0, 1, 1])

axes.plot(x, x ** 2, 
          color = "red",               # setting line color      
          linewidth = 3,               # setting line thickness
          linestyle = "-.",            # setting line style
          marker = "o",                # setting marker (circle) to unions
          markersize = 15,             # setting size of marker
          markerfacecolor = "green",   # setting inner color of marker
          markeredgecolor = "blue",    # setting edge color of marker
          markeredgewidth = 5          # setting edge thickness
         )

plt.show()
fig = plt.figure()

axes = fig.add_axes([0, 0, 1, 1])

axes.plot(x, x**2, color = "red", linewidth = 2, marker = "o", markersize = 10, markerfacecolor = "black", markeredgecolor = "blue", markeredgewidth = 3)

axes.set_xlim(0, 10)  # Makes X axis start from 0 to 10

axes.set_ylim(0, 40)  # Makes Y axis start from 0 to 40

plt.show()
