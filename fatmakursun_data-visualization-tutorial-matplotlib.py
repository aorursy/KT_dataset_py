import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

x = np.linspace(0, 5, 11)

y = x ** 2
x
y
plt.plot(x, y, 'r') # 'r' is the color red

plt.xlabel('X Axis Title Here')

plt.ylabel('Y Axis Title Here')

plt.title('String Title Here')

plt.show()
# plt.subplot(nrows, ncols, plot_number)

plt.subplot(1,2,1)

plt.plot(x, y, 'r--') # More on color options later

plt.subplot(1,2,2)

plt.plot(y, x, 'g*-');
# Create Figure (empty canvas)

fig = plt.figure()



# Add set of axes to figure

axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)



# Plot on that set of axes

axes.plot(x, y, 'b')

axes.set_xlabel('Set X Label') # Notice the use of set_ to begin methods

axes.set_ylabel('Set y Label')

axes.set_title('Set Title')
# Creates blank canvas

fig = plt.figure()



axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes



# Larger Figure Axes 1

axes1.plot(x, y, 'b')

axes1.set_xlabel('X_label_axes2')

axes1.set_ylabel('Y_label_axes2')

axes1.set_title('Axes 2 Title')



# Insert Figure Axes 2

axes2.plot(y, x, 'r')

axes2.set_xlabel('X_label_axes2')

axes2.set_ylabel('Y_label_axes2')

axes2.set_title('Axes 2 Title');
# Use similar to plt.figure() except use tuple unpacking to grab fig and axes

fig, axes = plt.subplots()



# Now use the axes object to add stuff to plot

axes.plot(x, y, 'r')

axes.set_xlabel('x')

axes.set_ylabel('y')

axes.set_title('title');
# Empty canvas of 1 by 2 subplots

fig, axes = plt.subplots(nrows=1, ncols=2)
# Axes is an array of axes to plot on

axes
for ax in axes:

    ax.plot(x, y, 'b')

    ax.set_xlabel('x')

    ax.set_ylabel('y')

    ax.set_title('title')



# Display the figure object    

fig
fig, axes = plt.subplots(nrows=1, ncols=2)



for ax in axes:

    ax.plot(x, y, 'g')

    ax.set_xlabel('x')

    ax.set_ylabel('y')

    ax.set_title('title')



fig    

plt.tight_layout()
fig = plt.figure(figsize=(8,4), dpi=100)
fig, axes = plt.subplots(figsize=(12,3))



axes.plot(x, y, 'r')

axes.set_xlabel('x')

axes.set_ylabel('y')

axes.set_title('title');
fig.savefig("filename.png")
fig.savefig("filename.png", dpi=200)
ax.set_title("title");
ax.set_xlabel("x")

ax.set_ylabel("y");
fig = plt.figure()



ax = fig.add_axes([0,0,1,1])



ax.plot(x, x**2, label="x**2")

ax.plot(x, x**3, label="x**3")

ax.legend()
# Lots of options....



ax.legend(loc=1) # upper right corner

ax.legend(loc=2) # upper left corner

ax.legend(loc=3) # lower left corner

ax.legend(loc=4) # lower right corner



# .. many more options are available



# Most common to choose

ax.legend(loc=0) # let matplotlib decide the optimal location

fig
# MATLAB style line color and style 

fig, ax = plt.subplots()

ax.plot(x, x**2, 'b.-') # blue line with dots

ax.plot(x, x**3, 'g--') # green dashed line
fig, ax = plt.subplots()



ax.plot(x, x+1, color="blue", alpha=0.5) # half-transparant

ax.plot(x, x+2, color="#8B008B")        # RGB hex code

ax.plot(x, x+3, color="#FF8C00")        # RGB hex code 
fig, ax = plt.subplots(figsize=(12,6))



ax.plot(x, x+1, color="red", linewidth=0.25)

ax.plot(x, x+2, color="red", linewidth=0.50)

ax.plot(x, x+3, color="red", linewidth=1.00)

ax.plot(x, x+4, color="red", linewidth=2.00)



# possible linestype options ‘-‘, ‘–’, ‘-.’, ‘:’, ‘steps’

ax.plot(x, x+5, color="green", lw=3, linestyle='-')

ax.plot(x, x+6, color="green", lw=3, ls='-.')

ax.plot(x, x+7, color="green", lw=3, ls=':')



# custom dash

line, = ax.plot(x, x+8, color="black", lw=1.50)

line.set_dashes([5, 10, 15, 10]) # format: line length, space length, ...



# possible marker symbols: marker = '+', 'o', '*', 's', ',', '.', '1', '2', '3', '4', ...

ax.plot(x, x+ 9, color="blue", lw=3, ls='-', marker='+')

ax.plot(x, x+10, color="blue", lw=3, ls='--', marker='o')

ax.plot(x, x+11, color="blue", lw=3, ls='-', marker='s')

ax.plot(x, x+12, color="blue", lw=3, ls='--', marker='1')



# marker size and color

ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)

ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)

ax.plot(x, x+15, color="purple", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")

ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8, 

        markerfacecolor="yellow", markeredgewidth=3, markeredgecolor="green");
fig, axes = plt.subplots(1, 3, figsize=(12, 4))



axes[0].plot(x, x**2, x, x**3)

axes[0].set_title("default axes ranges")



axes[1].plot(x, x**2, x, x**3)

axes[1].axis('tight')

axes[1].set_title("tight axes")



axes[2].plot(x, x**2, x, x**3)

axes[2].set_ylim([0, 60])

axes[2].set_xlim([2, 5])

axes[2].set_title("custom axes range");
plt.scatter(x,y)
from random import sample

data = sample(range(1, 1000), 100)

plt.hist(data)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]



# rectangular box plot

plt.boxplot(data,vert=True,patch_artist=True);   