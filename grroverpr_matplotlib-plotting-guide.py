%matplotlib inline 
# always import like this

import matplotlib.pyplot as plt
import numpy as np  # necessity
import pandas as pd  # necessity
np.random.seed(100)  # random seed set for reproducibility
fig, ax = plt.subplots(1,1) # 1,1 - means 1 row and 1 column (i.e one plot only)

print('figtype:', type(fig))
print('axis_type:' , type(ax))
print('plt_gca_id', id(plt.gca())) 
print('axis_id', id(ax))
plt.close()
print('plt_gca_id', id(plt.gca()))  # new plot is made in new chunk. so new id
print('axis_id', id(ax)) # but ax is still a saved variable from last code chunk
plt.close()
# getter example
one_tick = fig.axes[0].yaxis.get_major_ticks()[0]
print('Type is :', type(one_tick), '\n')

# setter example
print('set title :', fig.axes[0].yaxis.get_major_ticks()[0])
print('get title :', ax.get_title())

# check id of object
print('')
print('id of axes: ', id(fig.axes[0]))

# check with plt.bla()
plt.title('new random name')
print('id now:', id(plt))
plt.close()
x = np.arange(1,11)
y1 = np.arange(10,101, 10)
y2 = np.arange(50,4,-5)
data = np.column_stack((y1, y2)) # just concatenating 2 column arrays


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (12,4)) # 1st width, 2nd height

# customizing axis 1
ax1.plot(x, y1, color='lightblue', linewidth=3) # make line-plot on axis 1
ax1.scatter([2,4,6,5], [5,15,25, 65],           # add scatter plot in axis 1
           color='red',
           marker='^', 
            edgecolor = 'b')
ax1.set_xlim(1, 8)          # set limit of x axis
ax1.set_title('First plot') # set title
ax1.set_xlabel('X label1')
ax1.set_ylabel('Y label1')


# customizing axis 2
ax2.bar(x, y1, color='lightgreen', linewidth=1, edgecolor = 'blue' ) # make bar-plot on axis 2
ax2.set_title('Second plot')
ax2.set_xlabel('$X label2$')
ax2.set_ylabel('$Y label2$')
ax2.yaxis.tick_right()

fig.tight_layout()
y3 = [1,1,1,1,4,4,4,5,5,5,5,5,5,7,7,6]
gridsize = (2, 4) # 4 rows, 2 columns
fig = plt.figure(figsize=(12, 7)) # this creates a figure without axes
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=1, rowspan=2) # 2nd argument = origin of individual box
ax2 = plt.subplot2grid(gridsize, (0, 1), colspan=1, rowspan=2)
ax3 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=1)
ax4 = plt.subplot2grid(gridsize, (1, 2), colspan=2, rowspan=1)

# Now that we have made multiple axes, we can customize each plot individually

# customizing axis 1
ax1.plot(y1, x, color='lightblue', linewidth=3) # make line-plot on axis 1
ax1.scatter([2,4,6,5], [5,15,25, 65],           # add scatter plot in axis 1
           color='red',
           marker='^', 
            edgecolor = 'b')
ax1.set_title('First plot') # set title
ax1.set_xlabel('Y1')
ax1.set_ylabel('X')

# customizing axis 2
ax2.plot(y2, x, color='lightgreen', linewidth=3) # make line-plot on axis 1
ax2.set_xlabel('Y2')
ax2.set_title('Second plot')

# customizing axis 3
ax3.bar(x,y1, color= 'lightgreen', edgecolor = 'b')
ax3.set_title('Barplot of X and Y1')
ax3.set_xlabel('Y1')
ax3.set_ylabel('#')
ax3.yaxis.tick_right()

# customizing axis 4
ax4.hist(y3, color = 'lightblue', edgecolor = 'b')
ax4.set_title('Histogram of y3')
ax4.set_xlabel('Y3')
ax4.set_ylabel('#')
ax4.yaxis.tick_right()

# super title of figure
fig.suptitle('Grids of multiple plots of different sizes ', y = 1.05, fontsize=15)

# clean up whitespace padding
fig.tight_layout()
# creating random array of integers values

img = np.random.randint(0,400,100)
img = img.reshape((10,10)) # 10x10 image
img
fig, ax = plt.subplots()
im = ax.imshow(img, cmap='gist_earth') # use of imshow 
# adding a colorbar below plot to visualize the range of colors

im = ax.imshow(img, cmap='seismic')
fig.colorbar(im, orientation='horizontal')
fig
# let's see values of variables in case we forgot them

x,y1
fig, axes = plt.subplots(2,2, figsize = (12,6)) # axes has all ax1 .. ax2 axes. 
ax1, ax2, ax3, ax4 = axes.flatten() # extracting all 4 axes using flatten

ax1.plot(x,y1, linewidth = 4) # set linewidth
ax1.set_title('Plot 1')

ax2.plot(x,y1, ls = 'solid')  # use solid line
ax2.set_title('Plot 2')

ax3.plot(x,y1, ls = '--') # dashed line
ax3.set_title('Plot 3')

ax4.plot(x,y1, '--', x**2, y1, '-.') # makes 2 lines. one dashed, other dotted dashed
ax4.set_title('Plot 4')


fig.tight_layout()
y3
fig, axes = plt.subplots(1,3, figsize = (12,4))

# also we can directly index from axes variable
axes[0].hist(y3) # Plot a histogram
axes[1].boxplot(y3) # Make a box and whisker plot
axes[2].violinplot(y3) # violin plot

fig.tight_layout()
# cooking some data

data = 2 * np.random.random((10, 10))
data2 = 3 * np.random.random((10, 10))
data.shape, data2.shape
fig2, axes2 = plt.subplots(1,3, figsize = (12,4))

axes2[0].pcolor(data2)    # Pseudocolor plot of 2D array
axes2[1].contour(data)    # Contour plot
axes2[2].contourf(data)   # Filled contour plot

fig2.tight_layout()
# copied pasted same line plots from above
fig, axes = plt.subplots(2,2, figsize = (12,6)) # axes has all ax1 .. ax2 axes. 
ax1, ax2, ax3, ax4 = axes.flatten() # extracting all 4 axes using flatten

ax1.plot(x,y1, linewidth = 4) # set linewidth
ax1.set_title('Plot 1')

ax2.plot(x,y1, linewidth = 4)  # use solid line
ax2.set_title('Plot 2')

ax3.plot(x,y1, ls = '--') # dashed line
ax3.set_title('Plot 3')

ax4.plot(x,y1, '--', x**2, y1, '-.') # makes 2 lines. one dashed, other dotted dashed
ax4.set_title('Plot 4')
ax4.set_label(['x', 'x_square'])

# setting margins now
ax1.margins(x=0.0,y=1)   # Add padding to a plot. a lot of padding to y, no paddig to x
ax2.margins(x=0.6,y=0)   # Add padding to a plot. a lot of padding to x, no paddig to y
ax3.axis('equal')        # Set the aspect ratio of the plot to 1. 
ax4.set(xlim=[0,5],
       ylim=[10,20])     # Set limits for x-and y-axis


fig.tight_layout()
x,y1
# basic scatter plots. bonus: see how to use marker
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (12,4)) # axes has all ax1 .. ax2 axes. 

ax1.scatter(x,y1, marker='.') # set linewidth
ax1.set_title('Plot 1')

ax2.scatter(x,y1, marker="o")  # use solid line
ax2.set_title('Plot 2')

ax1.xaxis.set(ticks=range(1,11), # Manually set x-ticks
 ticklabels=[3,100,-12,"foo", 'whatever', 6,7,8,100,1000])

ax2.tick_params(axis='y',  # Make y-ticks longer and go in and out
               direction='inout',
               length=15)

fig.tight_layout()
ax1.spines['top'].set_visible(False)    # Make the top axis line for a plot invisible
ax1.spines['right'].set_visible(False) # Make the right axis line for a plot invisible


ax2.spines['bottom'].set_position(('outward',30)) # set position of bottom axis outward by scale of 30
fig