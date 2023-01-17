# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib as mlp

import matplotlib.pyplot as plt

import numpy as np



# plot simple sin & cos function



plt.style.use('classic')



x = np.linspace(1,10,100)

plt.plot(x,np.sin(x))

plt.plot(x,np.cos(x))

plt.show()  # plt.show() starts an event loop, looks for all currently active figure objects,and opens one or more interactive windows that display your figure or figures.



%matplotlib
x = np.linspace(0, 10, 100)

fig=plt.figure()

plt.plot(x,np.sin(x),'_')

plt.plot(x,np.cos(x), '_')



fig.savefig('my_figure.png')

fig.canvas.get_supported_filetypes()

# one more way to draw graph



plt.figure()



# create the first of two panels and set current axis



plt.subplot(2,1,1) # (rows, columns, panel number)

plt.plot(x, np.sin(x))



# create the second panel and set current axis



plt.subplot(2,1,2)

plt.plot(x, np.cos(x))
# Simple Line Plots

# Perhaps the simplest of all plots is the visualization of a single function y = f(x) . Here we will take a first look at creating a simple plot of this type. As with all the following

# sections, we’ll start by setting up the notebook for plotting and importing the func‐ tions we will use:



%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import numpy as np



# For all Matplotlib plots, we start by creating a figure and an axes. In their simplest form, a figure and axes can be created as follows



fig=plt.figure()

ax=plt.axis()



# Once we have created an axes, we can use the ax.plot function to plot some data. Let’s start with a simple sinusoid

x=np.linspace(1,10,1000)



x = np.linspace(0, 10, 1000)

plt.plot(x, np.sin(x))



# If we want to create a single figure with multiple lines, we can simply call the plot function multiple times

plt.plot(x, np.sin(x))

plt.plot(x, np.cos(x))

# plt.plot(x, np.tan(x))



# Adjusting the Plot: Line Colors and Styles



# The first adjustment you might wish to make to a plot is to control the line colors and styles. The plt.plot() function takes additional arguments that can be used to spec‐

# ify these. To adjust the color, you can use the color keyword, which accepts a string argument representing virtually any imaginable color. The color can be specified in a variety of ways





plt.plot(x, np.sin(x - 0), color='blue')      # specify color by name

plt.plot(x, np.sin(x - 1), color='g')        # short color code (rgbcmyk)

plt.plot(x, np.sin(x - 2), color='0.75')     # Grayscale between 0 and 1

plt.plot(x, np.sin(x - 3), color='#FFDD44')      # Hex code (RRGGBB from 00 to FF)

plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3))    # RGB tuple, values 0 and 1

plt.plot(x, np.sin(x - 5), color='chartreuse');     # all HTML color names supported



# If no color is specified, Matplotlib will automatically cycle through a set of default colors for multiple lines.



# Similarly, you can adjust the line style using the linestyle keyword



plt.plot(x, x + 0, linestyle='solid')

plt.plot(x, x + 1, linestyle='dashed')

plt.plot(x, x + 2, linestyle='dashdot')

plt.plot(x, x + 3, linestyle='dotted')

    

 # For short, you can use the following codes:



plt.plot(x, x + 4, linestyle='-') # solid

plt.plot(x, x + 5, linestyle='--') # dashed

plt.plot(x, x + 6, linestyle='-.') # dashdot

plt.plot(x, x + 7, linestyle=':') # dotted

# If you would like to be extremely terse, these linestyle and color codes can be com‐ bined into a single nonkeyword argument to the plt.plot() function



plt.plot(x, x + 0, '-g') # solid green  # x & x+1 is drawing a line here

plt.plot(x, x + 1, '--c') # dashed cyan

plt.plot(x, x + 2, '-.k') # dashdot black

plt.plot(x, x + 3, ':r'); # dotted red



# These single-character color codes reflect the standard abbreviations in the RGB (Red/Green/Blue) and CMYK (Cyan/Magenta/Yellow/blacK) color systems, com‐monly used for digital color graphics.

# Adjusting the Plot: Axes Limits



# Matplotlib does a decent job of choosing default axes limits for your plot, but some‐times it’s nice to have finer control. The most basic way to adjust axis limits is to use the plt.xlim() and plt.ylim() methods



plt.plot(x,np.sin(x))

plt.xlim(0, 11)

plt.ylim(0, 1.5)
# If for some reason you’d like either axis to be displayed in reverse, you can simply reverse the order of the arguments



plt.plot(x,np.sin(x))

plt.xlim(10,0)

plt.ylim(1.2, -1.2)
# A useful related method is plt.axis() (note here the potential confusion between axes with an e, and axis with an i). The plt.axis() method allows you to set the x

# and y limits with a single call, by passing a list that specifies [xmin, xmax, ymin,ymax]



plt.plot(x,np.sin(x))

plt.axis([-1,11,0,6])
# The plt.axis() method goes even beyond this, allowing you to do things like auto‐ matically tighten the bounds around the current plot



plt.plot(x,np.sin(x))

plt.axis('tight')
# It allows even higher-level specifications, such as ensuring an equal aspect ratio so that on your screen, one unit in x is equal to one unit in y



plt.plot(x,np.sin(x))

plt.axis('equal')
# Labeling Plots

# we’ll briefly look at the labeling of plots: titles, axis labels, and simple legends.

# Titles and axis labels are the simplest such labels—there are methods that can be used to quickly set them



plt.plot(x,np.sin(x))

plt.title('A sign curve')

plt.xlabel("x value")

plt.ylabel("sinx value")
# When multiple lines are being shown within a single axes, it can be useful to create a plot legend that labels each line type.

# Again, Matplotlib has a built-in way of quickly creating such a legend. It is done via the (you guessed it) plt.legend() method.



plt.plot(x,np.sin(x),'g',label='sin(x)')

plt.plot(x,np.cos(x), 'r',label='cos(x)')



plt.axis('equal')

plt.legend()  # this method is responsible for displaying legend



# As you can see, the plt.legend() function keeps track of the line style and color, and matches these with the correct label. More information on specifying and formatting

# plot legends can be found in the plt.legend() docstring;
# In the object-oriented interface to plotting, rather than calling these functions indi‐ vidually, it is often more convenient to use the ax.set() method to set all these prop‐erties at once



ax = plt.axes()

ax.plot(x, np.sin(x))

ax.set(xlim=(0,10),ylim=(-2,2),xlabel='x',ylabel='sin(x)',title='A sign curve')



# Simple Scatter Plots

# Another commonly used plot type is the simple scatter plot, a close cousin of the line plot. Instead of points being joined by line segments, here the points are represented individually with a dot, circle, or other shape. 



%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import numpy as np



x = np.linspace(0, 10, 30)

y = np.sin(x)



plt.plot(x,y,'o',color='black')



# The third argument in the function call is a character that represents the type of sym‐bol used for the plotting. Just as you can specify options such as '-' and '--' to con‐

# trol the line style, the marker style has its own set of short string codes. 

rng = np.random.RandomState(0)

for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:

    plt.plot(rng.rand(5), rng.rand(5), marker,label="marker='{0}'".format(marker))



plt.legend(numpoints=1)

plt.xlim(0, 1.8);
# For even more possibilities, these character codes can be used together with line and color codes to plot points along with a line connecting them

plt.plot(x,y,'-ok') # line (-), circle marker (o), black (k)
# Additional keyword arguments to plt.plot specify a wide range of properties of the lines and markers



plt.plot(x,y,'-p',color='gray',markersize=15,linewidth=4,markerfacecolor='white',markeredgecolor='gray',markeredgewidth=2)

plt.ylim(-1.2,1.2)

plt.xlim(0,3)

# Scatter Plots with plt.scatter



plt.scatter(x,y,marker='o')
# Let’s show this by creating a random scatter plot with points of many colors and sizes.



rng = np.random.RandomState(0)

x = rng.randn(100)

y = rng.randn(100)

colors = rng.rand(100)

sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,cmap='viridis')

plt.colorbar(); # show color scale



# Notice that the color argument is automatically mapped to a color scale (shown here by the colorbar() command), and the size argument is given in pixels. In this way,

# the color and size of points can be used to convey information in the visualization, in order to illustrate multidimensional data.
#  we might use the Iris data from Scikit-Learn, where each sample is one of three types of flowers that has had the size of its petals and sepals carefully measured



from sklearn.datasets import load_iris

iris = load_iris()

features = iris.data.T

plt.scatter(features[0], features[1], alpha=0.2,s=100*features[3], c=iris.target, cmap='viridis')

plt.xlabel(iris.feature_names[0])

plt.ylabel(iris.feature_names[1])

# Basic Errorbars

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import numpy as np



x=np.linspace(0,10,50)

dy=0.8

y=np.sin(x)+dy*np.random.randn(50)



plt.errorbar(x,y,yerr=dy,fmt='.k')



# Here the fmt is a format code controlling the appearance of lines and points, and has the same syntax as the shorthand used in plt.plot
# In addition to these basic options, the errorbar function has many options to finetune the outputs. Using these additional options you can easily customize the aesthet‐ics of your errorbar plot



plt.errorbar(x,y,yerr=dy,fmt='o',color='black',ecolor='lightgray',elinewidth=3,capsize=0)
%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import numpy as np



# Visualizing a Three-Dimensional Function





# We’ll start by demonstrating a contour plot using a function z = f (x, y)



def f(x,y):

    return np.sin(x)**10+np.cos(10+y*x)*np.cos(x)





# A contour plot can be created with the plt.contour function. It takes three arguments: a grid of x values, a grid of y values, and a grid of z values. The x and y values

# represent positions on the plot, and the z values will be represented by the contour levels. 



x=np.linspace(0,5,50)

y=np.linspace(0,5,40)



# most straightforward way to prepare such data is to use the np.meshgrid function, which builds two-dimensional grids from one-dimensional arrays



X,Y=np.meshgrid(x,y)

Z=f(X,Y)



# Now let’s look at this with a standard line-only contour plot



plt.contour(X,Y,Z,color='black')



# Notice that by default when a single color is used, negative values are represented by dashed lines, and positive values by solid lines.



plt.contour(X, Y, Z, 20, cmap='RdGy')  # we chose the RdGy (short for Red-Gray) colormap
# Our plot is looking nicer, but the spaces between the lines may be a bit distracting. We can change this by switching to a filled contour plot using the plt.contourf()

# function (notice the f at the end), which uses largely the same syntax as plt.contour()



plt.contourf(X, Y, Z, 20, cmap='RdGy')

plt.colorbar()



# The colorbar makes it clear that the black regions are “peaks,” while the red regions are “valleys.”
# A better way to handle this is to use the plt.imshow() function, which inter‐prets a two-dimensional grid of data as an image.



plt.imshow(Z,extent=[0,5,0,5],origin='lower',cmap='RdGy')

plt.colorbar()

plt.axis(aspect='image')
# Finally, it can sometimes be useful to combine contour plots and image plots. For example, to create the effect shown in Figure 4-34, we’ll use a partially transparent

# background image (with transparency set via the alpha parameter) and over-plot contours with labels on the contours themselves (using the plt.clabel() function



contours=plt.contour(X,Y,Z,3,color='black')

plt.clabel(contours,inline=True,fontsize=8)



plt.imshow(Z,extent=[0,5,0,5],origin='lower',cmap='RdGy',alpha=0.5)

plt.colorbar()



# The combination of these three functions—plt.contour, plt.contourf, and plt.imshow—gives nearly limitless possibilities for displaying this sort of threedimensional data within a two-dimensional plot.

# A simple histogram can be a great first step in understanding a dataset. Earlier, we saw a preview of Matplotlib’s histogram function



%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

plt.style.use('seaborn-white')

data=np.random.randn(1000)

plt.hist(data)





# The hist() function has many options to tune both the calculation and the display; here’s an example of a more customized histogram



plt.hist(data, bins=30, alpha=1,histtype='stepfilled', color='red',edgecolor='none')



# The plt.hist docstring has more information on other customization options avail‐ able. I find this combination of histtype='stepfilled' along with some transpar‐

# ency alpha to be very useful when comparing histograms of several distributions
x1=np.random.normal(0,0.8,1000)

x2=np.random.normal(1,2,1000)

x3=np.random.normal(3,4,1000)



test = dict(histtype='stepfilled', alpha=0.3, bins=40)



plt.hist(x1,**test)

plt.hist(x2,**test)

plt.hist(x3,**test)

# If you would like to simply compute the histogram (that is, count the number of points in a given bin) and not display it, the np.histogram() function is available



counts, bin_edges = np.histogram(data, bins=5)

print(counts)

# Two-Dimensional Histograms and Binnings



# Just as we create histograms in one dimension by dividing the number line into bins, we can also create histograms in two dimensions by dividing points among twodimensional bins



# We’ll start by defining some data—an x and y array drawn from a multivariate Gaussian distribution



mean = [0, 0]

cov = [[1, 1], [1, 2]]

x, y = np.random.multivariate_normal(mean, cov, 10000).T



# plt.hist2d: Two-dimensional histogram



# One straightforward way to plot a two-dimensional histogram is to use Matplotlib’s plt.hist2d function



plt.hist2d(x,y,bins=30,cmap='Blues')

cb=plt.colorbar()

cb.set_label('Counts in bin')

counts,xedges,yedges=np.histogram2d(x,y,bins=30)



# For the generalization of this histogram binning in dimensions higher than two, see the np.histogramdd function
# plt.hexbin: Hexagonal binnings

# The two-dimensional histogram creates a tessellation of squares across the axes. Another natural shape for such a tessellation is the regular hexagon. For this purpose,

# Matplotlib provides the plt.hexbin routine, which represents a two-dimensional dataset binned within a grid of hexagons



plt.hexbin(x, y, gridsize=30, cmap='Blues')

cb = plt.colorbar(label='count in bin')



# plt.hexbin has a number of interesting options, including the ability to specify weights for each point, and to change the output in each bin to any NumPy aggregate

# Kernel density estimation



# Another common method of evaluating densities in multiple dimensions is kernel density estimation (KDE)



# One extremely quick and simple KDE implementation exists in the scipy.stats package. Here is a quick example of using the KDE on this data



from scipy.stats import gaussian_kde



# fit an array of size [Ndim, Nsamples]



data=np.vstack([x,y])

kde=gaussian_kde(data)



# evaluate on a regular grid



xgrid = np.linspace(-3.5, 3.5, 40)

ygrid = np.linspace(-6, 6, 40)

Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))



# Plot the result as an image



plt.imshow(Z.reshape(Xgrid.shape),origin='lower', aspect='auto', extent=[-3.5, 3.5, -6, 6],cmap='Blues')

cb = plt.colorbar()

cb.set_label("density")





# Plot legends give meaning to a visualization, assigning labels to the various plot ele‐ments. We previously saw how to create a simple legend; here we’ll take a look at cus‐

# tomizing the placement and aesthetics of the legend in Matplotlib



import matplotlib.pyplot as plt

plt.style.use('classic')

%matplotlib inline

import numpy as np



x=np.linspace(0,10,1000)

fig,ax=plt.subplots()

ax.plot(x,np.sin(x),'b',label='sine')

ax.plot(x, np.cos(x), '--r', label='Cosine')

ax.axis('equal')

leg=ax.legend()

# But there are many ways we might want to customize such a legend. For example, we can specify the location and turn off the frame



ax.legend(loc='upper left',frameon='false')

fig
# We can use the ncol command to specify the number of columns in the legend



ax.legend(frameon=False,loc='lower center',ncol=2)

fig
# We can use a rounded box (fancybox) or add a shadow, change the transparency (alpha value) of the frame, or change the padding around the text



ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

fig
# Choosing Elements for the Legend



# The plt.plot() command is able to create multiple lines at once, and returns a list of created line instances. Passing any of

# these to plt.legend() will tell it which to identify, along with the labels we’d like to specify



y=np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))

lines=plt.plot(x,y)



# lines is a list of plt.Line2D instances

plt.legend(lines[:2], ['first', 'second'])

# I generally find in practice that it is clearer to use the first method, applying labels to the plot elements you’d like to show on the legend



plt.plot(x, y[:, 0], label='first')

plt.plot(x, y[:, 1], label='second')

plt.plot(x, y[:, 2:])

plt.legend(framealpha=1, frameon=True)
# Multiple Legends

# creating a new legend artist from scratch, and then using the lower-level ax.add_artist() method to manually add the second artist to the plot



fig,ax=plt.subplots()

lines=[]

styles= ['-', '--', '-.', ':']

x = np.linspace(0, 10, 1000)



for i in range(4):

    lines += ax.plot(x, np.sin(x - i * np.pi / 2),styles[i], color='black')

    

ax.axis('equal')    



# specify the lines and labels of the first legend

ax.legend(lines[:2], ['line A', 'line B'],loc='upper right', frameon=False)



# Create the second legend and add the artist manually.

from matplotlib.legend import Legend



leg = Legend(ax, lines[2:], ['line C', 'line D'],loc='lower right', frameon=False)

ax.add_artist(leg)





# Customizing Colorbars



import matplotlib.pyplot as plt

plt.style.use('classic')



%matplotlib inline

import numpy as np



# As we have seen several times throughout this section, the simplest colorbar can be created with the plt.colorbar function 



x = np.linspace(0, 10, 1000)

I = np.sin(x) * np.cos(x[:, np.newaxis])



plt.imshow(I)

plt.colorbar()



# We can specify the colormap using the cmap argument to the plotting function that is creating the visualization



plt.imshow(I,cmap='gray')
from matplotlib.colors import LinearSegmentedColormap



def grayscale_cmap(cmap):

 """Return a grayscale version of the given colormap"""

 cmap = plt.cm.get_cmap(cmap)

 colors = cmap(np.arange(cmap.N))

 # convert RGBA to perceived grayscale luminance

 # cf. http://alienryderflex.com/hsp.html

 RGB_weight = [0.299, 0.587, 0.114]

 luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))

 colors[:, :3] = luminance[:, np.newaxis]

 return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

def view_colormap(cmap):

 """Plot a colormap with its grayscale equivalent"""

 cmap = plt.cm.get_cmap(cmap)

 colors = cmap(np.arange(cmap.N))

 cmap = grayscale_cmap(cmap)

 grayscale = cmap(np.arange(cmap.N))

 fig, ax = plt.subplots(2, figsize=(6, 2),

 subplot_kw=dict(xticks=[], yticks=[]))

 ax[0].imshow([colors], extent=[0, 10, 0, 1])

 ax[1].imshow([grayscale], extent=[0, 10, 0, 1])

    

 view_colormap('jet')   
# Color limits and extensions



# Matplotlib allows for a large range of colorbar customization. The colorbar itself is simply an instance of plt.Axes, so all of the axes and tick formatting tricks we’ve

# learned are applicable. The colorbar has some interesting flexibility; for example, we can narrow the color limits and indicate the out-of-bounds values with a triangular

# arrow at the top and bottom by setting the extend property. This might come in handy
# make noise in 1% of the image pixels

speckles = (np.random.random(I.shape) < 0.01)

I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.figure(figsize=(10, 3.5))

plt.subplot(1, 2, 1)

plt.imshow(I, cmap='RdBu')

plt.colorbar()

plt.subplot(1, 2, 2)

plt.imshow(I, cmap='RdBu')

plt.colorbar(extend='both')

plt.clim(-1, 1)
# Discrete colorbars



# Colormaps are by default continuous, but sometimes you’d like to represent discrete values. The easiest way to do this is to use the plt.cm.get_cmap() function, and pass

# the name of a suitable colormap along with the number of desired bins



plt.imshow(I,cmap=plt.cm.get_cmap('Blues',6))

plt.colorbar()

plt.clim(-1,1)
# Example: Handwritten Digits 



# load images of the digits 0 through 5 and visualize several of them



from sklearn.datasets import load_digits

digits=load_digits(n_class=6)



fig,ax=plt.subplots(8,8,figsize=(6, 6))

for i, axi in enumerate(ax.flat):

     axi.imshow(digits.images[i], cmap='binary')

     axi.set(xticks=[], yticks=[])
## project the digits into 2 dimensions using IsoMap



from sklearn.manifold import Isomap

iso=Isomap(n_components=2)

projection=iso.fit_transform(digits.data)



# plot the results



plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))

plt.colorbar(ticks=range(6), label='digit value')

plt.clim(-0.5, 5.5)



%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

import numpy as np



# plt.axes: Subplots by Hand



ax1=plt.axes()  # standard axes

ax2=plt.axes([0.65,0.65,0.2,0.2])
# The equivalent of this command within the object-oriented interface is fig.add_axes(). Let’s use this to create two vertically stacked axes

fig=plt.figure()

ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],xticklabels=[], ylim=(-1.2, 1.2))

ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],ylim=(-1.2, 1.2))

x=np.linspace(0,10)

ax1.plot(np.sin(x))

ax2.plot(np.cos(x))
# plt.subplot: Simple Grids of Subplots

for i in range(1, 7):

 plt.subplot(2, 3, i)

 plt.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')

    
# The command plt.subplots_adjust can be used to adjust the spacing between these plots.



fig=plt.figure()

fig.subplots_adjust(hspace=0.4,wspace=0.4)

for i in range(1,7):

    ax = fig.add_subplot(2, 3, i)

    ax.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')
# plt.subplots: The Whole Grid in One Go



# Here we’ll create a 2×3 grid of subplots, where all axes in the same row share their y-axis scale, and all axes in the same column share their x-axis scale 



fig,ax= plt.subplots(2,3,sharex='col',sharey='row')

# # axes are in a two-dimensional array, indexed by [row, col]

for i in range(2):

    for j in range(3):

        ax[i,j].text(0.5,0.5,str((i,j)),fontsize=18,ha='center')

        

fig



# In comparison to plt.subplot(), plt.subplots() is more consistent with Python’s conventional 0-based indexing
# plt.GridSpec: More Complicated Arrangements



grid=plt.GridSpec(2,3,wspace=0.4,hspace=0.4)



#From this we can specify subplot locations and extents using the familiar Python slic‐ing syntax 



plt.subplot(grid[0,0])

plt.subplot(grid[0, 1:])

plt.subplot(grid[1, :2])

plt.subplot(grid[1, 2])
# # Create some normally distributed data



# Create some normally distributed data

mean = [0, 0]

cov = [[1, 1], [1, 2]]

x, y = np.random.multivariate_normal(mean, cov, 3000).T

# Set up the axes with gridspec

fig = plt.figure(figsize=(6, 6))

grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

main_ax = fig.add_subplot(grid[:-1, 1:])

y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)

x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)





# scatter points on the main axes

main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)

# histogram on the attached axes

x_hist.hist(x, 40, histtype='stepfilled',

orientation='vertical', color='gray')

x_hist.invert_yaxis()

y_hist.hist(y, 40, histtype='stepfilled',orientation='horizontal', color='gray')

y_hist.invert_xaxis()

# Text and Annotation

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib as mpl

plt.style.use('seaborn-whitegrid')

import numpy as np

import pandas as pd
fig,ax=plt.subplots(facecolor='lightgray')

ax.axis([0,10,0,10])



# transform=ax.transData is the default, but we'll specify it anyway



ax.text(1,5,".Data:(1,5)",transform=ax.transData)

ax.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax.transAxes)

ax.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig.transFigure)



ax.set_ylim(-6,6)

ax.set_xlim(0,2)

fig
# Arrows and Annotation



# using the plt.annotate() function. This function creates some text and an arrow, and the arrows can be very flexibly specified.



%matplotlib inline

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)

ax.plot(x, np.cos(x))

ax.axis('equal')



ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),arrowprops=dict(facecolor='red', shrink=5.05))

ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),arrowprops=dict(arrowstyle="->",connectionstyle="angle3,angleA=0,angleB=-90"))
# The arrow style is controlled through the arrowprops dictionary, which has numerous options available.



%matplotlib inline

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)

ax.plot(x, np.cos(x))

ax.axis('equal')
# Customizing Ticks #Major and Minor Ticks



%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import numpy as np



ax = plt.axes(xscale='log', yscale='log')



print(ax.xaxis.get_major_locator())

print(ax.xaxis.get_minor_locator())
print(ax.xaxis.get_major_formatter())

print(ax.xaxis.get_minor_formatter())
# Hiding Ticks or Labels



# the most common tick/label formatting operation is the act of hiding ticks or labels. We can do this using plt.NullLocator() and plt.NullFormatter()



ax=plt.axes()

ax.plot(np.random.rand(50))



ax.yaxis.set_major_locator(plt.NullLocator())

ax.xaxis.set_major_formatter(plt.NullFormatter())

fig,ax=plt.subplots(5,5,figsize=(5,5))

fig.subplots_adjust(hspace=0,wspace=0)



# Get some face data from scikit-learn



from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces().images



for i in range(5):

    for j in range(5):

        ax[i, j].xaxis.set_major_locator(plt.NullLocator())

        ax[i, j].yaxis.set_major_locator(plt.NullLocator())

        ax[i, j].imshow(faces[10 * i + j], cmap="bone")
# Reducing or Increasing the Number of Ticks



fig,ax=plt.subplots(4,4,sharex=True,sharey=True)
# plt.MaxNLocator(), which allows us to specify the maximum number of ticks that will be displayed

# # For every axis, set the x and y major locator



for axi in ax.flat:

     axi.xaxis.set_major_locator(plt.MaxNLocator(3))

     axi.yaxis.set_major_locator(plt.MaxNLocator(3))



    

fig

# Fancy Tick Formats



# Plot a sine and cosine curve

fig, ax = plt.subplots()

x = np.linspace(0, 3 * np.pi, 1000)

ax.plot(x, np.sin(x), lw=3, label='Sine')

ax.plot(x, np.cos(x), lw=3, label='Cosine')



# Set up grid, legend, and limits



ax.grid(True)

ax.legend(frameon=False)

ax.axis('equal')

ax.set_xlim(0, 3 * np.pi)





ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))

ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

fig
def format_func(value,tick_number):

# find number of multiples of pi/2

# we’ll instead use plt.FuncFormatter, which accepts a user-defined function giving fine-grained control over the tick outputs

    N=int(np.round(2*value/np.pi))

    if N==0:

        return "0"

    elif N==1:

        return r"$\pi/2$"

    elif N==2:

        return r"$\pi$"

    elif N%2>0:

        return r"${0}\pi/2$".format(N)

    else:

        return r"${0}\pi$".format(N // 2)





ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))



fig

    

    
# Plot Customization by Hand



import matplotlib.pyplot as plt

plt.style.use('classic')

import numpy as np

%matplotlib inline



x=np.random.randn(1000)

plt.hist(x)



# We can adjust this by hand to make it a much more visually pleasing plot





# draw solid white grid lines

plt.grid(color='w', linestyle='solid')



# hide axis spines

for spine in ax.spines.values():

    spine.set_visible(False)





#Changing the Defaults: rcParams

# We’ll start by saving a copy of the current rcParams dictionary, so we can easily reset these changes in the current session



IPython_default=plt.rcParams.copy()



#Now we can use the plt.rc function to change some of these settings



from matplotlib import cycler

colors=cycler('color',['#EE6666', '#3388BB', '#9988DD','#EECC55', '#88BB44', '#FFBBBB'])

plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',axisbelow=True, grid=True, prop_cycle=colors)

plt.rc('grid', color='w', linestyle='solid')

plt.rc('xtick', direction='out', color='gray')

plt.rc('ytick', direction='out', color='gray')

plt.rc('patch', edgecolor='#E6E6E6')

plt.rc('lines', linewidth=2)



plt.hist(x)
for i in range(4):

    plt.plot(np.random.rand(10))

    
# Stylesheets

# The available styles are listed in plt.style.available



plt.style.available[:5]



#The basic way to switch to a stylesheet is to call



# Let’s create a function that will make two basic types of plot:



def hist_and_lines():

    np.random.seed(0)

    fig,ax=plt.subplots(1,2,figsize=(11, 4))

    ax[0].hist(np.random.randn(1000))

    for i in range(3):

        ax[1].plot(np.random.rand(10))

    ax[1].legend(['a', 'b', 'c'], loc='lower left') 
#Default style



# reset rcParams



hist_and_lines()
#FiveThirtyEight style

with plt.style.context('fivethirtyeight'):

    hist_and_lines()
# ggplot

with plt.style.context('ggplot'):

 hist_and_lines()
#Bayesian Methods for Hackers style

with plt.style.context('bmh'):

 hist_and_lines()
# Dark background



with plt.style.context('dark_background'):

     hist_and_lines()
# Grayscale

with plt.style.context('grayscale'):

     hist_and_lines()
# Seaborn style



import seaborn

hist_and_lines()
# We enable three-dimensional plots by importing the mplot3d toolkit

from mpl_toolkits import mplot3d



# Once this submodule is imported, we can create a three-dimensional axes by passing the keyword projection='3d' to any of the normal axes creation routines



%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt



fig = plt.figure()

ax = plt.axes(projection='3d')
# Three-Dimensional Points and Lines

#we can create these using the ax.plot3D and ax.scatter3D functions.



ax=plt.axes(projection='3d')

# Data for a three-dimensional line



zline=np.linspace(0,15,1000)

yline=np.cos(zline)

xline=np.sin(zline)



ax.plot3D(xline,yline,zline,'red')



## Data for three-dimensional scattered points



zdata = 15 * np.random.random(100)

xdata = np.sin(zdata) + 0.1 * np.random.randn(100)

ydata = np.cos(zdata) + 0.1 * np.random.randn(100)

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')





#Three-Dimensional Contour Plots



def f(x,y):

    return np.sin(np.sqrt(x**2+y**2))



x = np.linspace(-6, 6, 30)

y = np.linspace(-6, 6, 30)



X,Y=np.meshgrid(x,y)

Z=f(X,Y)



fig=plt.figure()

ax = plt.axes(projection='3d')

ax.contour3D(X, Y, Z, 50, cmap='binary')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z')



#Sometimes the default viewing angle is not optimal, in which case we can use the view_init method to set the elevation and azimuthal angles. 



ax.view_init(60,35)

fig
#Wireframes and Surface Plots

fig=plt.figure()

ax = plt.axes(projection='3d')

ax.plot_wireframe(X, Y, Z, color='black')

ax.set_title('wireframe')
# A surface plot is like a wireframe plot, but each face of the wireframe is a filled poly‐gon. Adding a colormap to the filled polygons can aid perception of the topology of

# the surface being visualized



ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

ax.set_title('surface')
r=np.linspace(0,6,20)

theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)

r,theta=np.meshgrid(r,theta)

X=r*np.sin(theta)

Y=r*np.cos(theta)

Z=f(X,Y)

ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
# Surface Triangulations



theta = 2 * np.pi * np.random.random(1000)

r = 6 * np.random.random(1000)

x = np.ravel(r * np.sin(theta))

y = np.ravel(r * np.cos(theta))

z = f(x, y)



# We could create a scatter plot of the points to get an idea of the surface we’re sampling from



ax=plt.axes(projection='3d')

ax.scatter(x,y,z,c=z,cmap='viridis',linewidth=0.5)



# The function that will help us in this case is ax.plot_trisurf, which creates a surface by first finding a set of triangles formed

# between adjacent points



ax=plt.axes(projection='3d')

ax.plot_trisurf(x,y,z,cmap='viridis',edgecolor='none')
# Example: Visualizing a Möbius strip

theta=np.linspace(0,2*np.pi,30)

w=np.linspace(-0.25,0.25,8)

w,theta=np.meshgrid(w,theta)

phi=0.5*theta



# radius in x-y plane

r = 1 + w * np.cos(phi)



x = np.ravel(r * np.cos(theta))

y = np.ravel(r * np.sin(theta))

z = np.ravel(w * np.sin(phi))          

# triangulate in the underlying parameterization



from matplotlib.tri import Triangulation

tri = Triangulation(np.ravel(w), np.ravel(theta))

ax = plt.axes(projection='3d')

ax.plot_trisurf(x, y, z, triangles=tri.triangles,cmap='viridis', linewidths=0.2)

ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
#Seaborn Versus Matplotlib



import matplotlib.pyplot as plt

plt.style.use('classic')

%matplotlib inline

import numpy as np

import pandas as pd



#Now we create some random walk data:



# Create some data



rng = np.random.RandomState(0)

x=np.linspace(0,10,500)

y=np.cumsum(rng.randn(500,6),0)



# Plot the data with Matplotlib defaults

plt.plot(x,y)

plt.legend('ABCDEF',ncol=2,loc='upper left')
import seaborn as sns

sns.set()



#Now let’s rerun the same two lines as before

plt.plot(x,y)

plt.legend('ABCDEF', ncol=2, loc='upper left')
#Histograms, KDE, and densities

# Often in statistical data visualization, all you want is to plot histograms and joint dis‐tributions of variables



data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)

data = pd.DataFrame(data, columns=['x', 'y'])



for col in 'xy':

     plt.hist(data[col], alpha=0.5)
# Rather than a histogram, we can get a smooth estimate of the distribution using a kernel density estimation, which Seaborn does with sns.kdeplot



for col in 'xy':

    sns.kdeplot(data[col],shade=True)
# Histograms and KDE can be combined using distplot



sns.distplot(data['x'])

sns.distplot(data['y'])
# If we pass the full two-dimensional dataset to kdeplot, we will get a two-dimensional visualization of the data

sns.kdeplot(data)
# We can see the joint distribution and the marginal distributions together using sns.jointplot. For this plot, we’ll set the style to a white background



with sns.axes_style('white'):

    sns.jointplot("x","y",data,kind='kde')
# There are other parameters that can be passed to jointplot—for example, we can use a hexagonally based histogram instead

with sns.axes_style('white'):

    sns.jointplot("x","y",data,kind='hex')
# Pair plots



iris=sns.load_dataset("iris")

iris.head()
# Visualizing the multidimensional relationships among the samples is as easy as call‐ing sns.pairplot



sns.pairplot(iris,hue='species',size=2.5)
# Faceted histograms



tips=sns.load_dataset('tips')

tips.head()
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)

grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))
# Factor plots



with sns.axes_style(style='ticks'):

 g = sns.factorplot("day", "total_bill", "sex", data=tips, kind="box")

 g.set_axis_labels("Day", "Total Bill");

# Joint distributions



# Similar to the pair plot we saw earlier, we can use sns.jointplot to show the jointdistribution between different datasets, along with the associated marginal distribu‐tions

with sns.axes_style('white'):

 sns.jointplot("total_bill", "tip", data=tips, kind='hex')
# The joint plot can even do some automatic kernel density estimation and regression



sns.jointplot("total_bill","tip",data=tips,kind='reg')
# Bar plots



# Time series can be plotted with sns.factorplot



planets=sns.load_dataset('planets')

planets.head()
with sns.axes_style('white'):

    g=sns.factorplot("year",data=planets,aspect=2,kind="count",color="steelblue")

    g.set_xticklabels(step=5)
with sns.axes_style('white'):

 g = sns.factorplot("year", data=planets, aspect=4.0, kind='count',hue='method', order=range(2001, 2015))

 g.set_ylabels('Number of Planets Discovered')



    

    
# Example: Exploring Marathon Finishing Times



import numpy as np

import csv as csv

import pandas as pd



teja=pd.read_csv('/input/population-data/bd-dec19-age-specific-fertility-rates.csv',header=0)

teja.head()

# import plotly

import plotly as py

from plotly.offline import init_notebook_mode, iplot,plot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud
# pip install plotly==3.10.0