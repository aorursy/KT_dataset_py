!pip install joypy

import joypy
!pip install pywaffle

from pywaffle import Waffle
!pip install calmap

import calmap
import os

import numpy as np



import pandas as pd

from pandas.plotting import andrews_curves

from pandas.plotting import parallel_coordinates



from sklearn.cluster import AgglomerativeClustering



import seaborn as sns



#matplotlib and related imports

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.path import Path

from matplotlib.patches import PathPatch

from matplotlib.patches import Patch

import matplotlib.patches as patches



from scipy.spatial import ConvexHull

from scipy.signal import find_peaks

from scipy.stats import sem

import scipy.cluster.hierarchy as shc



import squarify



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import statsmodels.tsa.stattools as stattools

from statsmodels.tsa.seasonal import seasonal_decompose



from dateutil.parser import parse



from IPython.display import Image



import geopandas

import folium

from folium.plugins import TimeSliderChoropleth

from branca.element import Template, MacroElement



def print_files():

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))
# ----------------------------------------------------------------------------------------------------

# get the data

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)

C, S = np.cos(X), np.sin(X)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (15, 5))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

ax.plot(X,S, color = "red", alpha = 0.5, lw = 3, label = "Sine")

ax.plot(X,C, color = "green", alpha = 0.5, lw = 3, label = "Cosine")



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# change the x and y limit

plt.xlim(X.min()*1.5, X.max()*1.5)

plt.ylim(C.min()*1.5, C.max()*1.5)



# change the ticks

# ticks are just a way to 'change the values' represented on the x and y axis

plt.xticks(

    [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],

    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'] # r before each string means raw string

)



plt.yticks(

    [-1, 0, +1],

    [r'$-1$', r'$0$', r'$+1$']

)



# removes the right and top spines

ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')



# changes the position of the other spines

ax.xaxis.set_ticks_position('bottom')

ax.spines['bottom'].set_position(('data',0))

ax.spines['bottom'].set_color('black') # this helps change the color

ax.spines['bottom'].set_alpha(.3) # and adds some transparency to the spines



ax.yaxis.set_ticks_position('left')

ax.spines['left'].set_position(('data',0))

ax.spines['left'].set_color('black')

ax.spines['left'].set_alpha(.3)



# annotate different values

t = 2*np.pi/3

# plot a straight line to connect different points

plt.plot([t, t], [0, np.sin(t)], color ='red', linewidth = 1.5, linestyle = "--", alpha = 0.5)

plt.scatter(t, np.sin(t), 50, color ='red', alpha = 0.5)

plt.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$' + "\nNotice that the arrows are optional",

             xy = (t, np.sin(t)), 

             xycoords = 'data',

             fontsize = 16)



# do the same for cosine

plt.plot([t, t], [0, np.cos(t)], color = 'green', linewidth = 1.5, linestyle = "--", alpha = 0.5)

plt.scatter(t, np.cos(t), 50, color = 'green', alpha = 0.5)

plt.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',

             xy = (t, np.cos(t)), 

             xycoords = 'data',

             xytext = (t/2, -1), 

             fontsize = 16,

             arrowprops = dict(arrowstyle = "->", connectionstyle = "arc3,rad=.2"))



# adjust the x and y ticks

for label in ax.get_xticklabels() + ax.get_yticklabels():

    label.set_fontsize(16)

    label.set_bbox(dict(facecolor = 'white', edgecolor = 'None', alpha = 0.65 ))

    

# add the title adn a legend

plt.title("Sine and Cosine functions (Original by N. Rougier)")

plt.legend(loc = "upper left", fontsize = 12);
# create the figure

fig = plt.figure()



# add a subplot to the figure (the explicit way)

# Passing the numbers is optional and you can pass 111 but I will stick with this way. 

# That's why I call it explicit way.

# 1, 1, 1 means: 1 axes in a 1 row 1 column grid. More on this later.

ax1 = fig.add_subplot(1, 1, 1)



# some data

x = [1, 2, 3, 4, 5]

y = [3, 2, 1, 4, 5]



# plot basic things

ax1.plot(x, y);
# the same plot can be achieved doing this way

fig = plt.figure()

ax1 = fig.subplots()

ax1.plot(x, y);
# you can even simplify it more by just doing this

plt.plot(x, y);
# we can also achieve the same using plt.axes

ax1 = plt.axes()

ax1.plot(x, y);
# and also using plt.subplot

ax1 = plt.subplot()

ax1.plot(x, y);
# now let's to the same but using add_axes()



# create the figure

fig = plt.figure()

# add axes

fig.add_axes()

# gca is get current axes, since matplotlib always plots on the current axes.

ax1 = plt.gca()

# plot

ax1.plot(x, y);
# or doing the same just using axes

ax1 = plt.axes()

ax1.plot(x, y);
fig = plt.figure()

# create a 4 plots and use tuple unpacking to name everyplot

(ax1, ax2), (ax3, ax4) = fig.subplots(2,2)

ax1.plot([1,2,3], color = "red")

ax2.plot([3,2,1], color = "blue")

ax3.plot([4,4,4], color = "orange")

ax4.plot([5,4,5], color = "black")

plt.tight_layout()
# you can do the same using a for loop

nrows = 2

ncolumns = 2

fig, axes = plt.subplots(nrows, ncolumns)



# axes is just a tuple as we saw before

# since se specified 

for row in range(nrows):

    for column in range(ncolumns):

        ax = axes[row, column]

        ax.plot(np.arange(10))
# Look at all the crazy stuff matplotlib allows you to do.



fig = plt.figure(figsize = (20, 10))

# create a 4 plots and use tuple unpacking to name everyplot

(ax1, ax2), (ax3, ax4) = fig.subplots(2,2)

ax1.plot([1,2,3], color = "red")



ax2.plot([3,2,1], color = "blue")



ax3.plot([4,4,4], color = "orange")

ax3_bis = fig.add_axes([0.15, 0.15, 0.15, 0.15])

ax3_bis.plot([1,2,1], color = "pink") # you add it to the figure!

ax3_bis.annotate("Small annotation inside a small added axes",

                xy = (0.5, 0.5),

                xycoords = "axes fraction",

                va = "center",

                ha = "center")



ax4.plot([5,4,5], color = "black")

_ = ax4.annotate("Just to demonstrate the power of matplotlib", 

             xy = (0.5, 0.5), # fraction of the ax4. In the center.

             xycoords = "axes fraction", # you can also specify data and pass the values of the x and y axis.

             va = "center",

             ha = "center")



plt.tight_layout()
fig = plt.figure()

gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0,0])

ax1.plot([1,2,3,])

ax2 = fig.add_subplot(gs[0,1])

ax2.plot([1,3,1,])

ax3 = fig.add_subplot(gs[1,0])

ax3.plot([3,2,1,])

ax4 = fig.add_subplot(gs[1,1])

ax4.plot([3,1,3,])

plt.tight_layout()
fig = plt.figure(figsize = (10, 5))

gs = fig.add_gridspec(3, 3)

ax1 = fig.add_subplot(gs[0, :])

ax1.plot([1,2,1,2])

ax1.set_title("Random text")

ax2 = fig.add_subplot(gs[1,0])

ax2.plot(1,3,1)

ax3 = fig.add_subplot(gs[1,1])

ax3.plot(3,1,3)

ax4 = fig.add_subplot(gs[2,:-1])

ax4.scatter([1,2,3], [1,2,3])

ax5 = fig.add_subplot(gs[1:, -1])

ax5.bar([1,2,3], [1,2,3])

ax5_bis = fig.add_axes([0.75, 0.5, 0.1, 0.1])

ax5_bis.plot([3,1,2])

plt.tight_layout()
# Remmember, use this function to see all the data available for this challenge.

print_files()
# Useful for:

# Visualize the relationship between data.



# More info: 

# https://en.wikipedia.org/wiki/Scatter_plot



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/midwest_filter.csv' 

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12, 6))

ax = fig.add_subplot(1,1,1,)



# ----------------------------------------------------------------------------------------------------

# iterate over each category and plot the data. This way, every group has it's own color. Otherwise everything would be blue

for cat in sorted(list(df["category"].unique())):

    # filter x and the y for each category

    ar = df[df["category"] == cat]["area"]

    pop = df[df["category"] == cat]["poptotal"]

    

    # plot the data

    ax.scatter(ar, pop, label = cat, s = 10)

    

# ----------------------------------------------------------------------------------------------------

# prettify the plot



# eliminate 2/4 spines (lines that make the box/axes) to make it more pleasant

ax.spines["top"].set_color("None") 

ax.spines["right"].set_color("None")



# set a specific label for each axis

ax.set_xlabel("Area") 

ax.set_ylabel("Population")



# change the lower limit of the plot, this will allow us to see the legend on the left

ax.set_xlim(-0.01) 

ax.set_title("Scatter plot of population vs area.")

ax.legend(loc = "upper left", fontsize = 10);
# Useful for:

# Visualize the relationship between data but also helps us encircle a specific group we might want to draw the attention to.



# More info: 

# https://en.wikipedia.org/wiki/Scatter_plot



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/midwest_filter.csv' 

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12, 6))

ax = fig.add_subplot(1,1,1,)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

size_total = df["poptotal"].sum()

# we want every group to have a different marker

markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"] 



# ----------------------------------------------------------------------------------------------------

# iterate over each category and plot the data. This way, every group has it's own color and marker.

for cat, marker in zip(sorted(list(df["category"].unique())), markers):

    # filter x and the y for each category

    ar = df[df["category"] == cat]["area"]

    pop = df[df["category"] == cat]["poptotal"]

    

    # this will allow us to set a specific size for each group.

    size = pop/size_total

    

    # plot the data

    ax.scatter(ar, pop, label = cat, s = size*10000, marker = marker)



# ----------------------------------------------------------------------------------------------------

# create an encircle

# based on this solution

# https://stackoverflow.com/questions/44575681/how-do-i-encircle-different-data-sets-in-scatter-plot



# steps to take:



# filter a specific group

encircle_data = df[df["state"] == "IN"]



# separete x and y

encircle_x = encircle_data["area"]

encircle_y = encircle_data["poptotal"]



# np.c_ concatenates over the second axis

p = np.c_[encircle_x,encircle_y]



# uing ConvexHull (we imported it before) to calculate the limits of the polygon

hull = ConvexHull(p)



# create the polygon with a specific color based on the vertices of our data/hull

poly = plt.Polygon(p[hull.vertices,:], ec = "orange", fc = "none")



# add the patch to the axes/plot)

ax.add_patch(poly)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# eliminate 2/4 spines (lines that make the box/axes) to make it more pleasant

ax.spines["top"].set_color("None")

ax.spines["right"].set_color("None")



# set a specific label for each axis

ax.set_xlabel("Area")

ax.set_ylabel("Population")



# change the lower limit of the plot, this will allow us to see the legend on the left

ax.set_xlim(-0.01) 

ax.set_title("Bubble plot with encircling")

ax.legend(loc = "upper left", fontsize = 10);
# Useful for:

# Visualize the relationship between data but also helps us encircle a specific group we might want to draw the attention to.



# More info: 

# https://en.wikipedia.org/wiki/Scatter_plot



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/midwest_filter.csv' 

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12, 6))

ax = fig.add_subplot(1,1,1,)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

size_total = df["poptotal"].sum()

# we want every group to have a different marker

markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"] 



# ----------------------------------------------------------------------------------------------------

# create an encircle

# based on this solution

# https://stackoverflow.com/questions/44575681/how-do-i-encircle-different-data-sets-in-scatter-plot

def encircle(x,y, ax = None, **kw):

    '''

    Takes an axes and the x and y and draws a polygon on the axes.

    This code separates the differents clusters

    '''

    # get the axis if not passed

    if not ax: ax=plt.gca()

    

    # concatenate the x and y arrays

    p = np.c_[x,y]

    

    # to calculate the limits of the polygon

    hull = ConvexHull(p)

    

    # create a polygon from the hull vertices

    poly = plt.Polygon(p[hull.vertices,:], **kw)

    

    # add the patch to the axes

    ax.add_patch(poly)



# ----------------------------------------------------------------------------------------------------

# iterate over each category and plot the data. This way, every group has it's own color and marker.

# on the iteration we will calculate our hull/polygon for each group and connect specific groups

for cat, marker in zip(sorted(list(df["category"].unique())), markers):

    # filter x and the y for each category

    ar = df[df["category"] == cat]["area"]

    pop = df[df["category"] == cat]["poptotal"]

    

    # this will allow us to set a specific size for each group.

    size = pop/size_total

    

    # plot the data

    ax.scatter(ar, pop, label = cat, s = size*10000, marker = marker)

    

    try:

        # try to add a patch

        encircle(ar, pop, ec = "k", alpha=0.1)

    except:

        # if we don't have enough poins to encircle just pass

        pass



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# eliminate 2/4 spines (lines that make the box/axes) to make it more pleasant

ax.spines["top"].set_color("None")

ax.spines["right"].set_color("None")



# set a specific label for each axis

ax.set_xlabel("Area")

ax.set_ylabel("Population")



# change the lower limit of the plot, this will allow us to see the legend on the left

ax.set_xlim(-0.01) 

ax.set_title("Bubble plot with encircling")

ax.legend(loc = "upper left", fontsize = 10);
# Useful for:

# This is a normal scatter plot but we also plot a simple regression line to see the correlation between the x and the y variables.



# More info: 

# https://visual.ly/m/scatter-plots-regression-lines/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# filter only 2 clases to separate it more easily on the plot

df = df[df["cyl"].isin([4,8])]



# ----------------------------------------------------------------------------------------------------

# plot the data using seaborn

sns.lmplot("displ", "hwy", df, hue = "cyl")



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# since we are using seaborn and this library uses matplotlib behind the scenes

# you can call plt.gca (get current axes) and use all the familiar matplotlib commands

ax = plt.gca()



# change the upper limit of the plot to make it more pleasant

ax.set_xlim(0, 10)

ax.set_ylim(0, 50)



# set title

ax.set_title("Scatter plot with regression");
# Useful for:

# This is a normal scatter plot but we also plot a simple regression line to see the correlation between the x and the y variables.

# This plot is similar to the previous one but plots each data on separate axes



# More info: 

# https://visual.ly/m/scatter-plots-regression-lines/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# filter only 2 clases to separate it more easily on the plot

df = df[df["cyl"].isin([4,8])]





# ----------------------------------------------------------------------------------------------------

# plot the data using seaborn

axes = sns.lmplot("displ", 

                  "hwy", 

                  df, 

                  hue = "cyl", 

                  col = "cyl" # by specifying the col, seaborn creates several axes for each group

                 )



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change the upper limit of the plot to make it more pleasant

axes.set( xlim = (0.5, 7.5), ylim = (0, 50))



# set title for all axes using plt

plt.suptitle("Scatter plot with regression lines on different axes", fontsize = 10);
# Useful for:

# Draw a scatterplot where one variable is categorical. 

# This is useful to see the distribution of the points of each category.



# More info: 

# https://seaborn.pydata.org/generated/seaborn.stripplot.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# separate x and y variables

x = df["cty"]

y = df["hwy"]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

plt.figure(figsize = (10, 7))



# ----------------------------------------------------------------------------------------------------

# plot the data using seaborn

ax = sns.stripplot(x, y)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# set title

ax.set_title("Jitter plot");
# Useful for:

# Draw a scatterplot where one variable is categorical. 

# In this plot we calculate the size of overlapping points in each category and for each y.

# This way, the bigger the bubble the more concentration we have in that region.



# More info: 

# https://seaborn.pydata.org/generated/seaborn.stripplot.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting



# we need to make a groupby by variables of interest

gb_df = df.groupby(["cty", "hwy"]).size().reset_index(name = "counts")



# sort the values

gb_df.sort_values(["cty", "hwy", "counts"], ascending = True, inplace = True)



# create a color for each group. 

# there are several way os doing, you can also use this line: 

# colors = [plt.cm.gist_earth(i/float(len(gb_df["cty"].unique()))) for i in range(len(gb_df["cty"].unique()))]

colors = {i:np.random.random(3,) for i in sorted(list(gb_df["cty"].unique()))}



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 5))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# iterate over each category and plot the data. This way, every group has it's own color and sizwe.

for x in sorted(list(gb_df["cty"].unique())):

    

    # get x and y values for each group

    x_values = gb_df[gb_df["cty"] == x]["cty"]

    y_values = gb_df[gb_df["cty"] == x]["hwy"]

    

    # extract the size of each group to plot

    size = gb_df[gb_df["cty"] == x]["counts"]

    

    # extract the color for each group and covert it from rgb to hex

    color = matplotlib.colors.rgb2hex(colors[x])

    

    # plot the data

    ax.scatter(x_values, y_values, s = size*10, c = color)

    

# ----------------------------------------------------------------------------------------------------

# prettify the plot



# set title

ax.set_title("Count plot");
# Useful for:

# This plot is a combination of 2 plots.

# On one side we have a normal scatter plot that is helpful to see the relationship between data (x and y axis)

# But we also add a histogram that is useful to see the concentration/bins and the distribution of a series.



# More info: 

# https://en.wikipedia.org/wiki/Histogram



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# separate x and y

x = df["displ"]

y = df["hwy"]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 5))

# in this case we use gridspec.

# check the basics section of this kernel if you need help.

gs = fig.add_gridspec(5, 5)

ax1 = fig.add_subplot(gs[:4, :-1])



# ----------------------------------------------------------------------------------------------------

# plot the data



# main axis: scatter plot

# this line is very nice c = df.manufacturer.astype('category').cat.codes

# since it basically generate a color for each category

ax1.scatter(x, y, c = df.manufacturer.astype('category').cat.codes) 



# set the labels for x and y

ax1.set_xlabel("Dist")

ax1.set_ylabel("Hwy")



# set the title for the main plot

ax1.set_title("Scatter plot with marginal histograms")



# prettify the plot

# get rid of some of the spines to make the plot nicer

ax1.spines["right"].set_color("None")

ax1.spines["top"].set_color("None")



# using familiar slicing, get the bottom axes and plot

ax2 = fig.add_subplot(gs[4:, :-1])

ax2.hist(x, 40, orientation = 'vertical', color = "pink")



# invert the axis (it looks up side down)

ax2.invert_yaxis()



# prettify the plot

# set the ticks to null

ax2.set_xticks([])

ax2.set_yticks([])

# no axis to make plot nicer

ax2.axison = False



# using familiar slicing, get the left axes and plot

ax3 = fig.add_subplot(gs[:4, -1])

ax3.hist(y, 40, orientation = "horizontal", color = "pink")



# prettify the plot

# set the ticks to null

ax3.set_xticks([])

ax3.set_yticks([])

# no axis to make plot nicer

ax3.axison = False



# make all the figures look nicier

fig.tight_layout()
# Useful for:

# A box plot or boxplot is a method for graphically depicting groups of numerical data through their quartiles.

# It helps to see the dispersion of a series, thanks to the whiskers



# More info: 

# https://en.wikipedia.org/wiki/Box_plot



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv'

df = pd.read_csv(PATH)





# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

x = df["displ"]

y = df["hwy"]



# in this plot we create the colors separatly

colors = df["manufacturer"].astype("category").cat.codes



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 5))

# in this case we use gridspec.

# check the basics section of this kernel if you need help.

gs = fig.add_gridspec(6, 6)

ax1 = fig.add_subplot(gs[:4, :-1])



# ----------------------------------------------------------------------------------------------------

# plot the data



# main axis: scatter plot

# this line is very nice c = df.manufacturer.astype('category').cat.codes

# since it basically generate a color for each category

ax1.scatter(x, y, c = df.manufacturer.astype('category').cat.codes) 



# set the labels for x and y

ax1.set_xlabel("Dist")

ax1.set_ylabel("Hwy")



# set the title for the main plot

ax1.set_title("Scatter plot with marginal histograms")



# prettify the plot

# get rid of some of the spines to make the plot nicer

ax1.spines["right"].set_color("None")

ax1.spines["top"].set_color("None")



# using familiar slicing, get the left axes and plot

ax2 = fig.add_subplot(gs[4:, :-1])

ax2.boxplot(x, 

            vert = False,  

            whis = 0.75 # make the boxplot lines shorter

           )

# prettify the plot

# set the ticks to null

ax2.set_xticks([])

ax2.set_yticks([])



# left plot

ax3 = fig.add_subplot(gs[:4, -1])

ax3.boxplot(y,  

            whis = 0.75 # make the boxplot lines shorter

           )

# prettify the plot

# set the ticks to null

ax3.set_xticks([])

ax3.set_yticks([])



# make all the figures look nicier

fig.tight_layout()
# Useful for:

# The correlation plot helps us to comparte how correlated are 2 variables between them



# More info: 

# https://en.wikipedia.org/wiki/Covariance_matrix#Correlation_matrix



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/mtcars.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 5))

ax = fig.add_subplot()



# plot using matplotlib

ax.imshow(df.corr(), cmap = 'viridis', interpolation = 'nearest')

# set the title for the figure

ax.set_title("Heatmap using matplotlib");
# Useful for:

# The correlation plot helps us to comparte how correlated are 2 variables between them



# More info: 

# https://en.wikipedia.org/wiki/Covariance_matrix#Correlation_matrix



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/mtcars.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# calculate the correlation between all variables

corr = df.corr()

# create a mask to pass it to seaborn and only show half of the cells 

# because corr between x and y is the same as the y and x

# it's only for estetic reasons

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 5))



# plot the data using seaborn

ax = sns.heatmap(corr, 

                 mask = mask, 

                 vmax = 0.3, 

                 square = True,  

                 cmap = "viridis")

# set the title for the figure

ax.set_title("Heatmap using seaborn");
# Useful for:

# Plot pairwise relationships in a dataset. 

# Helps you to see in a glance of an eye all distribution and correlation of variables.



# More info: 

# https://seaborn.pydata.org/generated/seaborn.pairplot.html



# ----------------------------------------------------------------------------------------------------

# get the data

df = sns.load_dataset('iris')



# plot the data using seaborn

sns.pairplot(df, 

             hue = "species" # helps to separate the values by specios

            );
# Useful for:

# Plot pairwise relationships in a dataset. 

# Helps you to see in a glance of an eye all distribution and correlation of variables.

# This plot also plots a regression line to fit each of the data



# More info: 

# https://seaborn.pydata.org/generated/seaborn.pairplot.html



# ----------------------------------------------------------------------------------------------------

# get the data

df = sns.load_dataset('iris')



# plot the data using seaborn

sns.pairplot(df, 

             kind = "reg", # make a regression line for eac hue and each variables

             hue = "species"

            );
# Useful for:

# Based on a metric to compare, this plot helps you to see the divergence of the a value 

# to that metric (it could be mean, median or others).



# More info: 

# https://blog.datawrapper.de/divergingbars/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mtcars.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# here we standarize the data

# More info:

# https://statisticsbyjim.com/glossary/standardization/

df["x_plot"] = (df["mpg"] - df["mpg"].mean())/df["mpg"].std()



# sort value and reset the index

df.sort_values("x_plot", inplace = True)

df.reset_index(inplace = True)



# create a color list, where if value is above > 0 it's green otherwise red

colors = ["red" if x < 0 else "green" for x in df["x_plot"]]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 8))

ax = fig.add_subplot()

# plot using horizontal lines and make it look like a column by changing the linewidth

ax.hlines(y = df.index, xmin = 0 , xmax = df["x_plot"],  color = colors, linewidth = 5)



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# set x and y axis

ax.set_xlabel("Mileage")

ax.set_ylabel("Car Name")



# set a title

ax.set_title("Diverging plot in matplotlib")



# make a grid to help separate the lines

ax.grid(linestyle='--', alpha=0.5)



# change the y ticks

# first you set the yticks

ax.set_yticks(df.index)



# then you change them using the car names

# same can be achived using plt.yticks(df.index, df.cars)

ax.set_yticklabels(df.cars);
# Useful for:

# This plot is really useful to show the different performance of deviation of data.

# We use text to annotate the value and make more easy the comparison.



# More info: 

# https://blog.datawrapper.de/divergingbars/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mtcars.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# here we standarize the data

# More info:

# https://statisticsbyjim.com/glossary/standardization/

df["x_plot"] = (df["mpg"] - df["mpg"].mean())/df["mpg"].std()



# sort value and reset the index

df.sort_values("x_plot", inplace = True)

df.reset_index(inplace=True)



# create a color list, where if value is above > 0 it's green otherwise red

colors = ["red" if x < 0 else "green" for x in df["x_plot"]]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 8))

ax = fig.add_subplot()



# plot horizontal lines that go from zero to the value

# here we make the linewidht very thin.

ax.hlines(y = df.index, xmin = 0 , color = colors,  xmax = df["x_plot"], linewidth = 1)



# ----------------------------------------------------------------------------------------------------

# plot the data

# iterate over x and y and annotate text and plot the data

for x, y in zip(df["x_plot"], df.index):

    # annotate text

    ax.text(x - 0.1 if x < 0 else x + 0.1, 

             y, 

             round(x, 2), 

             color = "red" if x < 0 else "green",  

             horizontalalignment='right' if x < 0 else 'left', 

             size = 10)

    # plot the points

    ax.scatter(x, 

                y, 

                color = "red" if x < 0 else "green", 

                alpha = 0.5)



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# set title

ax.set_title("Diverging plot in matplotlib")

# change x lim

ax.set_xlim(-3, 3)



# set labels

ax.set_xlabel("Mileage")

ax.set_ylabel("Car Name")



# make a grid

ax.grid(linestyle='--', alpha=0.5)



# instead of y = 1, 2, 3...

# put the car makers on the y axis

ax.set_yticks(df.index)

ax.set_yticklabels(df.cars)



# change the spines to make it nicer

ax.spines["top"].set_color("None")

ax.spines["left"].set_color("None")



# with this line, we change the right spine to be in the middle

# as a vertical line from the origin

ax.spines['right'].set_position(('data',0))

ax.spines['right'].set_color('black')
# Useful for:

# This plot is really useful to show the different performance of deviation of data.

# We use text to annotate the value and make more easy the comparison.

# This plot is very similar to the previous 2

# But here we don't draw any lines and just play with the size of each point and make it a little bigger



# More info: 

# https://blog.datawrapper.de/divergingbars/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mtcars.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# here we standarize the data

# More info:

# https://statisticsbyjim.com/glossary/standardization/

df["x_plot"] = (df["mpg"] - df["mpg"].mean())/df["mpg"].std()



# sort value and reset the index

df.sort_values("x_plot", inplace = True)

df.reset_index(inplace=True)



# create a color list, where if value is above > 0 it's green otherwise red

colors = ["red" if x < 0 else "green" for x in df["x_plot"]]





# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

# iterate over x and y and annotate text and plot the data

for x, y in zip(df["x_plot"], df.index):

    

    # make a horizontal line from the y till the x value

    # this doesn't appear in the original 50 plot challenge

    ax.hlines(y = y, 

               xmin = -3,  

               xmax = x, 

               linewidth = 0.5,

               alpha = 0.3,

               color = "red" if x < 0 else "green")

    

    # annotate text

    ax.text(x, 

             y, 

             round(x, 2), 

             color = "black",

             horizontalalignment='center', 

             verticalalignment='center',

             size = 8)

    

    # plot the points

    ax.scatter(x, 

                y, 

                color = "red" if x < 0 else "green", 

                s = 300,

                alpha = 0.5)



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# set title

ax.set_title("Diverging plot in matplotlib")



# change x lim

ax.set_xlim(-3, 3)



# set labels

ax.set_xlabel("Mileage")

ax.set_ylabel("Car Name")



# instead of y = 1, 2, 3...

# put the car makers on the y axis

ax.set_yticks(df.index)

ax.set_yticklabels(df.cars)



# change the spines to make it nicer

ax.spines["top"].set_color("None")

ax.spines["left"].set_color("None")



# with this line, we change the right spine to be in the middle

# as a vertical line from the origin

ax.spines['right'].set_position(('data',0))

ax.spines['right'].set_color('grey')
# Useful for:

# This plot is really useful to show the different performance of deviation of data.

# In this plot we use rectagles and matplotlib patches to draw the attention to specific points



# More info: 

# https://blog.datawrapper.de/divergingbars/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mtcars.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# here we standarize the data

# More info:

# https://statisticsbyjim.com/glossary/standardization/

df["x_plot"] = (df["mpg"] - df["mpg"].mean())/df["mpg"].std()



# sort value and reset the index

df.sort_values("x_plot", inplace = True)

df.reset_index(inplace = True)



# we plot everything with a black color except a specific Fiat model

# this way we visually communicate something to the user

df["color"] = df["cars"].apply(lambda car_name: "orange" if car_name == "Fiat X1-9" else "black")





# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (8, 12))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

# plot horizontal lines from the origin to each data point

ax.hlines(y = df.index, 

          xmin = 0,

          xmax = df["x_plot"],

          color = df["color"],

          alpha = 0.6)



# plot the dots

ax.scatter(x = df["x_plot"],

          y = df.index,

          s = 100,

          color = df["color"],

          alpha = 0.6)



# add patches

# with this piece of code, we can draw pretty much any patch or shape

# since we are interested in a rectangle, we must submit a list with the 

# coordinates

def add_patch(verts, ax, color):

    '''

    Takes the vertices and the axes as argument and adds the patch to our plot.

    '''

    codes = [

        Path.MOVETO,

        Path.LINETO,

        Path.LINETO,

        Path.LINETO,

        Path.CLOSEPOLY,

    ]



    path = Path(verts, codes)

    pathpatch = PathPatch(path, facecolor = color, lw = 2, alpha = 0.3)

    ax.add_patch(pathpatch)



# coordinates for the bottom shape

verts_bottom = [

   (-2.5, -0.5),  # left, bottom

   (-2.5, 2),  # left, top

   (-1.5, 2),  # right, top

   (-1.5, -0.5),  # right, bottom

   (0., 0.),  # ignored

]



# coordinates for the upper shape

verts_upper = [

   (1.5, 27),  # left, bottom

   (1.5, 33),  # left, top

   (2.5, 33),  # right, top

   (2.5, 27),  # right, bottom

   (0., 0.),  # ignored

]



# use the function to add them to the existing plot

add_patch(verts_bottom, ax, color = "red")

add_patch(verts_upper, ax, color = "green")



# annotate text

ax.annotate('Mercedes Models', 

            xy = (0.0, 11.0), 

            xytext = (1.5, 11), 

            xycoords = 'data', 

            fontsize = 10, 

            ha = 'center', 

            va = 'center',

            bbox = dict(boxstyle = 'square', fc = 'blue', alpha = 0.1),

            arrowprops = dict(arrowstyle = '-[, widthB=2.0, lengthB=1.5', lw = 2.0, color = 'grey'), color = 'black')



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# set title

ax.set_title("Diverging Lollipop of Car Mileage")



# autoscale

ax.autoscale_view()



# change x lim

ax.set_xlim(-3, 3)



# set labels

ax.set_xlabel("Mileage")

ax.set_ylabel("Car Name")



# instead of y = 1, 2, 3...

# put the car makers on the y axis

ax.set_yticks(df.index)

ax.set_yticklabels(df.cars)



# change the spines to make it nicer

ax.spines["right"].set_color("None")

ax.spines["top"].set_color("None")



# add a grid

ax.grid(linestyle='--', alpha=0.5);
# Useful for:

# This plot is really useful to show the different performance of deviation of data.

# In this plot we use rectagles and matplotlib patches to draw the attention to specific points

# This example shows how to add patches more easily



# More info: 

# https://blog.datawrapper.de/divergingbars/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mtcars.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# here we standarize the data

# More info:

# https://statisticsbyjim.com/glossary/standardization/

df["x_plot"] = (df["mpg"] - df["mpg"].mean())/df["mpg"].std()



# sort value and reset the index

df.sort_values("x_plot", inplace = True)

df.reset_index(inplace = True)



# we plot everything with a black color except a specific Fiat model

# this way we visually communicate something to the user

df["color"] = df["cars"].apply(lambda car_name: "orange" if car_name == "Fiat X1-9" else "black")





# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (8, 12))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

# plot horizontal lines from the origin to each data point

ax.hlines(y = df.index, 

          xmin = 0,

          xmax = df["x_plot"],

          color = df["color"],

          alpha = 0.6)



# plot the dots

ax.scatter(x = df["x_plot"],

          y = df.index,

          s = 100,

          color = df["color"],

          alpha = 0.6)



# add patches more easily

# It's easier to draw shapes like rectangles or squares, since this way

# we only must specify 2 points, and matplotlib does all the job

p1 = patches.Rectangle((-2.0, -1), width = .3, height = 3, alpha = .2, facecolor = 'red')

p2 = patches.Rectangle((1.5, 27), width = .8, height = 5, alpha = .2, facecolor = 'green')

ax.add_patch(p1)

ax.add_patch(p2)



# annotate text

ax.annotate('Mercedes Models', 

            xy = (0.0, 11.0), 

            xytext = (1.5, 11), 

            xycoords = 'data', 

            fontsize = 10, 

            ha = 'center', 

            va = 'center',

            bbox = dict(boxstyle = 'square', fc = 'blue', alpha = 0.1),

            arrowprops = dict(arrowstyle = '-[, widthB=2.0, lengthB=1.5', lw = 2.0, color = 'grey'), color = 'black')



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# set title

ax.set_title("Diverging Lollipop of Car Mileage")



# autoscale

ax.autoscale_view()



# change x lim

ax.set_xlim(-3, 3)



# set labels

ax.set_xlabel("Mileage")

ax.set_ylabel("Car Name")



# instead of y = 1, 2, 3...

# put the car makers on the y axis

ax.set_yticks(df.index)

ax.set_yticklabels(df.cars)



# change the spines to make it nicer

ax.spines["right"].set_color("None")

ax.spines["top"].set_color("None")



# add a grid

ax.grid(linestyle='--', alpha=0.5);
# (omit pandas warning: "Using an implicitly registered datetime converter for a matplotlib plotting method. ...")

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



# Useful for:

# Area chart is really useful when you want to drawn the attention about when a series is below a certain point.

# The area between axis and line are commonly emphasized with colors, textures and hatchings. 

# Commonly one compares two or more quantities with an area chart.



# More info: 

# https://en.wikipedia.org/wiki/Area_chart



# ----------------------------------------------------------------------------------------------------

# get the data



PATH = "/kaggle/input/the-50-plot-challenge/economics.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# create the variation between 2 consecutive rows

df["pce_monthly_change"] = (df["psavert"] - df["psavert"].shift(1))/df["psavert"].shift(1)



# convert todatetime

df["date_converted"] = pd.to_datetime(df["date"])



# filter our df for a specific date

df = df[df["date_converted"] < np.datetime64("1975-01-01")]



# separate x and y 

x = df["date_converted"]

y = df["pce_monthly_change"]



# calculate the max values to annotate on the plot

y_max = y.max()



# find the index of the max value

x_ind = np.where(y == y_max)



# find the x based on the index of max

x_max = x.iloc[x_ind]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (15, 10))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

ax.plot(x, y, color = "black")

ax.scatter(x_max, y_max, s = 300, color = "green", alpha = 0.3)



# annotate the text of the Max value

ax.annotate(r'Max value',

             xy = (x_max, y_max), 

             xytext = (-90, -50), 

             textcoords = 'offset points', 

             fontsize = 16,

             arrowprops = dict(arrowstyle = "->", connectionstyle = "arc3,rad=.2")

           )

# ----------------------------------------------------------------------------------------------------

# prettify the plot

# fill the area with a specific color

ax.fill_between(x, 0, y, where = 0 > y, facecolor='red', interpolate = True, alpha = 0.3)

ax.fill_between(x, 0, y, where = 0 <= y, facecolor='green', interpolate = True, alpha = 0.3)



# change the ylim to make it more pleasant for the viewer

ax.set_ylim(y.min() * 1.1, y.max() * 1.1)



# change the values of the x axis

# extract the first 3 letters of the month

xtickvals = [str(m)[:3].upper() + "-" + str(y) for y,m in zip(df.date_converted.dt.year, df.date_converted.dt.month_name())]



# this way we can set the ticks to be every 6 months.

ax.set_xticks(x[::6])



# change the current ticks to be our string month value

# basically pass from this: 1967-07-01

# to this: JUL-1967

ax.set_xticklabels(xtickvals[::6], rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})



# add a grid

ax.grid(alpha = 0.3)



# set the title

ax.set_title("Monthly variation return %");
# Useful for:

# This is a normal bar chart but ordered in a specific way.

# From the lowest to the highest values

# It's useful to show comparisons among discrete categories.



# More info: 

# https://en.wikipedia.org/wiki/Bar_chart



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# groupby and create the target x and y

gb_df = df.groupby(["manufacturer"])["cyl", "displ", "cty"].mean()

gb_df.sort_values("cty", inplace = True)

# fitler x and y

x = gb_df.index

y = gb_df["cty"]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

for x_, y_ in zip(x, y):

    # this is very cool, since we can pass a function to matplotlib

    # and it will plot the color based on the result of the evaluation

    ax.bar(x_, y_, color = "red" if y_ < y.mean() else "green", alpha = 0.3)

    

     # add some text

    ax.text(x_, y_ + 0.3, round(y_, 1), horizontalalignment = 'center')



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# Add a patch below the x axis line to enphasize that they are below the mean

# I had to copy and paste this code, since I didn't manage to figure it out.

# red one

p2 = patches.Rectangle((.124, -0.005), width = .360, height = .13, alpha = .1, facecolor = 'red', transform = fig.transFigure)

fig.add_artist(p2)



# green one

p1 = patches.Rectangle((.124 + .360, -0.005), width = .42, height = .13, alpha = .1, facecolor = 'green', transform = fig.transFigure)

fig.add_artist(p1)



# rotate the x ticks 90 degrees

ax.set_xticklabels(x, rotation=90)



# add an y label

ax.set_ylabel("Average Miles per Gallon by Manufacturer")



# set a title

ax.set_title("Bar Chart for Highway Mileage");
# Useful for:

# The purpose of this kind of chart is the same as a normal bar chart.

# The lollipop chart is often claimed to be useful compared to a normal bar chart, 

# if you are dealing with a large number of values and when the values are all high, such as in the 80-90% range (out of 100%). 

# Then a large set of tall columns can be visually aggressive.



# More info: 

# https://datavizproject.com/data-type/lollipop-chart/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# groupby and create the target x and y

gb_df = df.groupby(["manufacturer"])["cyl", "displ", "cty"].mean()

gb_df.sort_values("cty", inplace = True)

# fitler x and y

x = gb_df.index

y = gb_df["cty"]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

for x_, y_ in zip(x, y):

    # make a scatter plot

    ax.scatter(x_, y_, color = "red" if y_ < y.mean() else "green", alpha = 0.3, s = 100)

    

    # add vertical lines to connect them to the data point (head of the lollipop)

    ax.vlines(x_, ymin = 0, ymax = y_, color = "red" if y_ < y.mean() else "green", alpha = 0.3)

    

    # add text with the data

    ax.text(x_, y_ + 0.5, round(y_, 1), horizontalalignment='center')

    

# ----------------------------------------------------------------------------------------------------

# prettify the plot

# change the ylim

ax.set_ylim(0, 30)



# rotate the x ticks 90 degrees

ax.set_xticklabels(x, rotation = 90)



# add an y label

ax.set_ylabel("Average Miles per Gallon by Manufacturer")



# set a title

ax.set_title("Lollipop Chart for Highway Mileage");
# Useful for:

# This plot is the same as the diverging dot plot but here we don't add the line.

# # This plot is really useful to show the different performance of deviation of data.



# More info: 

# https://www.mathsisfun.com/data/dot-plots.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# groupby and create the target x and y

gb_df = df.groupby(["manufacturer"])["cyl", "displ", "cty"].mean()

gb_df.sort_values("cty", inplace = True)

# fitler x and y

x = gb_df.index

y = gb_df["cty"]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

for x_, y_ in zip(x, y):

    ax.scatter(y_, x_, color = "red" if y_ < y.mean() else "green", alpha = 0.3, s = 100)

    

# ----------------------------------------------------------------------------------------------------

# prettify the plot

# change the xlim

ax.set_xlim(8, 27)



# add an y label

ax.set_xlabel("Average Miles per Gallon by Manufacturer")



# set the title

ax.set_title("Dot Plot for Highway Mileage")



# create the grid only for the y axis

ax.grid(which = 'major', axis = 'y', linestyle = '--');
# Useful for:

# This chart is very useful to show the variation of some kind of data

# between two points in time (you can expand it for more points though).



# More info: 

# https://datavizproject.com/data-type/slope-chart/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH  = '/kaggle/input/the-50-plot-challenge/gdppercap.csv'

df = pd.read_csv(PATH)



# create a column with the colors, since we will be iterating and changing the value based on their performance

# if the value at the starting point is bigger than the ending, green color

# otherwise, red color

df["color"] = df.apply(lambda row: "green" if row["1957"] >= row["1952"] else "red", axis = 1)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (8, 12))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

for cont in df["continent"]:

    

    # prepare the data for plotting

    # extract each point and the color

    x_start = df.columns[1]

    x_finish = df.columns[2]

    y_start = df[df["continent"] == cont]["1952"]

    y_finish = df[df["continent"] == cont]["1957"]

    color = df[df["continent"] == cont]["color"]

    

    # plot eac point

    ax.scatter(x_start, y_start, color = color, s = 200)

    ax.scatter(x_finish, y_finish, color = color, s = 200*(y_finish/y_start))

    

    # connect the starting point and the ending point with a line

    # check the bouns section for more

    ax.plot([x_start, x_finish], [float(y_start), float(y_finish)], linestyle = "-", color = color.values[0])

    

    # annotate the value for each continent

    ax.text(ax.get_xlim()[0] - 0.05, y_start, r'{}:{}k'.format(cont, int(y_start)/1000), horizontalalignment = 'right', verticalalignment = 'center', fontdict = {'size':8})

    ax.text(ax.get_xlim()[1] + 0.05, y_finish, r'{}:{}k'.format(cont, int(y_finish)/1000), horizontalalignment = 'left', verticalalignment = 'center', fontdict = {'size':8})



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# get the x and y limits

x_lims = ax.get_xlim()

y_lims = ax.get_ylim()



# change the x and y limits programmaticaly

ax.set_xlim(x_lims[0] - 1, x_lims[1] + 1);



# add 2 vertical lines

ax.vlines(x_start, 0, y_lims[1], color = "grey", alpha = 0.3, lw = 0.5)

ax.vlines(x_finish, 0, y_lims[1], color = "grey", alpha = 0.3, lw = 0.5)



# for each vertical line, add text: BEFORE and AFTER to help understand the plot

ax.text(x_lims[0], y_lims[1], "BEFORE", horizontalalignment = 'right', verticalalignment = 'center')

ax.text(x_lims[1], y_lims[1], "AFTER", horizontalalignment = 'left', verticalalignment = 'center')



# set and x and y label

ax.set_xlabel("Years")

ax.set_ylabel("Mean GPD per Capita")



# add a title

ax.set_title("Slopechart: Comparing GDP per Capita between 1952 and 1957")



# remove all the spines of the axes

ax.spines["left"].set_color("None")

ax.spines["right"].set_color("None")

ax.spines["top"].set_color("None")

ax.spines["bottom"].set_color("None")
# Useful for:

# It's scope if very similar as a slope chart

# Dumbbell plot (also known as Dumbbell chart, Connected dot plot) is great for displaying changes between two points in time, two conditions or differences between two groups.



# More info: 

# https://www.amcharts.com/demos/dumbbell-plot/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/health.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (8, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

for i, area in zip(df.index, df["Area"]):

    

    # extract the data for each area

    start_data = df[df["Area"] == area]["pct_2013"].values[0]

    finish_data = df[df["Area"] == area]["pct_2014"].values[0]

    

    # plot the starting and ending plots

    ax.scatter(start_data, i, c = "blue", alpha = .8)

    ax.scatter(finish_data, i, c = "blue", alpha = .2)

    

    # connect them with an horizontal line

    ax.hlines(i, start_data, finish_data, color = "blue", alpha = .2)

    

# ----------------------------------------------------------------------------------------------------

# prettify the plot



# set x and y label

ax.set_xlabel("Pct change")

ax.set_ylabel("Mean GDP per Capita")



# set the title

ax.set_title("Dumbell Chart: Pct Change - 2013 vs 2014")



# add grid lines for the x axis to better separate the data

ax.grid(axis = "x")



# change the x limit programatically

x_lim = ax.get_xlim()

ax.set_xlim(x_lim[0]*.5, x_lim[1]*1.1)



# change the x ticks to be rounded pct %

x_ticks = ax.get_xticks()

ax.set_xticklabels(["{:.0f}%".format(round(tick*100, 0)) for tick in x_ticks]);
# Useful for:

# This is one of the most fundamental plots to master

# It's shows the approximate distributin of numerical or categorical data.



# More info: 

# https://en.wikipedia.org/wiki/Histogram



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

gb_df = df[["class", "displ"]].groupby("class")

lx = []

ln = []



# handpicked colors

colors = ["#543005", "#8c510a", "#bf812d", "#80cdc1", "#35978f", "#01665e", "#003c30"]



# iterate over very groupby group and 

# append their values as a list

# THIS IS A CRUCIAL STEP

for _, df_ in gb_df:

    lx.append(df_["displ"].values.tolist())

    ln.append(list(set(df_["class"].values.tolist()))[0])



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (8, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data



# hist returns a tuple of 3 values

# let's unpack it

n, bins, patches = ax.hist(lx, bins = 30, stacked = True, density = False, color = colors)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change x lim

ax.set_ylim(0, 25)

# set the xticks to reflect every third value

ax.set_xticks(bins[::3])



# set a title

ax.set_title("Stacked Histogram of displ colored by class")



# add a custom legend wit class and color

# you have to pass a dict

ax.legend({class_:color for class_, color in zip(ln, colors)})



# set the y label

ax.set_ylabel("Frequency");
# Useful for:

# This is one of the most fundamental plots to master

# It's shows the approximate distributin of numerical or categorical data.



# More info: 

# https://en.wikipedia.org/wiki/Histogram



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

gb_df = df[["class", "manufacturer"]].groupby("class")

lx = []

ln = []



# handpicked colors

colors = ["#543005", "#8c510a", "#bf812d", "#80cdc1", "#35978f", "#01665e", "#003c30"]



# iterate over very groupby group and 

# append their values as a list

# THIS IS A CRUCIAL STEP

for _, df_ in gb_df:

    lx.append(df_["manufacturer"].values.tolist())

    ln.append(list(set(df_["class"].values.tolist()))[0])



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (8, 8))

ax = fig.add_subplot()



# hist returns a tuple of 3 values

# let's unpack it

n, bins, patches = ax.hist(lx, bins = 30, stacked = True, density = False, color = colors)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# rotate the x axis label

ax.tick_params(axis = 'x', labelrotation = 90)



# add a custom legend wit class and color

# you have to pass a dict

ax.legend({class_:color for class_, color in zip(ln, colors)})



# add a title

ax.set_title("Stacked histogram of manufacturer colored by class")



# set an y label

ax.set_ylabel("Frequency");
# Useful for:

# A density plot is a representation of the distribution of a numeric variable. 

# It uses a kernel density estimate to show the probability density function of the variable



# More info: 

# https://www.data-to-viz.com/graph/density.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 8))



# ----------------------------------------------------------------------------------------------------

# plot the data

# the idea is to iterate over each class

# extract their data ad plot a sepate density plot

for cyl_ in df["cyl"].unique():

    # extract the data

    x = df[df["cyl"] == cyl_]["cty"]

    # plot the data using seaborn

    sns.kdeplot(x, shade=True, label = "{} cyl".format(cyl_))



# set the title of the plot

plt.title("Density Plot of City Mileage by n_cilinders");
# Useful for:

# A density plot is a representation of the distribution of a numeric variable. 

# It uses a kernel density estimate to show the probability density function of the variable

# This variation plots the histogram aswel



# More info: 

# https://www.data-to-viz.com/graph/density.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 8))



# ----------------------------------------------------------------------------------------------------

# plot the data

# the idea is to iterate over each class

# extract their data ad plot a sepate density plot

# with their histogram

for class_ in ["compact", "suv", "minivan"]:

    # extract the data

    x = df[df["class"] == class_]["cty"]

    # plot the data using seaborn

    sns.distplot(x, kde = True, label = "{} class".format(class_))

    

# set the title of the plot

plt.title("Density Plot of City Mileage by vehicle type");
# Useful for:

# Joyplot are one of the favorites. 

# Joyplots are essentially just a number of stacked overlapping density plots, that look like a mountain ridge, if done right.



# More info: 

# https://sbebo.github.io/posts/2017/08/01/joypy/

# http://sigmaquality.pl/data-plots/perfect-plots-joyplot-plot/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

plt.figure(figsize = (16,10), dpi = 80)

# plot the data using joypy

fig, axes = joypy.joyplot(df, 

                          column = ['hwy', 'cty'], # colums to be plotted.

                          by = "class", # separate the data by this value. Creates a separate distribution for each one.

                          ylim = 'own', 

                          figsize = (14,10)

                         )



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# add a title

plt.title('Joy Plot of City and Highway Mileage by Class', fontsize = 22);
# Useful for:

# Joyplot are one of the favorites. 

# Joyplots are essentially just a number of stacked overlapping density plots, that look like a mountain ridge, if done right.

# This is another example for the Titanic dataset



# More info: 

# https://sbebo.github.io/posts/2017/08/01/joypy/

# http://sigmaquality.pl/data-plots/perfect-plots-joyplot-plot/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/titanic/train.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# create 2 columns with the age of a person if their are male (MALE_AGE) or female (FEMALE_AGE)

df["MALE_AGE"] = df.apply(lambda row: row["Age"] if row["Sex"] == "male" else np.nan, axis = 1)

df["FEMALE_AGE"] = df.apply(lambda row: row["Age"] if row["Sex"] == "female" else np.nan, axis = 1)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

plt.figure(dpi = 380)

# plot the data using joypy

fig, axes = joypy.joyplot(df, 

                          column=['FEMALE_AGE', 'MALE_AGE'], # colums to be plotted.

                          by = "Pclass", # separate the data by this value. Creates a separate distribution for each one.

                          ylim = 'own', 

                          figsize = (12,8), 

                          legend = True, 

                          color = ['#f4cccc', '#0c343d'], 

                          alpha = 0.4

                         )



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# add a title with a specific color and fontisze

plt.title('Titanic disaster: age distribution of casualties by the class', fontsize = 26, color = 'darkred', alpha = .5)



# change the font of the text on the plot

plt.rc("font", size = 20)



# set and x label with a specific color and size

plt.xlabel('Age of passengers by PClass',  fontsize = 16, color = 'darkred', alpha = 1);
# Useful for:

# This plot is very cool if you want to show the distribution of some categorical values

# and mark some interesting value, like median, mean of max values with a specific color



# More info: 

# https://www.statisticshowto.com/what-is-a-dot-plot/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)

# sort the values

df.sort_values(["manufacturer", "cty"], inplace = True)

lc = []



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12, 6))

ax = fig.add_subplot()



# iterate over each car manufacturer

for i, car in enumerate(df["manufacturer"].unique()):

    # prepare the data for plotting

    # get x and y

    x = df[df["manufacturer"] == car]["cty"]

    y = [car for i_ in range(len(x))]

    

    # calculate the median value

    x_median = np.median(x)

    

    # plot the data

    ax.scatter(x, y, c = "white", edgecolor = "black", s = 30)

    ax.scatter(x_median, i, c = "red",  edgecolor = "black", s = 80)

    

    # add some horizontal line so we can easily track each manufacturer with their distribution

    ax.hlines(i, 0, 40, linewidth = .1)

    

    # append the car name 

    # we need this to change the y labels

    lc.append(car)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change x and y label

ax.set_xlim(5, 40)

ax.set_ylim(-2, 16)



# change the ylabel fontsize

ax.tick_params(axis = "y", labelsize = 12)



# set a title

ax.set_title("Distribution of City Mileage by maker", fontsize = 12)



# annotate some text that will be placed below the legend

ax.text(35, 5.5, "$red \; dots \; are \; the \: median$", fontdict={'size':8}, color='firebrick')



# create a custom legend

# a red circe for the median

red_patch = plt.plot([],[], marker = "o", ms = 10, ls = "", mec = None, color = 'firebrick', label = "Median")



# add the patch and render the legend

plt.legend(handles = red_patch, loc = 7, fontsize = 12)



# remove 3 spines to make a prettier plot

ax.spines["right"].set_color("None")

ax.spines["left"].set_color("None")

ax.spines["top"].set_color("None");
# Useful for:

# Boxplot is a fundamenta chart in statistics.

# It helps to show the distribution of categorical data through quartiles.

# It helps also to see the dispersion of a series, thanks to the whiskers



# More info: 

# https://en.wikipedia.org/wiki/Box_plot



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

plt.figure(figsize = (10, 10), dpi = 80)

# plot the data using seaborn

ax = sns.boxplot(x = "class", y = "hwy", data = df)





# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change the font of the x and y ticks (numbers on the axis)

ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)

ax.tick_params(axis = 'y', labelsize = 12)



# set and x and y label

ax.set_xlabel("Class", fontsize = 14)

ax.set_ylabel("HWY", fontsize = 14)



# set a title

ax.set_title("Boxplot", fontsize = 14);
# Useful for:

# Boxplot is a fundamenta chart in statistics.

# It helps to show the distribution of categorical data through quartiles.

# It helps also to see the dispersion of a series, thanks to the whiskers.

# This plot adds annotation for each box to add additional information to the plot.



# More info: 

# https://en.wikipedia.org/wiki/Box_plot



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting



# vectors to plot

vects = [df[df["class"] == car]["hwy"] for car in df["class"].unique()]



# labels for the x axis

labels = [class_ for class_ in df["class"].unique()]



# handpicked colors

colors = ["#543005", "#8c510a", "#bf812d", "#80cdc1", "#35978f", "#01665e", "#003c30"]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (16, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

plot1 = ax.boxplot(vects,

    notch = False, 

    vert = True,

    meanline = True,

    showmeans = True,

    patch_artist=True

)



# iterate over every box and add some annotations

for box, color, vect, label, tick in zip(plot1["boxes"], # using this line, you can iterate over every box

                                         colors, 

                                         vects, 

                                         labels, 

                                         ax.get_xticks()):

    # change the color of the box

    box.set(facecolor = color)

    # add text

    ax.annotate("{} obs".format(len(vect)), 

                xy = (tick, np.median(vect)),

               xytext = (15, 50),

               textcoords = "offset points",

               arrowprops = dict(arrowstyle = "->", connectionstyle = "arc3,rad=.2"),

               fontsize = 12)



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# change the x labels

ax.set_xticklabels(labels = labels)



# change the rotation and the size of the x ticks (numbers of x axis)

ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)



# set the title for the plot

ax.set_title("Box plor of Highway Mileage by Vehicle Class", fontsize = 16);
# Useful for:

# This plot is very cool, since it allows you to have on the same plot

# a box plot and a dot plot. 

# This way it allows you to have more information to analyze.

# Using seaborn we can also pass the hue to differentiate between classes.



# More info: 

# https://en.wikipedia.org/wiki/Box_plot

# https://en.wikipedia.org/wiki/Dot_plot_(statistics)



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

plt.figure(figsize = (10, 10), dpi= 80)

# plot the data using seaborn

# since we don't create a specific separete plot

# everything will be rendered on the same axes

sns.boxplot(x = "class", y = "hwy", data = df, hue = "cyl")

sns.stripplot(x = 'class', y = 'hwy', data = df, color = 'black', size = 3, jitter = 1)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# get the current figure

ax = plt.gca()

# get the xticks to iterate over

xticks = ax.get_xticks()



# iterate over every xtick and add a vertical line

# to separate different classes

for tick in xticks:

    ax.vlines(tick + 0.5, 0, np.max(df["hwy"]), color = "grey", alpha = .1)



# rotate the x and y ticks

ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)

ax.tick_params(axis = 'y', labelsize = 12)



# add x and y label

ax.set_xlabel("Class", fontsize = 14)

ax.set_ylabel("HWY", fontsize = 14)



# add a title and put the legend on a specific location

ax.set_title("Boxplot and stripplot on the same figure", fontsize = 14)

ax.legend(loc = "lower left", fontsize = 14);
# Useful for:

# Violin plot is another fundamental plot in statistics

# It helps you see the probability density of the data at different values.



# More info: 

# https://en.wikipedia.org/wiki/Violin_plot



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

plt.figure(figsize = (10, 10), dpi= 80)

sns.violinplot(x = "class", 

               y = "hwy", 

               data = df, 

               scale = 'width', 

               inner = 'quartile'

              )



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# get the current figure

ax = plt.gca()

# get the xticks to iterate over

xticks = ax.get_xticks()



# iterate over every xtick and add a vertical line

# to separate different classes

for tick in xticks:

    ax.vlines(tick + 0.5, 0, np.max(df["hwy"]), color = "grey", alpha = .1)

    

# rotate the x and y ticks

ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)

ax.tick_params(axis = 'y', labelsize = 12)



# add x and y label

ax.set_xlabel("Class", fontsize = 14)

ax.set_ylabel("HWY", fontsize = 14)



# set title

ax.set_title("Violinplot", fontsize = 14);
# Useful for:

# Violin plot is another fundamental plot in statistics

# It helps you see the probability density of the data at different values.



# More info: 

# https://en.wikipedia.org/wiki/Violin_plot



# ----------------------------------------------------------------------------------------------------

# get the data

tips = sns.load_dataset("tips")



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

plt.figure(figsize = (10, 10), dpi= 80)



# plot the data using seaborn

# the cool thing is that we put split = True and have 4 violin plots instead of 8

ax = sns.violinplot(x = "day", y = "total_bill", hue = "sex", split = True, data = tips)



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# set a title and change the legend location

ax.set_title('Distribution of total bill amount per day', fontsize = 16)

ax.legend(loc = "upper left", fontsize = 10);
# Useful for:

# The population chart is a type of funnel chart.

# It really helps out to see the gain/loss of certain amount at every stage in a process.



# More info: 

# https://en.wikipedia.org/wiki/Population_pyramid



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/email_campaign_funnel.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

gb_df = df.groupby(["Stage", "Gender"])["Users"].sum().to_frame().reset_index()

gb_df.set_index("Stage", inplace = True)



# separate the different groups to be plotted

x_male = gb_df[gb_df["Gender"] == "Male"]["Users"]

x_female = gb_df[gb_df["Gender"] == "Female"]["Users"]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 10))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

ax.barh(x_male.index, x_male, color = "red", alpha = 0.3, label = "Male pyramid")

ax.barh(x_female.index, x_female, color = "green", alpha = 0.3, label = "Female pyramid")



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# add the legend to a specific location

ax.legend(loc = "upper left", fontsize = 12)

# set xlabel

ax.set_xlabel("Users", fontsize = 12)

# set the title for the plot

ax.set_title("Population Pyramid", fontsize = 14)

# change the x and y ticks to a smaller size

ax.tick_params(axis = 'y', labelsize = 12)

ax.tick_params(axis = 'x', labelsize = 12)
# Useful for:

# This is a normal barplot (we show count of each classes)

# But seabron makes it really easy to plot this effortlessly



# More info: 

# https://seaborn.pydata.org/tutorial/categorical.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/titanic/train.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

# plot the data using seaborn

fig = plt.figure(figsize = (12, 6))

ax = sns.catplot("Survived", 

                 col = "Pclass", 

                 data = df, 

                 kind = "count",  

                 palette = 'tab20',  

                 aspect = .8

                 )
# Useful for:

# This is a normal barplot (we show count of each classes)

# But seabron makes it really easy to plot this effortlessly

# In this plot we add an additional filter to separate even more the data



# More info: 

# https://seaborn.pydata.org/tutorial/categorical.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/titanic/train.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

# plot the data using seaborn

fig = plt.figure(figsize = (12, 6))

ax = sns.catplot(x = "Age",

                 y = "Embarked",

                 col = "Pclass",

                 hue = "Sex",

                 data = df, 

                 kind = "violin",  

                 palette = 'tab20',  

                 aspect = .8

                 )
# Useful for:

# Waffle charts are very useful to show the composition of a certain column

# of different categories



# More info: 

# https://datavizproject.com/data-type/percentage-grid/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# create a dictionary of each class and their totals

values = {k:v for k,v in zip(df["class"].value_counts().index, df["class"].value_counts().values)}



# ----------------------------------------------------------------------------------------------------

# plot the data using pywaffle

plt.figure(

    FigureClass = Waffle,

    rows = 7,

    columns = 34,

    values = values,

    legend = {'loc': 'upper left', 'bbox_to_anchor': (1, 1), "fontsize": "12"},

    figsize = (20, 7)

)



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# set a title

plt.title("Waffle chart using pywaffle", fontsize = 12); # actually the semicolon helps here to avoid Out
# Useful for:

# Waffle charts are very useful to show the composition of a certain column

# of different categories

# This plot tries to replicate a waffle chart using only matplotlib



# More info: 

# https://datavizproject.com/data-type/percentage-grid/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# we need this when we will create the axes

# and the colors for each column

rows = 10

columns = 7



ncats = len(df["class"].value_counts().index)

colors = [plt.cm.inferno_r(i/float(columns)) for i in range(columns)]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 5))

# create 70 smaller axes to change their color in the next loop

axes = fig.subplots(nrows = rows,

                    ncols = columns)



# ----------------------------------------------------------------------------------------------------

# plot the data

# iterate over all rows and columns

# check the basic part of this kernel if you need help

for col in range(columns):

    for row in range(rows):

        # get every axes we created

        ax = axes[row, col]

        # get the corresponding color

        color = colors[col]

        # change the background color of each axes

        ax.set_facecolor(color)

        # get rid of the x and y ticks (no numbers on x and y axis)

        ax.set_xticks([])

        ax.set_yticks([])



# add a title to the FIGURE

# Note: that matplotlib always plots on the last axes

# if we do it by ax.set_title, we will add a title to the 70'th axes

# and we don't want that

fig.suptitle("Waffle chart using raw matplotlib", fontsize = 14)



# create a legend for each category

legend_elements = [Patch(facecolor = color, 

                         edgecolor = 'white', 

                         label = str(i)) for i, color in enumerate(colors)]



# add the lgend and the patch to the figure

fig.legend(handles = legend_elements, loc = 'lower left', bbox_to_anchor = (0.0, 1.01), ncol = 2, borderaxespad = 0, frameon = False);
# Useful for:

# A pie chart (or a circle chart) is a circular statistical graphic, which is divided into slices to illustrate numerical proportion



# More info: 

# https://en.wikipedia.org/wiki/Pie_chart



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# create a dictionary of classes and their totals

d = df["class"].value_counts().to_dict()



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (18, 6))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

ax.pie(d.values(), # pass the values from our dictionary

       labels = d.keys(), # pass the labels from our dictonary

       autopct = '%1.1f%%', # specify the format to be plotted

       textprops = {'fontsize': 10, 'color' : "white"} # change the font size and the color of the numbers inside the pie

      )



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# set the title

ax.set_title("Pie chart")



# set the legend and add a title to the legend

ax.legend(loc = "upper left", bbox_to_anchor = (1, 0, 0.5, 1), fontsize = 10, title = "Vehicle Class");
# Useful for:

# A pie chart (or a circle chart) is a circular statistical graphic, which is divided into slices to illustrate numerical proportion

# Nested pie chart goes one step further and separate every outer level of the pie chart

# with the composition on a lower level



# More info: 

# https://en.wikipedia.org/wiki/Pie_chart



# ----------------------------------------------------------------------------------------------------

# get the data

size = 0.3

vals = np.array([[60., 32.], [37., 40.], [29., 10.]])



# create the outer and inner colors

cmap = plt.get_cmap("tab20c")

outer_colors = cmap(np.arange(3)*4)

inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig, ax = plt.subplots()



# ----------------------------------------------------------------------------------------------------

# plot the data

# outer level

ax.pie(vals.sum(axis = 1), # plot the total [60., 32.] = 92

       radius = 1, 

       colors = outer_colors,

       wedgeprops = dict(width = size, edgecolor = 'w'))



# inner level

ax.pie(vals.flatten(), # using flatten we plot 60, 32 separetly

       radius = 1 - size, 

       colors = inner_colors,

       wedgeprops = dict(width = size, edgecolor = 'w'))



# set the title for the plot

ax.set(aspect = "equal", title = 'Nested pie chart');
# Useful for:

# Treemap is very cool and can be used mane different contexts

# usually we want to show the composition of some totals by groups

# very often, smaller groups tend to be very small squares



# More info: 

# https://en.wikipedia.org/wiki/Treemapping



# ----------------------------------------------------------------------------------------------------

# get the data



PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# get the values

label_value = df["class"].value_counts().to_dict()



# create the labels using a list comprehesion

labels = ["{} has {} obs".format(class_, obs) for class_, obs in label_value.items()]



# create n colors based on the number of labels we have

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

plt.figure(figsize = (20, 10))



# ----------------------------------------------------------------------------------------------------

# plot the data using squarify

squarify.plot(sizes = label_value.values(), label = labels,  color = colors, alpha = 0.8)



# ----------------------------------------------------------------------------------------------------

# prettify the plot

# add a title to the plot

plt.title("Treemap using external libraries");
# Useful for:

# A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent



# More info: 

# https://en.wikipedia.org/wiki/Bar_chart



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mpg_ggplot2.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# get a dictionary with x and y from a dictionary

d = df["manufacturer"].value_counts().to_dict()



# create n colors based on the number of labels we have

colors = [plt.cm.Spectral(i/float(len(d.keys()))) for i in range(len(d.keys()))]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

ax.bar(d.keys(), d.values(), color = colors)



# iterate over every x and y and annotate the value on the top of the barchart

for i, (k, v) in enumerate(d.items()):

    ax.text(k, # where to put the text on the x coordinates

            v + 1, # where to put the text on the y coordinates

            v, # value to text

            color = colors[i], # color corresponding to the bar

            fontsize = 10, # fontsize

            horizontalalignment = 'center', # center the text to be more pleasant

            verticalalignment = 'center'

           )



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change the size of the x and y ticks

ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)

ax.tick_params(axis = 'y', labelsize = 12)



# change the ylim

ax.set_ylim(0, 45)



# set a title for the plot

ax.set_title("Number of Vehicles per Manufacturer", fontsize = 14);
# Useful for:

# Timeseries is a special type of plots where the time component is present.



# More info: 

# https://study.com/academy/lesson/time-series-plots-definition-features.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/AirPassengers.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# helper function to transform and work with the time column

def create_date_tick(df):

    '''

    Converts dates from this format: Timestamp('1949-01-01 00:00:00')

    To this format: 'Jan-1949'

    '''

    df["date"] = pd.to_datetime(df["date"]) # convert to datetime

    df["month_name"] = df["date"].dt.month_name() # extracts month_name

    df["month_name"] = df["month_name"].apply(lambda x: x[:3]) # passes from January to Jan

    df["year"] = df["date"].dt.year # extracts year

    df["new_date"] = df["month_name"].astype(str) + "-" + df["year"].astype(str) # Concatenaes Jan and year --> Jan-1949



# create the time column and the xtickslabels column

create_date_tick(df)



# get the y values (the x is the index of the series)

y = df["value"]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

ax.plot(y, color = "red", alpha = .5, label = "Air traffic")



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# set the gridlines

ax.grid(alpha = .3)



# change the ylim

ax.set_ylim(0, 700)



# get the xticks and the xticks labels

xtick_location = df.index.tolist()[::6]

xtick_labels = df["new_date"].tolist()[::6]



# set the xticks to be every 6'th entry

# every 6 months

ax.set_xticks(xtick_location)



# chage the label from '1949-01-01 00:00:00' to this 'Jan-1949'

ax.set_xticklabels(xtick_labels, rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})



# change the size of the font of the x and y axis

ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)

ax.tick_params(axis = 'y', labelsize = 12)



# set the title and the legend of the plot

ax.set_title("Air Passsengers Traffic (1949 - 1969)", fontsize = 12)

ax.legend(loc = "upper left", fontsize = 10);
# Useful for:

# Timeseries is a special type of plots where the time component is present.

# This plot is a continuation of the previous one.

# Here we use scatter markers and text to annotate relevant events.

# In our case, the local maxima and minima



# More info: 

# https://study.com/academy/lesson/time-series-plots-definition-features.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/AirPassengers.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# helper function to transform and work with the time column

def create_date_tick(df):

    '''

    Converts dates from this format: Timestamp('1949-01-01 00:00:00')

    To this format: 'Jan-1949'

    '''

    df["date"] = pd.to_datetime(df["date"]) # convert to datetime

    df["month_name"] = df["date"].dt.month_name() # extracts month_name

    df["month_name"] = df["month_name"].apply(lambda x: x[:3]) # passes from January to Jan

    df["year"] = df["date"].dt.year # extracts year

    df["new_date"] = df["month_name"].astype(str) + "-" + df["year"].astype(str) # Concatenaes Jan and year --> Jan-1949



# create the time column and the xtickslabels column

create_date_tick(df)



# get the y values (the x is the index of the series)

y = df["value"]



# find local maximum INDEX using scipy library

max_peaks_index, _ = find_peaks(y, height=0) 



# find local minimum INDEX using numpy library

doublediff2 = np.diff(np.sign(np.diff(-1*y))) 

min_peaks_index = np.where(doublediff2 == -2)[0] + 1



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12, 8))

ax = fig.add_subplot()



# plot the data using matplotlib

ax.plot(y, color = "blue", alpha = .5, label = "Air traffic")



# we have the index of max and min, so we must index the values in order to plot them

ax.scatter(x = y[max_peaks_index].index, y = y[max_peaks_index].values, marker = "^", s = 90, color = "green", alpha = .5, label = "Peaks")

ax.scatter(x = y[min_peaks_index].index, y = y[min_peaks_index].values, marker = "v", s = 90, color = "red", alpha = .5, label = "Troughs")



# iterate over some max and min in order to annotate the values

for max_annot, min_annot in zip(max_peaks_index[::3], min_peaks_index[1::5]):

    

    # extract the date to be plotted for max and min

    max_text = df.iloc[max_annot]["new_date"]

    min_text = df.iloc[min_annot]["new_date"]

    

    # add the text

    ax.text(df.index[max_annot], y[max_annot] + 50, s = max_text, fontsize = 8, horizontalalignment = 'center', verticalalignment = 'center')

    ax.text(df.index[min_annot], y[min_annot] - 50, s = min_text, fontsize = 8, horizontalalignment = 'center', verticalalignment = 'center')



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change the ylim

ax.set_ylim(0, 700)



# get the xticks and the xticks labels

xtick_location = df.index.tolist()[::6]

xtick_labels = df["new_date"].tolist()[::6]



# set the xticks to be every 6'th entry

# every 6 months

ax.set_xticks(xtick_location)



# chage the label from '1949-01-01 00:00:00' to this 'Jan-1949'

ax.set_xticklabels(xtick_labels, rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})



# change the size of the font of the x and y axis

ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)

ax.tick_params(axis = 'y', labelsize = 12)



# set the title and the legend of the plot

ax.set_title("Air Passsengers Traffic (1949 - 1969)", fontsize = 12)

ax.legend(loc = "upper left", fontsize = 10);
# Useful for:

# This plot are fundamental in timeseries analysis.

# Basically here we compare the a series again itself but with some lags.

# These are plots that graphically summarize the strength of a relationship with an observation in a time series with observations at prior time steps.



# More info: 

# https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/AirPassengers.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)



# ----------------------------------------------------------------------------------------------------

# plot the data using the built in plots from the stats module

plot_acf(df["value"], ax = ax1, lags = 50)

plot_pacf(df["value"], ax = ax2, lags = 15);
# Useful for:

# The cross correlation plot compares two series to see if there have a correlation.

# Remmember correlation not casuality



# More info: 

# https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9?gi=abf39ccba21b



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mortality.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# using this solution to calculate the cross correlation of 2 series

# https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas

def crosscorr(datax, datay, lag=0):

    """ 

    Lag-N cross correlation. 

    Parameters

    ----------

    lag : int, default 0

    datax, datay : pandas.Series objects of equal length



    Returns

    ----------

    crosscorr : float

    """

    return datax.corr(datay.shift(lag))



# get the cross correlation

xcov_monthly = [crosscorr(df["mdeaths"], df["fdeaths"], lag = i) for i in range(70)]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (8, 6))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

# notice that this is a regular barchart but the width is really small

ax.bar(x = np.arange(len(xcov_monthly)), height = xcov_monthly, width = .3)



# add some vertical lines that represent the significance level

# you have to calculate them apart

ax.hlines(0.25, 0, len(xcov_monthly), alpha = .3)

ax.hlines(0, 0, len(xcov_monthly), alpha = .3)

ax.hlines(-0.25, 0, len(xcov_monthly), alpha = .3)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# set the title

ax.set_title("Cross Correlation Plot. mdeaths vs fdeasths", fontsize = 14)



# change the x and y ticks size

ax.tick_params(axis = 'x', labelsize = 10)

ax.tick_params(axis = 'y', labelsize = 10);
# Useful for:

# The cross correlation plot compares two series to see if there have a correlation.

# Remmember correlation not casuality.

# This solution shows how to calculate the significance level



# More info: 

# https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9?gi=abf39ccba21b



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mortality.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting



# separte x and y

x = df['mdeaths']

y = df['fdeaths']



# Compute Cross Correlations

ccs = stattools.ccf(x, y)[:100]

nlags = len(ccs)



# Compute the Significance level

# ref: https://stats.stackexchange.com/questions/3115/cross-correlation-significance-in-r/3128#3128

conf_level = 2 / np.sqrt(nlags)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (8,6), dpi = 80)

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

# notice that this is a regular barchart but the width is really small

ax.bar(x = np.arange(len(ccs)), height = ccs, width = .3)



# add some vertical lines that represent the significance level

# you have to calculate them apart

# we calculated it before

ax.hlines(0, xmin = 0, xmax = 100, color = 'gray')  # 0 axis

ax.hlines(conf_level, xmin = 0, xmax = 100, color='gray')

ax.hlines(-conf_level, xmin = 0, xmax = 100, color='gray')



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# set the title

ax.set_title('$Cross\; Correlation\; Plot:\; mdeaths\; vs\; fdeaths$', fontsize = 14)



# change the xlim of the plot

ax.set_xlim(0,len(ccs))



# change the x and y ticks size

ax.tick_params(axis = 'x', labelsize = 10)

ax.tick_params(axis = 'y', labelsize = 10);
# Useful for:

# The theory behind timeseries, says that a series can be decomposed into 3 parts

# The trend

# The seasonal part

# And the residual

# This plots shows how to do this



# More info: 

# https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = '/kaggle/input/the-50-plot-challenge/AirPassengers.csv'

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# helper function to transform and work with the time column

def create_date_tick(df):

    '''

    Converts dates from this format: Timestamp('1949-01-01 00:00:00')

    To this format: 'Jan-1949'

    '''

    df["date"] = pd.to_datetime(df["date"]) # convert to datetime

    df.set_index("date", inplace = True)

    df["date"] = df.index

    df["month_name"] = df["date"].dt.month_name() # extracts month_name

    df["month_name"] = df["month_name"].apply(lambda x: x[:3]) # passes from January to Jan

    df["year"] = df["date"].dt.year # extracts year

    df["new_date"] = df["month_name"].astype(str) + "-" +df["year"].astype(str) # Concatenaes Jan and year --> Jan-1949



# create the time column and the xtickslabels column    

create_date_tick(df)



# decompose the series using stats module

# results in this case is a special class 

# whose attributes we can acess

result = seasonal_decompose(df["value"])



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

# make the subplots share teh x axis

fig, axes = plt.subplots(ncols = 1, nrows = 4, sharex = True, figsize = (12,10))



# ----------------------------------------------------------------------------------------------------

# plot the data

# using this cool thread:

# https://stackoverflow.com/questions/45184055/how-to-plot-multiple-seasonal-decompose-plots-in-one-figure

# This allows us to have more control over the plots



# plot the original data

result.observed.plot(ax = axes[0], legend = False)

axes[0].set_ylabel('Observed')

axes[0].set_title("Decomposition of a series")



# plot the trend

result.trend.plot(ax = axes[1], legend = False)

axes[1].set_ylabel('Trend')



# plot the seasonal part

result.seasonal.plot(ax = axes[2], legend = False)

axes[2].set_ylabel('Seasonal')



# plot the residual

result.resid.plot(ax = axes[3], legend = False)

axes[3].set_ylabel('Residual')



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# get the xticks and the xticks labels

xtick_location = df.index.tolist()[::6]

xtick_labels = df["new_date"].tolist()[::6]



# set the xticks to be every 6'th entry

# every 6 months

ax.set_xticks(xtick_location)



# chage the label from '1949-01-01 00:00:00' to this 'Jan-1949'

ax.set_xticklabels(xtick_labels, rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'});
# Useful for:

# Multiple timeseries is a special case when we plot 2 series and see their performance over time



# More info: 

# https://study.com/academy/lesson/time-series-plots-definition-features.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mortality.csv"

df = pd.read_csv(PATH)



# set the date column to be the index

df.set_index("date", inplace = True)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 5))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

ax.plot(df["mdeaths"], color = "red", alpha = .5, label = "mdeaths")

ax.plot(df["fdeaths"], color = "blue", alpha = .5, label = "fdeaths")



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# get the xticks and the xticks labels

xtick_location = df.index.tolist()[::6]

xtick_labels = df.index.tolist()[::6]



# set the xticks to be every 6'th entry

# every 6 months

ax.set_xticks(xtick_location)

ax.set_xticklabels(xtick_labels, rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'});



# change the x and y ticks to be smaller

ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)

ax.tick_params(axis = 'y', labelsize = 12)



# add legend, a title and grid to make it look nicer

ax.legend(loc = "upper left", fontsize = 10)

ax.set_title("Mdeaths and fdeaths over time", fontsize = 14)

ax.grid(axis = "y", alpha = .3)
# Useful for:

# Multiple timeseries is a special case when we plot 2 series and see their performance over time

# However, here since the data is on a different scale, we will add a secondary y axis



# More info: 

# https://study.com/academy/lesson/time-series-plots-definition-features.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/economics.csv"

df = pd.read_csv(PATH)



# set the date column to be the index

df.set_index("date", inplace = True)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# get the x and y values

x_1 = df["psavert"]

x_2 = df["unemploy"]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (14, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

# here is the main axis

ax.plot(x_1, color = "red", alpha = .3, label = "Personal savings rate")



# suing twinx we can create a secondary axis

ax2 = ax.twinx()

# plot the data on the secondary axis

ax2.plot(x_2, color = "blue", alpha = .3, label = "Unemployment rate")



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# get the xticks and the xticks labels (every 12 entry)

xtick_location = df.index.tolist()[::12]

xtick_labels = df.index.tolist()[::12]



# set the xticks to be every 12'th entry

# every 12 months

ax.set_xticks(xtick_location)

ax.set_xticklabels(xtick_labels, rotation = 90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'});



# change the x and y ticks to be smaller for the main axis and for the secondary axis

ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)

ax.tick_params(axis = 'y', labelsize = 12)

ax2.tick_params(axis = 'y', labelsize = 12)



# set a title and a grid

ax.set_title("Personal savings rate vs Unemployed rate: 2 axis", fontsize = 16)

ax.grid(axis = "y", alpha = .3)
# Useful for:

# This is a regular timeseries plot, but we add some confidence level/bands to the main series

# We can add +- 5% values or we can compute the errors and add error bands



# More info: 

# https://study.com/academy/lesson/time-series-plots-definition-features.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/user_orders_hourofday.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# create a groupby df calculating the mean for each group

gb_df = df.groupby(["order_hour_of_day"])["quantity"].mean().to_frame()



# separete x and calculate the upper and lower bands

x = gb_df["quantity"]

x_lower = x*0.95

x_upper = x*1.05



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

ax.plot(x, color = "white", lw = 3)

ax.plot(x_lower, color = "#bcbddc")

ax.plot(x_upper, color = "#bcbddc")



# fill the area between the 3 lines

ax.fill_between(x.index, x, x_lower, where = x > x_lower, facecolor='#bcbddc', interpolate = True)

ax.fill_between(x.index, x, x_upper, where = x_upper > x, facecolor='#bcbddc', interpolate = True)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change the ylim

ax.set_ylim(0, 25)



# set the x and y labels

ax.set_xlabel("Hour of day")

ax.set_ylabel("# Orders")



# get the xticks and the xticks labels

xtick_location = gb_df.index.tolist()[::2]

xtick_labels = gb_df.index.tolist()[::2]



# set the xticks to be every 2'th entry

# every 2 months

ax.set_xticks(xtick_location)

ax.set_xticklabels(xtick_labels, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline', "fontsize":"12"})



# change the x and y tick size

ax.tick_params(axis = 'x', labelsize = 12)

ax.tick_params(axis = 'y', labelsize = 12)



# add a title and a gridline

ax.set_title("Mean orders +- 5% interval ", fontsize = 16)

ax.grid(axis = "y", alpha = .3)
# Useful for:

# This is a regular timeseries plot, but we add some confidence level/bands to the main series

# We can add +- 5% values or we can compute the errors and add error bands

# This is a similar plot but for future sales competition dataset



# More info: 

# https://study.com/academy/lesson/time-series-plots-definition-features.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv"



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# a helper function that manipulates the df and returns weekly sales

# for shop 31

def create_df_future_sales(PATH):

    '''

    Creates the df from the future sales competition.

    Only for shop 31 and for educational purpose

    '''

    df = pd.read_csv(PATH)

    df = df[df["shop_id"] == 31][["item_cnt_day", "date"]]

    df["date"] = pd.to_datetime(df["date"])

    df.set_index("date", inplace = True)

    df = df.resample("W")["item_cnt_day"].sum().to_frame()

    

    return df



# get the needed df

df = create_df_future_sales(PATH)



# separete x and calculate the upper and lower bands

x = df["item_cnt_day"]

x_lower = x*0.95

x_upper = x*1.05



# ----------------------------------------------------------------------------------------------------

# instanciate the figure



fig = plt.figure(figsize = (12, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data using matplotlib

ax.plot(x, color = "black", lw = 3)

ax.plot(x_lower, color = "#bcbddc")

ax.plot(x_upper, color = "#bcbddc")



# fill the area between the 3 lines

ax.fill_between(x.index, x, x_lower, where = x > x_lower, facecolor='#bcbddc', interpolate = True)

ax.fill_between(x.index, x, x_upper, where = x_upper > x, facecolor='#bcbddc', interpolate = True)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# get the xticks and the xticks labels

xtick_location = df.index.tolist()[::4]

xtick_labels = df.index.tolist()[::4]



# set the xticks to be every 4'th entry

# every 4 week

ax.set_xticks(xtick_location)

ax.set_xticklabels(xtick_labels, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline', "fontsize":"12"})



# change the size of the x and y ticks

ax.tick_params(axis = 'x', labelsize = 12, rotation = 90)

ax.tick_params(axis = 'y', labelsize = 12)



# set the x and y label

ax.set_xlabel("Week of the year")

ax.set_ylabel("Mean sales")



# set the title and add a grid

ax.set_title("Mean sales +- 5% interval for shop 31", fontsize = 16)

ax.grid(axis = "y", alpha = .3)
# Useful for:

# This is a regular timeseries plot, but we add some confidence level/bands to the main series

# We can add +- 5% values or we can compute the errors and add error bands

# In this plot we will calculate the error bands using stats module



# More info: 

# https://study.com/academy/lesson/time-series-plots-definition-features.html



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/orders_45d.csv"

df_raw = pd.read_csv(PATH, parse_dates = ['purchase_time', 'purchase_date'])



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# Prepare Data: Daily Mean and SE Bands

df_mean = df_raw.groupby('purchase_date').quantity.mean()

df_se = df_raw.groupby('purchase_date').quantity.apply(sem).mul(1.96)



# prepare the xticks in a specific format

x = [d.date().strftime('%Y-%m-%d') for d in df_mean.index]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12,6), dpi = 80)

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

ax.plot(x, df_mean, color = "white", lw = 2)

ax.fill_between(x, df_mean - df_se, df_mean + df_se, color = "#3F5D7D") 



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change the spines to make it look cleaner

ax.spines["top"].set_alpha(0)

ax.spines["bottom"].set_alpha(1)

ax.spines["right"].set_alpha(0)

ax.spines["left"].set_alpha(1)



# get the xticks and the xticks labels

xtick_location = x[::6]

xtick_labels = [str(d) for d in x[::6]]



# set the xticks to be every 4'th entry

# every 4 week

ax.set_xticks(xtick_location)

ax.set_xticklabels(xtick_labels, fontdict = {'horizontalalignment': 'center', 'verticalalignment': 'center_baseline', "fontsize":"12"})



# add a title

ax.set_title("Daily Order Quantity of Brazilian Retail with Error Bands (95% confidence)", fontsize = 14)



# change the x and y limit

s, e = ax.get_xlim()

ax.set_xlim(s, e-2)

ax.set_ylim(4, 10)



# set the y label for the plot

ax.set_ylabel("# Daily Orders", fontsize = 12) 



# change the size of the x and y ticks

ax.tick_params(axis = 'x', labelsize = 12, rotation = 90)

ax.tick_params(axis = 'y', labelsize = 12)



# add some horizontal lines to make the plot look nicer

for y in range(5, 10, 1):    

    ax.hlines(y, xmin = s, xmax = e, colors = 'black', alpha = 0.5, linestyles = "--", lw = 0.5)
# Useful for:

# A stacked area chart is the extension of a basic area chart to display the evolution of the value of several groups on the same graphic. 

# The values of each group are displayed on top of each other. It allows to check on the same figure the evolution of both the total of a numeric variable, and the importance of each group. 

# If only the relative importance of each group interests you, you should probably draw a percent stacked area chart. 

# Note that this chart becomes hard to read if too many groups are displayed and if the patterns are really different between groups. In this case, think about using faceting instead.



# More info: 

# https://python-graph-gallery.com/stacked-area-plot/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/nightvisitors.csv"

df = pd.read_csv(PATH)

# set the data as index of the df

df.set_index("yearmon", inplace = True)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting



# get the x and y 

x = df.index

y = [df[col].values for col in df.columns]



# get the name of each group for the labels

labels = df.columns



# prepare some colors for each group to be ploted

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (14, 10))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

ax.stackplot(x,y, labels = labels, colors = colors)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# get the xticks and the xticks labels

xtick_location = df.index.tolist()[::3]

xtick_labels = df.index.tolist()[::3]



# set the xticks to be every 3'th entry

# every 3 entry

ax.set_xticks(xtick_location)

ax.set_xticklabels(xtick_labels, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline', "fontsize":"12"})



# change the x and y ticks to smaller size

ax.tick_params(axis = 'x', labelsize = 10, rotation = 90)

ax.tick_params(axis = 'y', labelsize = 10)



# set the x and y label

ax.set_xlabel("Date", fontsize = 12)

ax.set_ylabel("Visitors", fontsize = 12)



# change the ylim

ax.set_ylim(0, 90000)



# set a title and a legend

ax.set_title("Night visitors in Australian Regions", fontsize = 16)

ax.legend(fontsize = 8);
# Useful for:

# An area chart is really similar to a line chart, except that the area between the x axis and the line is filled in with color or shading.

# This draws the attention to the specific area.



# More info: 

# http://python-graph-gallery.com/area-plot/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/economics.csv"

df = pd.read_csv(PATH)



# set the date as the index of the df

df.set_index("date", inplace = True)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting



# get x and y

x = df["psavert"]

y = df["uempmed"]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (14, 8))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

ax.plot(x, color = "blue", alpha = .3, label = "Personal savings rate")

ax.plot(y, color = "red", alpha = .3, label = "Unemployment rate")



# fill the areas between the plots and the x axis

# this can create overlapping areas between lines

ax.fill_between(x.index, 0, x, color = "blue", alpha = .2)

ax.fill_between(x.index, 0, y, color = "red", alpha = .2)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# set the title

ax.set_title("Personal savings rate vs Unemployed rate", fontsize = 16)



# get the xticks and the xticks labels

xtick_location = df.index.tolist()[::12]

xtick_labels = df.index.tolist()[::12]



# set the xticks to be every 3'th entry

# every 3 entry

ax.set_xticks(xtick_location)

ax.set_xticklabels(xtick_labels, rotation = 90, fontdict = {'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})



# change the x and y ticks to smaller size

ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)

ax.tick_params(axis = 'y', labelsize = 12)



# more the spines of the axes

ax.spines["right"].set_color("None")

ax.spines["top"].set_color("None")



# set a legend and the y grid for the plot

ax.legend(fontsize = 10)

ax.grid(axis = "y", alpha = .3);
# Useful for:

# This is a very common plot you see everytime you connect to GitHub or Kaggle.

# Display the activity of a person for a certain period of time (usually a year)

# With a calendarmap



# More info: 

# https://pythonhosted.org/calmap/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/yahoo.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

df["date"] = pd.to_datetime(df["date"])

# the data must be a series with a datetime index

df.set_index("date", inplace = True)

x = df[df["year"] == 2014]["VIX.Close"]



# ----------------------------------------------------------------------------------------------------

# plot the data using calmap

calmap.calendarplot(x, fig_kws={'figsize': (16,10)}, yearlabel_kws={'color':'black', 'fontsize':14}, subplot_kws={'title':'Yahoo Stock Prices'});
# Useful for:

# Seasonal plots are a regular lineplot but where we represent a lot of data/seasons.

# If the data is increasing year after year, we can see the evolution of the variable very nicely

# in a smaller plot



# More info: 

# https://python-graph-gallery.com/line-chart/



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/AirPassengers.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting



# first of all. 

# We have the following format of our df

#         date	value

# 0	1949-01-01	  112

# 1	1949-02-01	  118

# 2	1949-03-01	  132

# 3	1949-04-01	  129

# 4	1949-05-01	  121



# Basically a lot of rows with each year month data

# In order to plot the data into a seasonal chart, we need the data in this format

# where each column is the year

# index_ 1949	1950	1951	1952	1953	1954	1955	1956	1957	1958	1959	1960

# 1	      112	 115	 145	 171	 196	 204	 242	 284	 315	 340	 360	 417

# 2	      118	 126	 150	 180	 196	 188	 233	 277	 301	 318	 342	 391



# To do so, we must create a repeating index (12  months) for each year



# create a repeating index of [1, 2, 3, .. 12] months x 12 times (12 years)

index_ = [i for i in range(1, 13)]*12



# set the index into the dataframe

df["index_"] = index_



# create a dictionary with the months name (we will use this later to change the x axis)

months_ = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

d = {k:v for k,v in zip(index_[:12], months_)}



# convert to datetime the date column

df["date"] = pd.to_datetime(df["date"])



# extract the year using pandas datatime (dt)

df["year"] = df["date"].dt.year



# drop the date

df.drop("date", inplace = True, axis = 1)



# create a pivot table

# traspose the rows into columns, where the columns name are the year to plot

df = df.pivot(values = "value", columns = "year", index = "index_")



# create n colors for each season

colors = [plt.cm.gist_earth(i/float(len(df.columns))) for i in range(len(df.columns))]



# get the x to plot

# since we are extracting it from our new df

# it has 12 values, one for each month

x = df.index



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12, 6))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

# iterate over every column in the dataframe and plot the data

for col, color in zip(df.columns, colors):

    # get the y to plot

    y = df[col]

    

    # plot the data using seaborn

    ax.plot(x, y, label = col, c = color)

    

    # get the x and y to annotate

    x_annotate = x[-1]

    y_annotate = df.iloc[11][col]

    

    # annotate at the end of each line some values

    ax.text(x_annotate + 0.3, y_annotate, col, fontsize = 8, c = color)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# set the x and y label

ax.set_xlabel("Months", fontsize = 13)

ax.set_ylabel("Air traffic", fontsize = 13)



# extract the x ticks location

xtick_location = df.index.tolist()



# using our dictionary, create a list of new xlabels

# basically instead of numbers, strings of months

months = [d[tick] for tick in xtick_location]



# change the x ticks with our new x ticks labels

ax.set_xticks(xtick_location)

ax.set_xticklabels(months, rotation = 90, fontdict = {'horizontalalignment': 'center', 'verticalalignment': 'center_baseline', "fontsize":"12"})



# change the y ticks font size

ax.tick_params(axis = 'y', labelsize = 12)



# change the y limit to make the plot a little bigger

ax.set_ylim(0, 700)



# get rid of spines from our plot

ax.spines["right"].set_color("None")

ax.spines["top"].set_color("None")



# add a grid to the plot

ax.grid(axis = "y", alpha = .3)



# set the title for the plot

ax.set_title("Monthly seasonal plot of air traffic (1949 - 1969)", fontsize = 15);
# Useful for:

# A dendrogram is a diagram representing a tree.

# It's very useful to represent hierarchy in a dataset



# More info: 

# https://en.wikipedia.org/wiki/Dendrogram



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/USArrests.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (10, 7))



# ----------------------------------------------------------------------------------------------------

# plot the data using the scipy package

dend = shc.dendrogram(shc.linkage(df[['Murder', 'Assault', 'UrbanPop', 'Rape']], method = 'ward'), 

                      labels = df["State"].values, 

                      color_threshold = 100)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# using plt.gca() we get the current figure

ax = plt.gca()



# set x and y label

ax.set_xlabel("County level")

ax.set_ylabel("# of incidents")



# change the x and y ticks size

ax.tick_params("x", labelsize = 10)

ax.tick_params("y", labelsize = 10)



# set a title

ax.set_title("US Arrests dendograms");
# Useful for:

# A cluster plots, help encircle data from a specific cluster, to help separte it more easily

# Before drawing the plot, you must first cluster the data into similar groups.



# More info: 

# https://en.wikipedia.org/wiki/Cluster_analysis



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/USArrests.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting



# get the x and y to plot

x = df["Murder"]

y = df["Assault"]



# first, create out cluster using the AgglomerativeClustering from sklearn

cluster = AgglomerativeClustering(n_clusters = 5, # notice that we specify the number of "optimal" clusters

                                  affinity = 'euclidean', # use the euclidean distance to compute similarity. The closer the better.

                                  linkage = 'ward'

                                 )  



# fit and predict the clusters based on this data

cluster.fit_predict(df[['Murder', 'Assault', 'UrbanPop', 'Rape']])  





# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (12, 10))

ax = fig.add_subplot()



# ----------------------------------------------------------------------------------------------------

# plot the data

ax.scatter(x, y)



# Encircle

def encircle(x,y, ax = None, **kw):

    '''

    Takes an axes and the x and y and draws a polygon on the axes.

    This code separates the differents clusters

    '''

    # get the axis if not passed

    if not ax: ax=plt.gca()

    

    # concatenate the x and y arrays

    p = np.c_[x,y]

    

    # to calculate the limits of the polygon

    hull = ConvexHull(p)

    

    # create a polygon from the hull vertices

    poly = plt.Polygon(p[hull.vertices,:], **kw)

    

    # add the patch to the axes

    ax.add_patch(poly)



# use our cluster fitted before to draw the clusters borders like we did at the beginning of the kernel

# basically go over each cluster and add a patch to the axes

encircle(df.loc[cluster.labels_ == 0, 'Murder'], df.loc[cluster.labels_ == 0, 'Assault'], ec = "k", fc = "gold", alpha = 0.2, linewidth = 0)

encircle(df.loc[cluster.labels_ == 1, 'Murder'], df.loc[cluster.labels_ == 1, 'Assault'], ec = "k", fc = "tab:blue", alpha = 0.2, linewidth = 0)

encircle(df.loc[cluster.labels_ == 2, 'Murder'], df.loc[cluster.labels_ == 2, 'Assault'], ec = "k", fc = "tab:red", alpha = 0.2, linewidth = 0)

encircle(df.loc[cluster.labels_ == 3, 'Murder'], df.loc[cluster.labels_ == 3, 'Assault'], ec = "k", fc = "tab:green", alpha = 0.2, linewidth = 0)

encircle(df.loc[cluster.labels_ == 4, 'Murder'], df.loc[cluster.labels_ == 4, 'Assault'], ec = "k", fc = "tab:orange", alpha = 0.2, linewidth = 0)



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change the size of x and y ticks

ax.tick_params("x", labelsize = 10)

ax.tick_params("y", labelsize = 10)



# set an x and y label

ax.set_xlabel("Murder", fontsize = 12)

ax.set_ylabel("Assault", fontsize = 12)



# set a title for the plot

ax.set_title("Agglomerative clustering of US arrests (5 Groups)", fontsize = 14);
# Useful for:

# Andrews curves allow one to plot multivariate data as a large number of curves that are created using the attributes of samples as coefficients for Fourier series.



# More info: 

# https://en.wikipedia.org/wiki/Andrews_plot



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/mtcars.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# prepare the data for plotting

# get the first 2 columns from our dataframe

X = df[list(df.columns)[:-2]]



# ----------------------------------------------------------------------------------------------------

# instanciate the figure 

fig = plt.figure(figsize = (12, 6))



# ----------------------------------------------------------------------------------------------------

# plot the data using pandas capabilities

ax = andrews_curves(X, 'cyl', colormap = 'Set1')



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change the x and y font size

ax.tick_params("x", labelsize = 10)

ax.tick_params("y", labelsize = 10)



# no gridlines

ax.grid(False)



# set legend and title to the plot

ax.legend(loc = "upper left", fontsize = 10,  title = "cyl")

ax.set_title("Andrews curves", fontsize = 14);
# Useful for:

# Parallel coordinates allows one to see clusters in data and to estimate other statistics visually. 

# Using parallel coordinates points are represented as connected line segments. Each vertical line represents one attribute. 

# One set of connected line segments represents one data point. Points that tend to cluster will appear closer together.



# More info: 

# https://en.wikipedia.org/wiki/Parallel_coordinates



# ----------------------------------------------------------------------------------------------------

# get the data

PATH = "/kaggle/input/the-50-plot-challenge/diamonds_filter.csv"

df = pd.read_csv(PATH)



# ----------------------------------------------------------------------------------------------------

# get the data

fig = plt.figure(figsize = (12, 6))



# ----------------------------------------------------------------------------------------------------

# plot the data using pandas capabilities

ax = parallel_coordinates(df, 'cut', colormap = "Dark2")



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change the x and y font size

ax.tick_params("x", labelsize = 10)

ax.tick_params("y", labelsize = 10)



# no gridlines

ax.grid(False)



# set legend and title to the plot

ax.legend(loc = "upper left", fontsize = 10,  title = "Diamond type")

ax.set_title("Parallel coordinates", fontsize = 14);
# If you have followed the kernel, you probably noticed that we pretty much have used the same tecniques.

# Let's wrap them all in this section



# HERE ARE THE MOST COMMON THINGS WE HAVE USED SO FAR



# ----------------------------------------------------------------------------------------------------

# instanciate the figure



# you create a basic figre like this

fig = plt.figure(figsize = (10, 10))

# ax = fig.add_subplot()



# if you need more plots you can use 

axes = fig.subplots(1,2)



# you can also use gridspec, which basically creates a matrix that you can index

# using familiar numpy slicing

# it more powerful but less common

# gs = fig.add_gridspec(5, 5)



# you need to add a new axes using add_subplot and specifiyng gridspec

# ax1 = fig.add_subplot(gs[:4, :-1])



# ----------------------------------------------------------------------------------------------------

# plot the data



# create some fake data

X = np.linspace(0,2,10)

Y = np.random.uniform(-.75,.5, 10)



# access the plots and draw some data

ax1 = axes[0]

ax1.plot(X, Y, color = "green")



# add a figre inside a figure

ax1_bis = fig.add_axes([0.25, 0.6, 0.2, 0.2])

ax1_bis.plot([1,2,1], color = "pink")



# get rid of x and y ticks

ax1_bis.set_xticks([])

ax1_bis.set_yticks([])



# get rid of spines

ax1_bis.spines["right"].set_color("None")

ax1_bis.spines["top"].set_color("None")



# change the spine position

ax1_bis.spines['left'].set_position(('data',1))



ax2 = axes[1]

ax2.plot(X, Y*-2, color = "red")



# create a secondary axis

ax2_bis = ax2.twinx()

ax2_bis.plot(X, Y)



# add a vertical line

ax2.vlines(1, min(Y*-2), max(Y*-2))



# add a horizontal line

ax1.hlines(-0.4, min(X), max(X))



# ----------------------------------------------------------------------------------------------------

# prettify the plot



# change the ticks size

ax1.tick_params("x", labelsize = 10)

ax1.tick_params("y", labelsize = 10)



ax2.tick_params("x", labelsize = 10)

ax2.tick_params("y", labelsize = 10)



# get the xticks and the xticks labels

# you can use the X specified

# xtick_location = X[::3]



# or you can get the current x

xtick_location = ax2.get_xticks()[::3]



# using specified X

# xtick_labels = ["New Label {}".format(i) for i in range(len(X))][::3]



# using current x

xtick_labels = ["New Label {}".format(i) for i in range(len(ax2.get_xticks()))][::3]



# set the xticks to be every 3'th entry

# every 3 entry

ax2.set_xticks(xtick_location)

ax2.set_xticklabels(xtick_labels, rotation = 90, fontdict = {'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})



# set x and y labels

ax1.set_xlabel("X label", fontsize = 12)

ax1.set_ylabel("Y label", fontsize = 12)



# set a title for plots

ax1.set_title("Title for plot1", fontsize = 12, color = 'black', ha = "center", va = "center")

ax2.set_title("Title for plot2", fontsize = 12, color = 'green', ha = "center", va = "center")



# annotate on plot

ax1.text(1, -0.6, "Text")



# annotate on plot

ax2.annotate('Annotate', 

             xy = (0, 0), 

             xytext = (0.5, 0.5), 

             xycoords = 'data', 

             fontsize = 8, 

             ha = 'center', 

             va = 'center',

             arrowprops = dict(arrowstyle = "-", connectionstyle = "angle,angleA=0,angleB=90,rad=0"), color = 'black'

            )



# add a custom legend

ax1.legend({class_:color for class_, color in zip([1, 2], ["black", "red"])}, loc = "upper left", title = "Custom Legend")



# set a title for figure and tight_layout()

fig.suptitle('Title for fig', fontsize = 20, color = 'darkred', alpha = .8, ha = "center", va = "center")

fig.tight_layout();
# this is a cheatsheet to plot lines in matplotlib

plt.plot([1, 1], [1, 2], linestyle="--", color = "black", label = "black")

plt.plot([0, 1], [2, 2], linestyle="-", color = "blue", label = "blue")

plt.plot([1, 1], [3, 2], linestyle="-.", color = "red", label = "red")

plt.plot([1, 2], [2, 2], linestyle=":", color = "magenta", label = "magenta")

plt.plot([1,2,3])

plt.plot([3, 2, 1])

plt.legend(loc = "upper left");
## this next line in markdown allows you to specify text in other colors

# ### <span style="color:green">* In matplotlib a basic plot starts with a figure and an axes.</span>



## this next lines allows you to put hyperlinks inside a notebook for fast reference.

## create a table of contents

# <a id = "table_of_contents"></a>

# Table of contents



# [Importing libraries and setting some helper functions](#Imports)



## code you have to put before a cell of code

# <a id = "Plot7"></a>

# # Plot 7: Marginal Boxplot

# [Go back to the Table of Contents](#table_of_contents)
# In this code we will reproduce using our knowledge the following figure from N. Rougier GitHub.

# It has to deal with the rule number 1, know you audience

# https://github.com/rougier/ten-rules/blob/master/figure-1.py



# ----------------------------------------------------------------------------------------------------

# get the data

diseases   = ["Kidney Cancer", "Bladder Cancer", "Esophageal Cancer", "Ovarian Cancer", "Liver Cancer", "Non-Hodgkin's\nlymphoma",

              "Leukemia", "Prostate Cancer", "Pancreatic Cancer", "Breast Cancer", "Colorectal Cancer", "Lung Cancer"]



men_deaths = [10000, 12000, 13000, 0, 14000, 12000, 16000, 25000, 20000, 500, 25000, 80000]



men_cases = [30000, 50000, 13000, 0, 16000, 30000, 25000, 220000, 22000, 600, 55000, 115000]



women_deaths = [6000, 5500, 5000, 20000, 9000, 12000, 13000, 0, 19000, 40000, 30000, 70000]



women_cases = [20000, 18000, 5000, 25000, 9000, 29000, 24000, 0, 21000, 160000, 55000, 97000]



# generate a df from the data

df = pd.DataFrame([diseases, men_deaths, men_cases, women_deaths, women_cases], ["diseases", "men_deaths", "men_cases", "women_deaths", "women_cases"]).T

df.sort_values("women_deaths", ascending = True, inplace = True)



# ----------------------------------------------------------------------------------------------------

# instanciate the figure

fig = plt.figure(figsize = (20, 10))

axes = fig.subplots(1, 2)



# ----------------------------------------------------------------------------------------------------

# plot the data



# for women cases, left plot

# access the plot

ax1 = axes[0]



# change the x and y limit

ax1.set_ylim(-.5, len(diseases))



# create a secondary axis

ax1_bis = ax1.twiny()

ax1_bis.invert_xaxis()  # labels read top-to-bottom



# plot a bar chart with the greater values

ax1_bis.barh(y = df["diseases"], 

         width = df["women_cases"],

         color = "red", 

         alpha = .1)



# plot a bar chart with the smaller values

# make the height a little smaller to fit inside the bigger one

ax1_bis.barh(y = df["diseases"], 

         width = df["women_deaths"],

         height = 0.6,

         color = "red", 

         alpha = 0.6)



# manipulate the spines of the plot to make it a little nicer

ax1.spines["top"].set_color("None")

ax1.spines["bottom"].set_color("None")

ax1.spines["left"].set_color("None")

ax1.spines["right"].set_color("None")



# manipulate the spines of the plot to make it a little nicer

ax1_bis.spines["top"].set_color("None")

ax1_bis.spines["bottom"].set_color("None")

ax1_bis.spines["left"].set_color("None")

ax1_bis.spines["right"].set_color("black")



# add vertical lines

ax1_bis.vlines(50_000, 0, len(diseases), color = "white", linestyle = "dotted")

ax1_bis.vlines(100_000, 0, len(diseases), color = "white", linestyle = "dotted")

ax1_bis.vlines(150_000, 0, len(diseases), color = "white", linestyle = "dotted")



# set for the main plot the x and y labels to none

ax1.set_yticks([])

ax1.set_xticks([])



# change the x ticks labels for the twinx acess

x_ticks = ax1_bis.get_xticks()

x_ticks[::2]

x_ticks_labels = ["WOMEN", "50,000", "100,000", "150,000"]



# change the x ticks with our new x ticks labels

ax1_bis.set_xticks(x_ticks[::2])

ax1_bis.set_xticklabels(x_ticks_labels, fontdict = {'horizontalalignment': 'center', 'verticalalignment': 'center_baseline', "fontsize":"12"})



# add text to the left plot

ax1_bis.text(150_000, 7, "Leading Causes\nOf Cancer Deaths", horizontalalignment = 'left', verticalalignment = 'baseline', fontdict = {'size':20})

ax1_bis.text(150_000, 6, "In 2007, there were more\nthan 1.4 million new cases\nof cancer in the Unite States.", horizontalalignment = 'left', verticalalignment = 'baseline', fontdict = {'size':12})



# annotate text

ax1_bis.annotate('NEW CASES', 

                 xy = (85000, 11), 

                 xytext = (135000, 11),

                 xycoords = 'data', 

                 fontsize = 12, 

                 ha = 'center', 

                 va = 'center',

                 arrowprops = dict(arrowstyle = "-", connectionstyle = "angle,angleA=0,angleB=90,rad=0"), color = 'black')



# annotate text

ax1_bis.annotate('DEATHS', 

                 xy = (60000, 11), 

                 xytext = (100000, 10.5), 

                 xycoords = 'data', 

                 fontsize = 12, 

                 ha = 'center', 

                 va = 'center',

                 arrowprops = dict(arrowstyle = "-", connectionstyle = "angle,angleA=0,angleB=90,rad=0"), color = 'black')



# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------



# for man cases, right plot

# access the plot

ax2 = axes[1]



# change the x and y limit

ax2.set_ylim(-.5, len(diseases))



# create a secondary axis

ax2_bis = ax2.twiny()



# plot a bar chart with the greater values

ax2_bis.barh(y = df["diseases"], 

         width = df["men_cases"],

         color = "blue", 

         alpha = .1)



# plot a bar chart with the smaller values

# make the height a little smaller to fit inside the bigger one

ax2_bis.barh(y = df["diseases"], 

         width = df["men_deaths"],

         height = 0.6,

         color = "blue", 

         alpha = 0.6)



# manipulate the spines of the plot to make it a little nicer

ax2.spines["top"].set_color("None")

ax2.spines["bottom"].set_color("None")

ax2.spines["left"].set_color("None")

ax2.spines["right"].set_color("None")



# manipulate the spines of the plot to make it a little nicer

ax2_bis.spines["top"].set_color("None")

ax2_bis.spines["bottom"].set_color("None")

ax2_bis.spines["left"].set_color("black")

ax2_bis.spines["right"].set_color("None")



# add vertical lines

ax2_bis.vlines(50_000, 0, len(diseases), color = "white", linestyle = "dotted")

ax2_bis.vlines(100_000, 0, len(diseases), color = "white", linestyle = "dotted")

ax2_bis.vlines(150_000, 0, len(diseases), color = "white", linestyle = "dotted")

ax2_bis.vlines(200_000, 0, len(diseases), color = "white", linestyle = "dotted")



# annotate text

ax2_bis.annotate('NEW CASES', 

                 xy = (100000, 11), 

                 xytext = (150000, 11),

                 xycoords = 'data', 

                 fontsize = 12, 

                 ha = 'center', 

                 va = 'center',

                 arrowprops = dict(arrowstyle = "-", connectionstyle = "angle,angleA=0,angleB=90,rad=0"), color = 'black')



# annotate text

ax2_bis.annotate('DEATHS', 

                 xy = (60000, 11), 

                 xytext = (130000, 10.5), 

                 xycoords = 'data', 

                 fontsize = 12, 

                 ha = 'center', 

                 va = 'center',

                 arrowprops = dict(arrowstyle = "-", connectionstyle = "angle,angleA=0,angleB=90,rad=0"), color = 'black')



# change the x ticks labels for the twinx acess

x_ticks = ax2_bis.get_xticks()

x_ticks_labels = ["MEN", "50,000", "100,000", "150,000", "200,000"]



# change the x ticks with our new x ticks labels

ax2_bis.set_xticks(x_ticks)

ax2_bis.set_xticklabels(x_ticks_labels, fontdict = {'horizontalalignment': 'center', 'verticalalignment': 'center', "fontsize":"12"})



# put the disease in the middle of the 2 y axis.

# on the github you have another solution

# I had to make this workaround since, my approach was a little different

for y_, disease in zip([0.095, 0.17, 0.245, 0.31, 0.385, 0.45, 0.525, 0.6, 0.667, 0.742, 0.81, 0.875], df["diseases"].values):

    plt.text(0.5, y_, disease, transform=fig.transFigure, fontsize=14,horizontalalignment='center', verticalalignment='center')



# change the x and y lables

ax2.set_xticks([])

ax2.set_yticks([])

plt.tight_layout();
# In this code we will reproduce using our knowledge the following figure from N. Rougier GitHub.

# It has to deal with the rule number 8, about chart junks.

# https://github.com/rougier/ten-rules/blob/master/figure-7.py

    

# ----------------------------------------------------------------------------------------------------

# get the data

p, n = 7, 32

X = np.linspace(0,2,n)

Y = np.random.uniform(-.75,.5,(p,n))



# ----------------------------------------------------------------------------------------------------

# instanciate the figure



# plot the chart junk

fig = plt.figure(figsize = (20, 10))

gs = fig.add_gridspec(p, 2)

ax_main = fig.add_subplot(gs[:, :1])



# plot everything into the same plot

for i, y in enumerate(Y):

    ax_main.plot(X, y, label = "Series {}".format(i + 1))

    

# ----------------------------------------------------------------------------------------------------

# plot a better solution for this chart

# for every row we created in gridspec

for i in range(p):

    # add a new figure

    ax_ = fig.add_subplot(gs[i, 1])

    for y in Y:

        # plot all the series but with a grey color and a medium alpha

        # this way the bars are like in the background

        ax_.plot(X, y, color = "grey", alpha = .5)

        # plot a new line, of the current line with a back color to stand out

        ax_.plot(X, Y[i], color = "black")



        # eliminate all the y ticks

        ax_.set_yticks([])        

        

        # for all the plots except the last one, eliminame the xticks

        if i < 6:

            ax_.set_xticks([])

        

        # manipulate the spines to make the plots look nicer

        ax_.spines["top"].set_color("None")

        ax_.spines["bottom"].set_color("None")

        ax_.spines["right"].set_color("None")

        ax_.spines["left"].set_color("Black")

        

        # set a specific y label, with the series name

        ax_.set_ylabel("Series {}".format(i + 1), horizontalalignment = "right", verticalalignment = "center", fontsize = 10, rotation = "horizontal")

                

        # for every plot, add a grey background

        ax_.patch.set_facecolor('grey')

        ax_.patch.set_alpha(0.1)

        

        # change the x and y limits

        ax_.set_xlim(0, 2)

        ax_.set_ylim(-1, 1)

        

        # add 3 vertical lines to separate better the plot

        ax_.vlines(0.5, -1, 1, lw = 1, alpha = .5)

        ax_.vlines(1, -1, 1, lw = 1, alpha = .5)

        ax_.vlines(1.5, -1, 1, lw = 1, alpha = .5)

        

# get the xticks of the last axes

x_ticks = ax_.get_xticks()



# change them to plot every 2 entrance

ax_.set_xticks(x_ticks[::2])

ax_.set_xticklabels(x_ticks[::2])



# change the x tick font size

ax_.tick_params("x", labelsize = 10)

        

# manipulate the main axes (left plot)

# change the x and y tick font size

ax_main.tick_params("x", labelsize = 10)

ax_main.tick_params("y", labelsize = 10)



# change the x and y limits

ax_main.set_xlim(0, 2)

ax_main.set_ylim(-1, 1)



# add a grid and a yellow facecolor

ax_main.grid("--")

ax_main.set_facecolor("yellow")



# add a legend for the main axes

ax_main.legend(loc = "upper left", fontsize = 10)



# add a title for the all figure

fig.suptitle('Junk chart on the left and a better approach on the right (rule #8)', fontsize = 26, color = 'darkred', alpha = .5);