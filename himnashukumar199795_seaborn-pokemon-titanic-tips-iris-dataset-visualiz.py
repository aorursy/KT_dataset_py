import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df= pd.read_csv('../input/Pokemon.csv',encoding='unicode_escape')
df.head()
df.describe()
## lmplot is used to plot Scatter Plot but it isn't purely a scatter plot .

sns.lmplot(x='Attack',y='Defense',data=df)
## To make it look better we remove regression line and colors by evolution_stage

sns.lmplot(x='Attack',y='Defense',data=df,fit_reg=True,hue='Stage')
plt.figure(figsize=(10,5))

sns.boxplot(data=df)
plt.figure(figsize=(10,5))

sns.boxplot(data=df[df.var().head().index])
## To remove unnecessary column's that are appearing in boxplot

new_df=df.drop(['Total','Stage','Legendary'],axis=1)

sns.boxplot(data=new_df)
## ViolinPlot are similar to box plot but are easy to understand 

## To define custom colors we can create a list and pass as an palette

## by default color is darkgrid in Violin Plot so making it whitegrid

sns.set_style('whitegrid')

pkmn_type_colors = ['#78C850',  # Grass

                    '#F08030',  # Fire

                    '#6890F0',  # Water

                    '#A8B820',  # Bug

                    '#A8A878',  # Normal

                    '#A040A0',  # Poison

                    '#F8D030',  # Electric

                    '#E0C068',  # Ground

                    '#EE99AC',  # Fairy

                    '#C03028',  # Fighting

                    '#F85888',  # Psychic

                    '#B8A038',  # Rock

                    '#705898',  # Ghost

                    '#98D8D8',  # Ice

                    '#7038F8',  # Dragon

                   ]
## Using matplotlib pyplot we can increase the size of figure

fig,ax=plt.subplots()

fig.set_size_inches(30,10)

sns.violinplot(x='Type 1',y='Attack',data=df,ax=ax,palette=pkmn_type_colors)
## Swarm Plot

fig,ax=plt.subplots()

fig.set_size_inches(17.7,9.27)

sns.swarmplot(x='Type 1',y='Attack',data=df,palette=pkmn_type_colors,ax=ax)
fig,ax=plt.subplots()

fig.set_size_inches(17.7,9.27)

sns.violinplot(x='Type 1',y='Attack',data=df,inner=None,##inner is used to remove bars inside violinplot

               palette=pkmn_type_colors)



sns.swarmplot(x='Type 1',

             y='Attack',

             data=df,

             color='k',## to make color black of dots

             alpha=0.8)## to make transparent

plt.title('Attack by Pokemon Type')
## Combining many columns into one

melted_df=pd.melt(new_df,id_vars=['Name','Type 1','Type 2'],var_name='Ability')
melted_df.head()
fig,ax=plt.subplots()

fig.set_size_inches(12,7)

sns.swarmplot(x='Ability',y='value',data=melted_df,hue='Type 1')
## Making it look more understandable

fig,ax=plt.subplots()

fig.set_size_inches(14,7)

sns.swarmplot(x='Ability',y='value',data=melted_df,hue='Type 1',palette=pkmn_type_colors)

plt.ylim(0,250)

plt.legend(bbox_to_anchor=(1, 1), loc=2) ## To place legend box outside the plot
sns.distplot(df.Defense)
plt.figure(figsize=(10,7))

sns.countplot(x='Type 1',data=df,palette=pkmn_type_colors)

plt.xticks(rotation=-45)
#factorplot name changed to catplot

my_plot=sns.catplot(x='Type 1',y='Defense',data=df,hue='Stage',col='Stage',palette=pkmn_type_colors,kind='swarm')

my_plot.set_xticklabels(rotation=-45)
plt.figure(figsize=(8,4))

sns.lmplot(x='Attack',y='Speed',data=df,fit_reg=False,hue='Stage')

sns.kdeplot(df.Attack,df.Speed,alpha=0.8)

plt.ylim(0,160)

plt.xlim(0,170) # Used to limit x and y co-ordinates
import matplotlib as mpl
# simple function to plot some offset sine waves

def sinplot(flip=1):

    x=np.linspace(0,14,100)

    for i in range(1,7):

        plt.plot(x,np.sin(x+i*0.5)*(7-i)*flip)



data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sinplot()

# plot with matplotlib default
sns.set()

# to switch to seaborn defaults
sinplot()

# plot with seaborn set
# Seaborn splits matplotlib parameters into two independent groups.

# The first group sets the aesthetic style of the plot, and the 

# second scales various elements of the figure so that it can be

# easily incorporated into different contexts.



# The interface for manipulating these parameters are two pairs of

# functions. To control the style, use the axes_style() and 

# set_style() functions. To scale the plot, use the plotting_context() and set_context() functions. In both cases, the first function 

# returns a dictionary of parameters and the second sets the 

# matplotlib defaults.
# There are five preset seaborn themes: darkgrid, whitegrid, 

#  dark, white, and ticks

sns.set_style('dark')

sinplot()
sns.set_style('white')

sinplot()
sns.set_style('darkgrid')

sinplot()
sns.set_style('whitegrid')

sinplot()
sns.set_style('ticks')

sinplot()
# Removing axis spines

sinplot()

sns.despine()
fig,ax=plt.subplots()

sns.violinplot(data=data)

sns.despine(offset=10,trim=True)

# when ticks don't cover the whole range of the axis,the trim

# parameter will limit the range of the surviving spines
# you can also control which spines are to remove with additional

# arguments

sns.set_style('darkgrid')

fig,ax=plt.subplots()

sns.violinplot(data=data)

sns.despine(left=True)
# temporarily setting figure style

with sns.axes_style('dark'):

    sinplot()   
sinplot()
# # Overriding elements of the seaborns style

# if you want to customize the seaborn styles , you can pass a 

# dictionary of parameters to the rc argument of axes_style() and 

# set_style().Note that you can only override the parameters that are

# part of the style definition through this method.(However the 

# higher level set() function takes a dictionary of any

# matplotlib parameters)

# If you want to see what parameters are included, you can just 

# call the function with no arguments,which will return the 

# current settings

sns.axes_style()
sns.set_style('dark',{'axes.facecolor':'0.9'})

sinplot()
sns.axes_style()
# Scaling Plot Elements

# A separate set of parameters control the scale of plot elements,

# which should let you use the same code to make plots that are 

# suited for use in settings where larger or smaller plots are

# appropriate.



# First let’s reset the default parameters by calling set():

sns.set()
# The four preset contexts, in order of relative size, are paper, 

# notebook, talk, and poster. The notebook style is the default, 

# and was used in the plots above.

sns.set_context('paper')

sinplot()
sns.set_context('talk')

sinplot()
sns.set_context('poster')

sinplot()
sns.plotting_context()
sns.set_context('notebook',rc={'lines.linewidth':'2.5'})

sinplot()
sns.plotting_context()
# CHOOSING COLOR PALETTES

# The most important function for working with discrete 

# color palettes is color_palette(). This function provides an 

# interface to many (though not all) of the possible ways you can

# generate colors in seaborn, and it’s used internally by any function 

# that has a palette argument (and in some cases for a color

# argument when multiple colors are needed).

# color_palette() will accept the name of any seaborn palette or 

# matplotlib colormap (except jet, which you should never use).

# It can also take a list of colors specified in any valid matplotlib 

# format (RGB tuples, hex color codes, or HTML color names). The 

# return value is always a list of RGB tuples.



# Finally, calling color_palette() with no arguments will return the

# current default color cycle.



# A corresponding function, set_palette(), takes the same arguments

# and will set the default color cycle for all plots. You can also

# use color_palette() in a with statement to temporarily change the

# default palette (see below).



# It is generally not possible to know what kind of color palette or

# colormap is best for a set of data without knowing about the

# characteristics of the data. Following that, we’ll break up 

# the different ways to use color_palette() and other seaborn

# palette functions by the three general kinds of color palettes:



# **************qualitative, sequential, and diverging***********



# sns.color_palette()

sns.choose_colorbrewer_palette('diverging')

# plotting univariate distribution

# The most convenient way to take a quick look at a univariate 

# distribution in seaborn is the distplot() function.By default

# this will draw a histogram and fit a kernel density estimate

x=np.random.normal(size=100)

sns.set()

sns.distplot(x)
# histogram

# histograms are likely familiar and a hist function already exists

# in matplotlib.A histogram represents the distribution of data by 

# forming bins along the range of the data nd then drawing bars to 

# show the number of observations that fall in each bin

# To illustrate this ,let's remove the desinty curve and add a rug

# plot,which draws a small vertical tick t each observation.You can 

# make the rug plot itself with the rugplot function but it is also 

# available in distplot()

sns.distplot(x,kde=False,rug=True)
# when drawing histograms the main choice you have is the number of

# bins to use and where to place them.Distplot() uses a simple rule

# to make good guess for what the right number is the default,but

# trying more or fewer bins might reveal other features in the data

sns.distplot(x,bins=20,rug=True,kde=False)
# KERNEL DENSITY ESTIMATION

# the kernel density estimate may be less familiar but it can be a

# useful tool for plotting the shape of a distribution.Like the

# hsitogram,the KDE plots encodes the density of observations on one

# axis with height along the other axis:

sns.distplot(x,hist=False,rug=True)
sns.kdeplot(x,shade=True)

sns.rugplot(x)
# the bandwidth parameter of the KDE controls how tightly the 

# estiamtion is to fit to the data,much like the bin size in a 

# histogram.It corresponds to the width of the kernels.The default

# behaviour tries to guess a good vlue using a common reference rule

# ,but it my be helpful to try larger or smaller values

sns.kdeplot(x)

sns.kdeplot(x,bw=2,label='bw:2')

sns.kdeplot(x,bw=0.2,label='bw:0.2')
# Fitting parametric distributions

# you can also use the distplot() to fit a parametric distribution

# to a dataset and visually evaluate how closely it corresponds to

# the observed data:

x=np.random.gamma(6,size=200)

from scipy import stats

sns.distplot(x,kde=False,fit=stats.gamma)
# Plotting Bivariate distributions

# The easiest way to do this in seaborn is to just use the joinplot()

# function,which creates a multi-panel figure that shows both the 

# bivariate(or joint) relationship between two variables along 

# with the univariate (or marginal) distributiob of each on separate

# axis

mean,cov=[0,1],[(1,0.5),(0.5,1)]

data=np.random.multivariate_normal(mean,cov,200)

df=pd.DataFrame(data,columns=['x','y'])

print(df)
# Scatterplots

# the most familiar way to visualize a bivariate distribution is

# a scatterlpot,where each observation is shown with point at the 

# x and y values.This is analogous to rugplot on two dimesnions

# .You can draw a Scatterplot with the matplotlib plt.scatter,and

# it is also the default kind of plot shown by the jointplot()

# function:

sns.jointplot(x='x',y='y',data=df)
# Hesxbin plots

# the bivariate analogue of a histogram is known as a hexbin plot 

# because it shows the count of observations that fall within 

# hexagonal bins.this plot works best with relatively large datasets.

# it's availble through the matplotlib plt.hexbin() function and as

# a style in jointplot.it looks best with a white background

x,y=np.random.multivariate_normal(mean,cov,1000).T

with sns.axes_style('white'):

    sns.jointplot(x=x,y=y,kind='hex',color='k')
# kernel density estimation

# it is also possible to use the kernel density estimation procedure

# described above to visualoze a bivariate distribution.In seaborn,

# this kind of plot is shown with a contour plot and is available

# as a style in jointplot()

sns.jointplot(x=x,y=y,kind='kde')
# you can also draw a 2d kernel density plot with the kdeplot()

# function.This allows you to draw this kind of plot onto a specific

# matplotlib axes,whereas the joint plot() function manages it's own

# figure

f,ax=plt.subplots()

sns.kdeplot(df['x'],df['y'],ax=ax)

sns.rugplot(df['x'],color='g',ax=ax)

sns.rugplot(df['y'],color='b',ax=ax,vertical=True)
# if you wish to show the bivariate density more continously,you can

# simply increase the number of contour levels:

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1,

reverse=True)



sns.kdeplot(df['x'],df['y'],n_levels=60,shade=True,cmap=cmap)
# the jointplot function uses a jointgrid to manage the figure.For

# more flexibility,you my want to draw your figure by using JointGrid

# directly.Jointplot() return the JointGrid object after plotting

# which you can use to add more layers or to tweak other aspects of

# the visualization

g=sns.jointplot(x='x',y='x',data=df,kind='kde',color='b')

g.plot_joint(plt.scatter,c='w',s=30,linewidth=1,marker='+')

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels('$x$','$y$')
# visualising pairwise relationships in a dataset

# To plot multiple pairwise bivariate distributions in a dataset,you

# can use the pirplot() function.This creates a matrix of axes

# and shows the relationship for each pair of columns in a DataFrame

# By default,it also draws the univariate distribution of each

# variable on the diagonal axes

iris=sns.load_dataset('iris')

sns.pairplot(iris)
# Much like the relationship between jointplot() and JointGrid , 

# the pairplot() function is built on top of a PairGrid object,which

# can be used directly for more flexibility:

g=sns.pairplot(iris)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot,cmap='Blues_d',n_levels=6)
# **************** Plotting with categorical data ***************



# It’s useful to divide seaborn’s categorical plots into three groups:

# those that show each observation at each level of the categorical 

# variable, those that show an abstract representation of each

# distribution of observations, and those that apply a statistical 

# estimation to show a measure of central tendency and confidence

# interval. The first includes the functions swarmplot() and 

# stripplot(), the second includes boxplot() and violinplot(), and 

# the third includes barplot() and pointplot(). These functions all 

# share a basic API for how they accept data, although each has

# specific parameters that control the particulars of the

# visualization that is applied to that data.



# Much like the relationship between regplot() and lmplot(), in

# seaborn there are both relatively low-level and relatively 

# high-level approaches for making categorical plots. The functions

# named above are all low-level in that they plot onto a specific

# matplotlib axes. There is also the higher-level factorplot(),

# which combines these functions with a FacetGrid to apply a 

# categorical plot across a grid of figure panels



sns.set(style='whitegrid',color_codes=True)
titanic=sns.load_dataset('titanic')

tips=sns.load_dataset('tips')

iris=sns.load_dataset('iris')
# Categorical scatterplots

# a simple way to show the values of some quantitative variable 

# across the levels of a categorical variable uses stripplot(),which

# generalizes a scatterplot to the case where one of the variables

# is categorical

sns.stripplot(x='day',y='total_bill',data=tips)
# in a strip plot the scatterplot points will usually overlap.this 

# makes it difficult to see the full distribution of data.One easy

# solution is to adjust the positions (only along the categorical

# axis) using some random jitter

sns.stripplot(x='day',y='total_bill',data=tips,jitter=True)
# a different approach would be to use the function swarmplot(),which

# positions each scatterplot point on the categorical axis with

# an algorithm that avoids overlapping points

sns.swarmplot(x='day',y='total_bill',data=tips)
# It's also possible to add a nested categorical variable with the 

# hue parameter.Above the color and position on the categorical 

# axis are redundant,but now each provides info about one of the 

# two variables

sns.swarmplot(x='day',y='total_bill',hue='sex',data=tips)
# In general, the seaborn categorical plotting functions try to 

# infer the order of categories from the data. If your data have a 

# pandas Categorical datatype, then the default order of the 

# categories can be set there. For other datatypes, string-typed 

# categories will be plotted in the order they appear in the 

# DataFrame, but categories that look numerical will be sorted:

sns.swarmplot(x='size',y='total_bill',data=tips)
# With these plots, it’s often helpful to put the categorical variable

# on the vertical axis (this is particularly useful when the category

# names are relatively long or there are many categories). You can 

# force an orientation using the orient keyword, but usually plot 

# orientation can be inferred from the datatypes of the variables 

# passed to x and/or y:

sns.swarmplot(x='total_bill',y='day',hue='sex',data=tips)
# ***** Distribution of observations within categories ******

# Boxplots



# This kind of plot shows the three quartile values of the 

# distribution along with extreme values. The “whiskers” extend to 

# points that lie within 1.5 IQRs of the lower and upper quartile,

# and then observations that fall outside this range are displayed 

# independently. Importantly, this means that each value in the 

# boxplot corresponds to an actual observation in the data:



sns.boxplot(x='day',y='total_bill',data=tips)
# For boxplots, the assumption when using a hue variable is that it 

# is nested within the x or y variable. This means that by default,

# the boxes for different levels of hue will be offset, as you can see

# above. If your hue variable is not nested, you can set the dodge

# parameter to disable offsetting:



sns.boxplot(x='day',y='total_bill',hue='time',data=tips)
tips['weekend']=tips['day'].isin(['Sat','Sun'])

sns.boxplot(x='day',y='total_bill',hue='weekend',data=tips,

dodge=False)
# ViolinPlots

# A different approach is a violinplot(),which combines a boxplot 

# with the kernel density estimation procedure



sns.violinplot(x='total_bill',y='day',data=tips,hue='time')
# This approach uses the kernel density estimate to provide a better 

# description of the distribution of values. Additionally,

# the quartile and whikser values from the boxplot are shown inside 

# the violin. Because the violinplot uses a KDE, there are some other 

# parameters that may need tweaking, adding some complexity relative 

# to the straightforward boxplot:



sns.violinplot(x='total_bill',y='day',hue='time',data=tips,

bw=0.1,scale='count',scale_hue=False)



# It’s also possible to “split” the violins when the hue parameter 

# has only two levels, which can allow for a more efficient use of 

# space:

sns.violinplot(x='day',y='total_bill',hue='sex',data=tips,split=True)



# Finally, there are several options for the plot that is drawn on 

# the interior of the violins, including ways to show each individual 

# observation instead of the summary boxplot values:



sns.violinplot(x='day',y='total_bill',hue='sex',data=tips,

split=True,inner='stick',palette='Set2')
# It can also be useful to combine stripplot() or swarmplot() 

# with violinplot() or boxplot() to show each observation along 

# with a summary of the distribution:



sns.violinplot(x='day',y='total_bill',data=tips)

sns.swarmplot(x='day',y='total_bill',data=tips,color='w',alpha=0.5)
# *********** Statistical Estimation Within Categories ***********



# Often, rather than showing the distribution within each category,

# you might want to show the central tendency of the values.



# BAR PLOT:

# In seaborn,the braplot() function operates on a full dataset and 

# shows an arbitrary estimate,using the mean by default.When there

# are multiple obersvations in each category,it also uses

# bootstrapping to compute a confidence interval around the estimate

# and plots that using error bars



sns.barplot(x='sex',y='survived',data=titanic)
sns.barplot(x='sex',y='survived',hue='class',data=titanic)
# A special case for the bar plot is when you want to show the number 

# of observations in each category rather than computing a statistic 

# for a second variable. This is similar to a histogram over a 

# categorical, rather than quantitative, variable. In seaborn, it’s 

# easy to do so with the countplot() function:



sns.countplot(x='deck',data=titanic)
sns.countplot(x='deck',hue='class',data=titanic)
# POINT PLOTS:



# An alternative style for visualizing the same information is 

# offered by the pointplot() function. This function also encodes

# the value of the estimate with height on the other axis, but rather 

# than show a full bar it just plots the point estimate and confidence

# interval. Additionally, pointplot connects points from the same hue 

# category. This makes it easy to see how the main relationship is 

# changing as a function of a second variable, because your eyes are

# quite good at picking up on differences of slopes:



sns.pointplot(x='sex',y='survived',data=titanic)
sns.pointplot(x='sex',y='survived',hue='class',data=titanic)
# To make figures that reproduce well in black and white, it can be 

# good to use different markers and line styles for the levels of 

# the hue category:



sns.pointplot(x='class',y='survived',hue='sex',data=titanic,

palette={'male':'g','female':'m'},markers=['o','^'],

linestyles=['--','-'])
# PLOTTING " WIDE-FORM " DATA



# While using “long-form” or “tidy” data is preferred, these 

# functions can also by applied to “wide-form” data in a variety of

# formats, including pandas DataFrames or two-dimensional numpy 

# arrays. These objects should be passed directly to the data 

# parameter:



sns.boxplot(data=iris)

# Additionally, these functions accept vectors of Pandas or

# numpy objects rather than variables in a DataFrame:



sns.violinplot(x='species',y='sepal_length',data=iris)

# Drawing multi-panel categorical plots

# As we mentioned above, there are two ways to draw categorical plots

# in seaborn. Similar to the duality in the regression plots, you can

# either use the functions introduced above, or the higher-level 

# function factorplot(), which combines these functions with a 

# FacetGrid() to add the ability to examine additional categories 

# through the larger structure of the figure. By default, 

# factorplot() produces a pointplot():



sns.factorplot(x='day',y='total_bill',data=tips)
sns.factorplot(x='day',y='total_bill',hue='smoker',data=tips)
# However, the kind parameter lets you chose any of the kinds of 

# plots discussed above:



sns.factorplot(x='day',y='total_bill',data=tips,kind='bar',

               hue='smoker')

# The main advantage of using a factorplot() is that it is very easy

# to “facet” the plot and investigate the role of other categorical 

# variables:



sns.factorplot(x='day',y='total_bill',hue='smoker',data=tips,

              kind='swarm',col='time')
# Any kind of plot can be drawn. Because of the way FacetGrid works,

# to change the size and shape of the figure you need to specify 

# the size and aspect arguments, which apply to each facet:



sns.factorplot(x='time',y='total_bill',hue='smoker',col='day',

               data=tips,kind='box',size=4,aspect=0.5)
# It is important to note that you could also make this plot by

# using boxplot() and FacetGrid directly. However, special care must

# be taken to ensure that the order of the categorical variables is 

# enforced in each facet, either by using data with a Categorical 

# datatype or by passing order and hue_order.



# Because of the generalized API of the categorical plots, they 

# should be easy to apply to other more complex contexts. For

# example, they are easily combined with a PairGrid to show 

# categorical relationships across several different variables:



g=sns.PairGrid(tips,x_vars=['smoker','time','sex'],

              y_vars=['total_bill','tip'],aspect=0.75,size=3.5)

g.map(sns.violinplot,palette='pastel')
# # VISUALIZING LINEAR RELATIONSHIPS

# the regression plots in seaborn are primarily intended to add a 

# visual guide that helps to emphasize patterns in a dataset during 

# exploratory data analyses. That is to say that seaborn is not itself

# a package for statistical analysis. To obtain quantitative measures 

# related to the fit of regression models, you should use statsmodels.

# The goal of seaborn, however, is to make exploring a dataset through

# visualization quick and easy, as doing so is just as (if not more)

# important than exploring a dataset through tables of statistics.



# FUNCTIONS TO DRAW LINEAR REGRESSION MODELS

# Two main functions in seaborn are used to visualize a linear 

# relationship as determined through regression. These functions, 

# regplot() and lmplot() are closely related, and share much of their 

# core functionality. It is important to understand the ways they

# differ, however, so that you can quickly choose the correct tool 

# for particular job.



# In the simplest invocation, both functions draw a scatterplot of 

# two variables, x and y, and then fit the regression model y ~ x 

# and plot the resulting regression line and a 95% confidence 

# interval for that regression:



sns.regplot(x='total_bill',y='tip',data=tips)
sns.lmplot(x='total_bill',y='tip',data=tips)



# You should note that the resulting plots are identical,

# except that the figure shapes are different. We will explain

# why this is shortly. For now, the other main difference to know

# about is that regplot() accepts the x and y variables in a variety 

# of formats including simple numpy arrays, pandas Series objects,

# or as references to variables in a pandas DataFrame object passed 

# to data. In contrast, lmplot() has data as a required parameter and 

# the x and y variables must be specified as strings. This data 

# format is called “long-form” or “tidy” data. Other than this input

# flexibility, regplot() possesses a subset of lmplot()‘s features,

# so we will demonstrate them using the latter.



# It’s possible to fit a linear regression when one of the variables

# takes discrete values, however, the simple scatterplot produced by 

# this kind of dataset is often not optimal
sns.lmplot(x='size',y='tip',data=tips)
# One option is to add some random noise (“jitter”) to the discrete

# values to make the distribution of those values more clear. Note 

# that jitter is applied only to the scatterplot data and does not 

# influence the regression line fit itself:



sns.lmplot(x='size',y='tip',data=tips,x_jitter=0.1)
# A second option is to collapse over the observations in each

# discrete bin to plot an estimate of central tendency along with

# a confidence interval:



sns.lmplot(x='size',y='tip',data=tips,x_estimator=np.mean)
# CONDITIONING ON OTHER VARIABLES

# The plots above show many ways to explore the relationship between

# a pair of variables. Often, however, a more interesting question

# is “how does the relationship between these two variables change

# as a function of a third variable?” This is where the difference

# between regplot() and lmplot() appears. While regplot() always

# shows a single relationship, lmplot() combines regplot() with 

# FacetGrid to provide an easy interface to show a linear regression 

# on “faceted” plots that allow you to explore interactions with up

# to three additional categorical variables.



# The best way to separate out a relationship is to plot both levels

# on the same axes and to use color to distinguish them:



sns.set(color_codes=True)

sns.lmplot(x='total_bill',y='tip',hue='smoker',data=tips)
# In addition to color, it’s possible to use different scatterplot

# markers to make plots the reproduce to black and white better.

# You also have full control over the colors used:



sns.lmplot(x='total_bill',y='tip',hue='smoker',markers=['o','x'],

          palette='Set1',data=tips)
# To add another variable, you can draw multiple “facets” which 

# each level of the variable appearing in the rows or columns of the

# grid:



sns.lmplot(x='total_bill',y='tip',hue='smoker',col='time',data=tips)
sns.lmplot(x='total_bill',y='tip',hue='smoker',col='time',row='sex',

          data=tips)