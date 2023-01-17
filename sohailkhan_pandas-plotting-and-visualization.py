# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%matplotlib inline

# first checkout the data
salaries = pd.read_csv('../input/Salaries.csv')
salaries.info()
# convert the pay columns to numeric
salaries = salaries.convert_objects(convert_numeric=True)
# Notes seems to be missing a lot of values
# drop it, with the drop method.  axis is either 0 for rows, 1 for columns
salaries = salaries.drop('Notes', axis=1)
salaries.describe()
# i am using seaborn to change aesthetics of the plots
sns.set_style("whitegrid")

# matplotlib.pyplot is the main module that provides the plotting API
x = [np.random.uniform(100) for _ in range(200)]
y = [np.random.uniform(100) for _ in range(200)]
plt.scatter(x,y)


# seaborn is a wrapper that abstracts out the aesthetics behind matplotlib 
# it provides several baked in plotting arrangements
# jointplot takes in a dataframe data parameter from which you can use the column
# names to specify the X and Y axis

sns.jointplot(x = 'X', y = 'Y', data = pd.DataFrame({'X': x, 'Y': y}))
# and a small change to get a density plot
sns.jointplot(x = 'X', y = 'Y', kind="kde",  data = pd.DataFrame({'X': x, 'Y': y}))
# this crashes kaggle for some reason

# using the Salaries data
# how does year affect the benefits an employee receives?
# ax = sns.violinplot(x="Year", y="Benefits", data=salaries)
# ax.set_ylim((0, salaries.Benefits.max()))
plt.figure()

# save a reference to a Figure object
fig = plt.figure()
# check out what you get when you try to get the string representation of it
str(fig)

# its public API is
print([attr for attr in dir(fig) if not attr.startswith('_')])
# add_subplot returns the axis object that you can reference
# and manipulate

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)

ax2.scatter(*[[np.random.normal(100) for i in range(100)] for i in range(2)])
# instead you have to run the sequence of commands in one cell

# create the figure to hold the subplots
fig = plt.figure()

# add a subplot     row, col, selected plot for reference
# i will be using one row that contains two plots
ax1 = fig.add_subplot(1,2,1)
ax1.plot(np.random.randn(50).cumsum(), 'k--')

# same as before
ax2 = fig.add_subplot(1,2,2)           # this indicates color and line style
ax2.plot(np.random.randn(50).cumsum(), 'r--')
pay_columns = salaries.columns[3:salaries.columns.get_loc('Year')]
pay_columns
# making a 2x3 figure with plots of histograms

# 2x3 array of col names, this is a tricky but useful way of grouping 
# list elements
pays_arrangement = list(zip(*(iter(pay_columns),) * 3))

#  Here I am using the plt.subplots command
#  The result of this action gives you 
#  a figure and 2x3 array of axes
fig, axes = plt.subplots(2,3)


# since I have a 2x3 array of col names and
# a 2x3 array of axes, i can iterate over them in parallel

for i in range(len(pays_arrangement)):
    for j in range(len(pays_arrangement[i])):
        # pass in axes to pandas hist
        salaries[pays_arrangement[i][j]].hist(ax=axes[i,j])
        
        # axis objects have a lot of methods for customizing the look of a plot
        axes[i,j].set_title(pays_arrangement[i][j])
plt.show()
# that doesn't look too good
# you can use a combination of figheight, figwidth, and subplot spacing to achieve a more readable 
# chart

#     2x3 array of axes
fig, axes = plt.subplots(2,3)

# set the figure height
fig.set_figheight(5)
fig.set_figwidth(12)

for i in range(len(pays_arrangement)):
    for j in range(len(pays_arrangement[i])):
        # pass in axes to pandas hist
        salaries[pays_arrangement[i][j]].hist(ax=axes[i,j])
        axes[i,j].set_title(pays_arrangement[i][j])
        
# add a row of emptiness between the two rows
plt.subplots_adjust(hspace=1)
# add a row of emptiness between the cols
plt.subplots_adjust(wspace=1)
plt.show()
# get one of the axes objects
ax = axes[1,1]

# to get the actual ticks on the xaxis
ax.get_xticks()

# to adjust the rotation of the ticks, use set_xticklabels of the axes object
# you need to pass it a list of ticks, so here you can pass the original ticks from above
ax.set_xticklabels(labels=ax.get_xticks(), 
                   # pass in the rotation offset
                   rotation=30)
# and here is a cleaner version using tick rotation and plot spacing
fig, axes = plt.subplots(2,3)

# set the figure height
fig.set_figheight(5)
fig.set_figwidth(12)

for i in range(len(pays_arrangement)):
    for j in range(len(pays_arrangement[i])):
        salaries[pays_arrangement[i][j]].hist(ax=axes[i,j])
        axes[i,j].set_title(pays_arrangement[i][j])
        
        #         set xticks      with these labels,
        axes[i,j].set_xticklabels(labels=axes[i,j].get_xticks(), 
                                  # with this rotation
                                  rotation=30)
        
plt.subplots_adjust(hspace=1)
plt.subplots_adjust(wspace=1)
plt.show()
# add a column for vowel counts in employee names
from re import IGNORECASE
salaries['VowelCounts'] = salaries.EmployeeName.str.count(r'[aeiou]', flags=IGNORECASE)


# i will be using this function to test out the colors, markers, and line styles
def generateVowelCountLinePlot(*all_at_once_args, **style_args):
    """Count the vowels in a name and see if there is a correlation between that and TotalPay..."""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    mean_totalpay = salaries.groupby('VowelCounts').TotalPay.agg('mean')
    
    X = mean_totalpay.index
    Y = mean_totalpay.values

    plt.title('Vowel Counts vs Average Salary')
    
    #  this is where the style arguments will be passed
    # eg: ax.plot(X,Y, linestyle='--')
    # or 
    #     ax.plot(x,y,'--')
    ax.plot(X,Y, *all_at_once_args, **style_args)
    return ax
# linestyle gives you the opportunity to specify the style of the line
generateVowelCountLinePlot(linestyle='--')
# color specifies the plot color
# the color can be passed in if you know the hex code for the color
# r == '#FF0000'
generateVowelCountLinePlot(color='r')
# and you can combine them
generateVowelCountLinePlot(linestyle='--', color='r')
# I can't seem to see where the actual points are located on the graphs above, add a marker
generateVowelCountLinePlot(marker='o')
# you can pass in all the style parameters at once as the first argument to the plotting function
# eg. plt.plot(X,Y, 'ro--')
generateVowelCountLinePlot('yo--')

# there are many more parameters that can be manipulated
# here, instead of using linear interpolation as in the above examples
# I will use a step interpolation
generateVowelCountLinePlot(drawstyle='steps-pre', label='steps-post')
# control x axis ranges and y axis ranges
# with xlim and ylim
# here i  modified the above function to take in a new kwarg of xlims

def generateVowelCountLinePlot(*all_at_once_args, xlims = None, **style_args):
    """Count the vowels in a name and see if there is a correlation between that and TotalPay..."""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    mean_totalpay = salaries.groupby('VowelCounts').TotalPay.agg('mean')
    X = mean_totalpay.index
    Y = mean_totalpay.values

    plt.title('Vowel Counts vs Average Salary')
    ax.plot(X,Y, *all_at_once_args, **style_args)
    
    if xlims:
        # setting the x axis limits if user passed in xlims kwarg
        ax.set_xlim(xlims)
    
    # make line chart start at 0
    ax.set_ylim((0, Y.max()))
    return ax
generateVowelCountLinePlot(xlims=(1,9))
generateVowelCountLinePlot('o--')

# modify the plot returned by the above function with plt.xlim
plt.xlim([3, 7])
plt.ylim([60000, 80000])
# modified the function again to demonstrate setting xticks

def generateVowelCountLinePlot(*all_at_once_args, 
                               xticks = None, 
                               **style_args):
    """Count the vowels in a name and see if there is a correlation between that and TotalPay..."""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    mean_totalpay = salaries.groupby('VowelCounts').TotalPay.agg('mean')
    X = mean_totalpay.index
    Y = mean_totalpay.values

    ax.plot(X,Y, *all_at_once_args, **style_args)
    
    # explicitly set the xticks
    # the result of set_xticks is suprisingly a ticks object
    ticks = ax.set_xticks(xticks) if xticks else None
    
    # adjust the axis labels and title label
    ax.set_title('Vowel Counts vs Average Salary')
    ax.set_xlabel('Number of Vowels')
    ax.set_ylabel('Average Total Salary')
    return ax

# call the function again, this time passing in explicit 
# xtick positions
generateVowelCountLinePlot('ro--', xticks=[1,5, 10])
# instead of numerical xticks you can actually manipulate the
# xticks to have more semantic labels

# recover the axes object returned by the function
ax = generateVowelCountLinePlot('ro--', xticks=[1,5, 10])

# and manipulate its xticklabels by getting its current xticks
# in addition i pass in a value for rotation and
# I specified that i want a small tick font size
labels = ax.set_xticklabels(['{} vowels'.format(i) for i in ax.get_xticks()],
                             rotation=30, fontsize='small')
# modify the function above to plot salaries vs vowel counts by years
def generateVowelCountLinePlotByYear(*all_at_once_args, colors='rgbyk', **style_args):
    """Count the vowels in a name and see if there is a correlation between that and TotalPay..."""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    # add additional grouping level
    mean_totalpay = salaries.groupby(('Year', 'VowelCounts')).TotalPay.agg('mean')
    years = salaries.Year.unique()
    
    # check to see if the colors string passed in has more than enough colors for the years
    if len(colors) < len(years):
        raise IndexError("Need more colors")
    
    # get index to specify color
    for idx, year in enumerate(years):
        X = mean_totalpay[year].index
        Y = mean_totalpay[year].values
        
        # plot the salary vs vowelcounts for current year using a 
        # predetermined color
        # give it a label of the year
        ax.plot(X,Y, color=colors[idx], label=str(year), *all_at_once_args, **style_args)
    

    # adjust the axis labels and title label
    ax.set_title('Vowel Counts vs Average Salary')
    ax.set_xlabel('Number of Vowels')
    ax.set_ylabel('Average Total Salary')
    
    # set
    ax.legend(loc='best')
    return ax
generateVowelCountLinePlotByYear('--')
# how do you get the max salary for a particular vowel count accross all years?
mean_totalpay = salaries.groupby(('Year', 'VowelCounts')).TotalPay.agg('mean')

# this gives you the first n rows
# of the groupby series
mean_totalpay[0:3:,]

# this gives you all the years for a particular vowel count
mean_totalpay[:, 3]

# then 
mean_totalpay[:, 3].max() # gives the max salary over all years for a particular vowel count
# you can place your annotations in a container to facilitate plotting them
# here I only need the x position and the annotation
# I will use pandas to figure out the y position of the annotation
annotations = [
    (6, 'Decreasing Salary!'),
    (14, 'Small sample size?')
]

# so modify the function once more to add annotations
def generateVowelCountLinePlotByYear(*all_at_once_args, colors='rgbyk', annotations = annotations, **style_args):
    """Count the vowels in a name and see if there is a correlation between that and TotalPay..."""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    # add additional grouping level
    mean_totalpay = salaries.groupby(('Year', 'VowelCounts')).TotalPay.agg('mean')
    years = salaries.Year.unique()
    

    
    # get index to specify color
    # plot a line for each year
    for idx, year in enumerate(years):
        X = mean_totalpay[year].index
        Y = mean_totalpay[year].values
        ax.plot(X,Y, color=colors[idx], label=str(year), *all_at_once_args, **style_args)
    

    # adjust the axis labels and title label
    ax.set_title('Vowel Counts vs Average Salary')
    ax.set_xlabel('Number of Vowels')
    ax.set_ylabel('Average Total Salary')
    
    # annotate the graph
    for x_pos, descrip in annotations:
        # find the max pay amongst all years
        # so that you can put the text above that point
        # i'm adding an offset of about 5000 to plot it above that max value
        y_pos = mean_totalpay[:, x_pos].max()+5000
        
        ax.text(x_pos, y_pos, descrip,
               family='monospace', fontsize=10)
    # set
    ax.legend(loc='best')
    return fig, ax

generateVowelCountLinePlotByYear()
# first, you can automatically plot different columns as lines
# on the same subplot with the plot method of the df
# you also get the legend and colors set up for you
ax = salaries.groupby('VowelCounts')[['BasePay', 'TotalPay', 'Benefits']].agg('mean').plot()
# After you get the axes object back from the pandas command 
# you can further manipulate its characteristics
ax = salaries.groupby('VowelCounts')[['BasePay', 'TotalPay', 'Benefits']].agg('mean').plot()

# refine the axes object
ax.set_title("Pandas Helper method")
ax.set_ylabel("$")
ax.set_xticks([5, 10])


# i will be using the JobTitle feature so I need to fix the representation a bit
salaries['JobTitle'] = salaries.JobTitle.str.lower() 

# get the top ten occupations
top_ten_occupations = salaries.JobTitle.value_counts().sort_values(ascending=False).head(10).index
top_ten_occupations
# aggregate by job title and pick out the BasePay, Benefits, and Overtime features
salaries_averages_by_occupation = (salaries[salaries.JobTitle.isin(top_ten_occupations)]
                                   .groupby('JobTitle')[['BasePay', 'Benefits', 'OvertimePay']]
                                   .aggregate('mean')
)

ax = salaries_averages_by_occupation.plot(kind='barh', figsize=(8,8))

ax.set_xlabel('Mean Pay')
# the above graph can be transformed into a proportions stacked bar graph

# use the dataframe method div to proportionalize the values by axis=0(row)
salary_percents = salaries_averages_by_occupation.div(salaries_averages_by_occupation.sum(1), 
                                                      axis=0)

# and plot the bar graph with a stacked argument.  
ax = salary_percents.plot(kind='bar', stacked=True, rot=90)


# the axis is barely visible and you need to get rid of that white space on top

ax = salary_percents.plot(kind='bar', stacked=True, rot=90)
# only plot y in the 0-1 range
ax.set_ylim((0,1))

# fix the legend
legend = plt.legend(loc='best', framealpha=.5)
legend.get_frame().set_facecolor('#FFFFFF')

# sns sets frame_on to false, so change it back to True
legend.set_frame_on(True)
# use last name length for histogram
salaries['LastNameLength'] = (salaries
                              .EmployeeName
                              .str.split()
                              .apply(lambda a: len(a[-1]))
                              )
# simple enough with the data frame hist method
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
salaries.LastNameLength.hist(bins=20, ax=ax, alpha=.3)
# create a figure that contains two plots.
# the first plot shows a kde plot of salary proportions relative to the max salary
# the second one contains a histogram of all salaries

# set up the plotting surface
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# pass in the first axis to the series plot method
ax = (salaries.TotalPay / salaries.TotalPay.max()).plot(kind='kde', ax = ax1, title="Distribution of Proportions")
ax.set_xlim((0,1))

# pass in the second axis to the series plot method
ax = salaries.TotalPay.hist(bins=20, ax=ax2)
ax.set_title("Distribution of Total Pay")

# and adjust
ax.set_xticklabels(labels = ax.get_xticks(), rotation=90)

# to make a title that spans all subplots, use plt.suptitle
plt.suptitle("Distribution Plots")
plt.subplots_adjust(wspace=1)
pd.scatter_matrix(salaries[pay_columns], figsize=(8,8))
# can it be made from scratch?
def myScatterMatrix(df, bins = 20, diagonals='hist', **kwargs):
    
    # get number of columns of dtype int or float
    for col in df.columns:
        if df[col].dtype not in [np.dtype('int'), np.dtype('float')]:
            df = df.drop(col, axis=1)
            
    cols = df.columns
    n = len(cols)
    
    # use matplotlib subplots command to generate a subplot with
    # an arrangement of n_rows by n_cols
    fig, axes = plt.subplots(n, n)
    
    # the figure height and width can be arbitrary, it would 
    # be easier to let the user specify the size
    fig.set_figheight(n+3)
    fig.set_figwidth(n+3)
    
    # get all 2 combinations of features
    for i in range(n):
        for j in range(n):
            ax = axes[i,j]
            if i == j:    
                # if the columns are the same, then make a hist plot or kde plot depending on
                # depending on what diagonal is 
                (df[cols[i]].hist(bins = bins, ax=ax, **kwargs) if diagonals != 'kde' \
                                                      else df[cols[i]].plot(kind='kde', **kwargs))
            else:
                ax.scatter(x=df[cols[j]], y = df[cols[i]], **kwargs)
    
            
            # have axes ticks only on the right edge and bottom edge
            # use the get_yaxis or get_xaxis and set_visible to control this behavior
            if not j == 0:
                ax.get_yaxis().set_visible(False)
            else:
                # the ith variable
                ax.set_ylabel(cols[i])
            
            if not i == n - 1:
                ax.get_xaxis().set_visible(False)
            else:
                ax.set_xticklabels(ax.get_xticks(), rotation=90, fontsize='small')
                # the jth variable
                ax.set_xlabel(cols[j])
            
            
   
    # get rid of spacing for subplots
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    # return the array of subplots
    plt.show()
    return axes
myScatterMatrix(salaries[pay_columns])
