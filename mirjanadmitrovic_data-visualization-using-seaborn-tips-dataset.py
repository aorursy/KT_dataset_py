# importing required libraries

import os #provides functions for interacting with the operating system

import numpy as np 

import pandas as pd

from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# loading dataset

tips_dataset='../input/seaborn-tips-dataset/tips.csv'

tips_raw = pd.read_csv(tips_dataset)
# make a copy of data 

tips=tips_raw.copy()
# run all the data

tips

# run the first 5 rows

tips.head()
# get number of rows and columns

tips.shape
# get attribute names 

tips.columns
# get information about a dataset (dtype, non-null values, memory usage)

tips.info()
# detect labels in categorical variables

for col in tips.columns[2:6]:

    print(col, np.unique(tips[col]))
# detect missing values

tips.isna().sum()
# Summary statistics

tips.describe() # only for numerical variables 

tips.describe().T # transpose

#tips.describe(include='all') # for all variables
# correlation matrix

tips.corr()
# Apsolute values - the number of records 



sns.countplot(x='sex', data=tips)

sns.despine() # no top and right axes spine



print(tips.sex.value_counts())
# change orientation, use same color for both label

sns.countplot(y='smoker', data=tips, color='b') 
# show value counts for two categorical variables

sns.countplot(x='sex', data=tips, hue='smoker', palette='viridis')
# show value counts for two categorical variables

sns.catplot(x='day', data=tips, hue='sex', palette='ch:.25', kind='count')
# facet along the columns to show a third categorical variable

sns.catplot(x='sex', hue='smoker', col='day', data=tips, kind='count')
# Relative values - the percentage of records

perc=tips['sex'].value_counts(normalize=True)*100

print(perc)

sns.barplot(x=perc.index, y=perc, data=tips)

sns.despine(left='True') # no top, left and right axes spine
fig, axes = plt.subplots(2, 2, figsize=(15,12)) # plot 4 graphs 



# histogram and density function, set title

sns.distplot(tips.total_bill, ax=axes[0,0]).set_title('Total_bill distribution')



#set number of bins and color, set title

sns.distplot(tips.total_bill, bins=50, color='r', ax=axes[0,1]).set_title('Total_bill distribution') 



# only histogram, without density function, set title

sns.distplot(tips.total_bill, kde=False, ax=axes[1,0]).set_title('Histogram') 



# only density function, without histogram, set title

sns.distplot(tips.total_bill, hist=False, ax=axes[1,1]).set_title('PDF of Total_bill')

sns.despine() # no top and right axes spine
fig, axes = plt.subplots(1, 2, figsize=(15,6)) # plot 2 graphs



# simple density function

sns.kdeplot(tips.total_bill, ax=axes[0])



# filled area under the curve, set color, remove legend, set title

sns.kdeplot(tips.tip, shade=True, color='purple', legend=False, ax=axes[1]).set_title('PDF of Tip') 
# detect the outliers



fig, axes = plt.subplots(1, 2,figsize=(15,6)) # plot 2 graphs



# use red color, set title

sns.boxplot(x='total_bill', data=tips, color='red', ax=axes[0]).set(title='Total_bill outliers') 



# change orientation, set title

sns.boxplot(x='tip', data=tips, orient='v', ax=axes[1]).set_title('Tip outliers') 
tips[tips.total_bill>=40]
tips[tips.tip>=6]
fig, axes = plt.subplots(1, 2, figsize=(15,6)) # plot 2 graphs 



# single horizontal violinplot

sns.violinplot(tips.total_bill, ax=axes[0])



# change orientation, set color

sns.violinplot(tips.tip, orient='v', color='red', ax=axes[1])
# line plot with confidence interval, set axes and title

sns.lineplot(x='size', y='tip', data=tips).set(xlabel='X axis- Size', ylabel='Y axis - Tip', title='Line plot - Size vs. Tip')
# show error bars and plot the standard error 

sns.lineplot(x='size', y='tip', hue='sex', data=tips, err_style='bars', ci=68)
fig, axes = plt.subplots(2, 2, figsize=(15,12)) # plot 4 graphs 



# simple scatter plot between two variables 

sns.scatterplot(x='total_bill', y='tip', data=tips, ax=axes[0,0])



# group by time and show the groups with different colors

sns.scatterplot(x ='total_bill', y ='tip', data = tips, hue= 'time', ax=axes[0,1])



# variable time by varying both color and marker

sns.scatterplot(x ='total_bill', y ='tip', data = tips, hue='time', style= 'time', ax=axes[1,0])



# vary colors and markers to show two different grouping variables

sns.scatterplot(x = 'total_bill', y = 'tip', hue= 'time', style= 'sex', data = tips)
sns.set(style='white') #set background



fig, axes = plt.subplots(2, 2, figsize=(15,12)) # plot 4 graphs 



# vary colors to show one grouping variable-size

sns.scatterplot(x='total_bill', y='tip', data=tips, hue='size', ax=axes[0,0])



# quantitative variable-size by varying the size of the points

sns.scatterplot(x='total_bill', y='tip', data=tips, hue='size', size='size', ax=axes[0,1])



# set the minimum and maximum point size and show all sizes in legend

sns.scatterplot(x='total_bill', y='tip', data=tips, hue='size', size='size', sizes=(10,200), ax=axes[1,0])



# vary colors and markers to show two different grouping variables -size,sex

sns.scatterplot(x='total_bill', y='tip', data=tips, hue='size', size='size', style='sex', sizes=(10,200), ax=axes[1,1])

sns.despine() 
# how could we use relplot instead of scatter plot



#sns.scatterplot(x='total_bill', y='tip', data=tips, hue='size', size='size', style='sex', sizes=(10,200))

sns.relplot(x='total_bill', y='tip', data=tips, hue='size', size='size', style='sex', sizes=(10,200))
sns.set(style='whitegrid') # set background for following graphs
# draw a single facet, set axes 

sns.relplot(x='total_bill', y='tip', hue='day', data = tips).set(xlabel='X - total_bill', ylabel='Y - tip')
# facet on the columns with another variable

sns.relplot(x='total_bill', y='tip', hue='day', col='time', data = tips)
# facet on the columns and rows

sns.relplot(x='total_bill', y='tip', hue='day', col='time', row='sex', data = tips)
sns.set(style='white') # set background and palette for following graphs
# scatterplot with marginal histograms

sns.jointplot(x='total_bill', y='tip', data=tips)
# add regression line and density function:

sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')
# replace the scatterplot with “hexbin” plot - shows the counts of observations that fall within hexagonal bins

sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex', color='purple')
# replace the scatterplot and histograms with density estimates

sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde', color='pink')
# bivariate densiti, more contour levels and a different color palette

sns.kdeplot(tips.total_bill,tips.tip, n_levels=30, cmap='Purples_d')
# 2 density function on same graph

# shade under the density curve, use a different color

sns.kdeplot(tips.total_bill,shade=True, color='b')

sns.kdeplot(tips.tip, shade=True, color='r')
# don't shade under the density curve, use a different color

female_tip=tips[tips['sex'] == 'Female'].tip.values

male_tip=tips[tips['sex'] == 'Male'].tip.values



sns.kdeplot(female_tip, color='red')

sns.kdeplot(male_tip, color='blue')
sns.set(style='ticks')



# simple linear relationship between two variables

sns.lmplot(x='total_bill', y='tip', data=tips)
# regression line without confidence interval

sns.lmplot(x='total_bill', y='tip', data=tips, ci=None)
# third variable, levels in different colors with markers

sns.lmplot(x='total_bill', y='tip', data=tips, hue='smoker', markers=['o','x'])
# facet on the columns and rows, set name of axes

sns.lmplot(x='total_bill', y='tip', data=tips, col='smoker',row='time').set_axis_labels('Total bill (in $ )', ' Tip ( in $ )')
# visualize the correlation matrix

sns.heatmap(data=tips.corr(),annot=True) # values of Pearson coefficient
# simple paiplot - all numerical varibles, scatter plots and histograms on diagonal

sns.pairplot(tips)
# select wanted variables

sns.pairplot(tips[['total_bill','tip']])  
# fit linear regression models to the scatter plots, show density plots on diagonal

sns.pairplot(tips, kind='reg', diag_kind="kde")  
# set markers for different levels

sns.pairplot(tips, hue='sex', markers=["+", "o"])  
# facets on column and row

g = sns.FacetGrid(tips, col='time',  row='smoker')

g = g.map(plt.scatter, 'total_bill', 'tip', edgecolor='w').set_titles('Scatter-plot')
# facets on column, with hue represent levels of a variable in different colors

g=sns.FacetGrid(tips, col='time',  hue='smoker')

g = (g.map(plt.scatter, 'total_bill', 'tip', edgecolor='w').add_legend())
pal = dict(Lunch='seagreen', Dinner='gray')



g = sns.FacetGrid(tips, col='sex', hue='time', palette=pal, hue_order=['Dinner', 'Lunch'])

g = (g.map(plt.scatter, 'total_bill', 'tip').add_legend())
sns.set(style='darkgrid',palette='Set2') # set background and palette



# grouped by a categorical variable

sns.pointplot(x='day', y='tip', data=tips)
# grouped by a two variables,separate lines

sns.pointplot(x='day', y='tip', data=tips, hue='smoker', dodge=True)
# separate the points for different hue levels with different marker and line style

sns.pointplot(x='day', y='tip', data=tips, hue='smoker', dodge=True, markers=['o','x'], linestyles=['dotted','--'])
# show standard deviation of observations instead of a confidence interval

sns.pointplot(x='tip', y='day', data=tips, ci='sd')
sns.set(style='whitegrid')# set background



fig, axes = plt.subplots(2, 2, figsize=(15,8)) # plot 4 graphs



# grouped by a categorical variable

sns.barplot(x='day', y='tip', data=tips, ax=axes[0,0])



# all bars in a single color

sns.barplot(x='day', y='tip', data=tips, color='salmon', saturation=.8, ax=axes[0,1])



# grouped by a two variables, show standard deviation of observations instead of a confidence interval

sns.barplot(x='day', y='tip', data=tips, hue='sex', ci='sd', ax=axes[1,0])



# grouped by new variable 

tips['weekend'] = tips['day'].isin(['Sat', 'Sun'])

sns.barplot(x='day', y='total_bill', hue="weekend", data=tips, dodge=False, ax=axes[1,1]) 
fig, axes = plt.subplots(2, 2, figsize=(15,8)) # plot 4 graphs



# the quartile information for a numerical column grouped by categorical column

sns.boxplot(x='time', y='total_bill', data = tips, ax=axes[0,0])



# set unique categorical label in different order

sns.boxplot(x='time', y='total_bill', data=tips,order=['Dinner', 'Lunch'], ax=axes[0,1])



# boxplot with swarmplot

sns.boxplot(x = 'time', y = 'total_bill', data = tips, ax=axes[1,0])

sns.swarmplot(x = 'time', y = 'total_bill', data = tips, color='.25', ax=axes[1,0])



# boxplot with nested grouping by two categorical variables and ticker line

sns.boxplot(x='time', y='total_bill', data = tips, hue='day', linewidth=2.5, ax=axes[1,1])
fig, axes = plt.subplots(3, 2, figsize=(15,12)) # plot 6 graphs



# vertical violinplot grouped by a categorical variable

sns.violinplot(x='day', y='total_bill', data=tips, ax=axes[0,0])



# vertical violinplot grouped by a categorical variable, set palette

sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', palette='muted', ax=axes[0,1])



# split violins to compare the across the hue variable

sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', palette='muted', split=True, ax=axes[1,0]) 



# show each observation with a stick inside the violin

sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', palette='muted', split=True, inner='stick', ax=axes[1,1])



# scale the violin width by the number of observations in each bin

sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', palette='muted', split=True, scale='count', ax=axes[2,0]) 



#Scale the density relative to the counts across all bins

sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', palette='muted', split=True, scale='count', inner='stick', ax=axes[2,1])
fig, axes = plt.subplots(2, 2, figsize=(15,12)) # plot 4 graphs



sns.swarmplot(x='time', y='tip', data=tips, ax=axes[0,0])



sns.swarmplot(x='time', y='tip', data=tips, hue='sex', ax=axes[0,1])



sns.swarmplot(x='time', y='tip', data=tips, hue='sex', dodge=True, ax=axes[1,0]) 



# combine swarm and violin plot

sns.swarmplot(x='time', y='tip', data=tips, color='k', ax=axes[1,1])

sns.violinplot(x='time', y='tip', data=tips, inner=None, ax=axes[1,1])
# simple relationship between a numerical and a categorical variable  

sns.catplot(x='day', y='total_bill', data=tips)
# facet on column, grouped by 2 variables

sns.catplot(x='day', y='tip', data=tips, hue='size', col='sex')
# facet on column, grouped by 2 variables

sns.catplot(x='day', y='total_bill', data=tips, hue='size', col='time', row='sex' )
# observations in one line

sns.catplot(x='day', y='tip', data=tips, hue='sex', jitter=False, alpha=.4)
# use a different plot kind to visualize the same data

sns.catplot(x='sex', y='total_bill',hue='smoker', col='time', data=tips, kind='point', dodge=True, height=4, aspect=.7)
# use a different plot kind to visualize the same data

sns.catplot(x='sex', y='total_bill', data=tips, hue='smoker', col='day', kind='bar')
# use a different plot kind to visualize the same data

sns.catplot(x='time', y='tip', data=tips, color='k', height=3, kind='swarm')
# use a different plot kind to visualize the same data

sns.catplot(x='time', y='tip', data=tips, kind='boxen')
# univariate plot on each facet

g = sns.FacetGrid(tips, col='time',  row='smoker')

g = g.map(plt.hist, 'total_bill', color='green').set_titles('Histogram')
# specify the order, change the height and aspect ratio of each facet

bins = np.arange(0, 65, 5)

g = sns.FacetGrid(tips, col='smoker', col_order=['Yes', 'No'], height=4, aspect=.5)

g = g.map(plt.hist, 'total_bill', bins=bins, color='m').add_legend()