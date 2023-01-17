##IMPORTING NECESSARY LIBRARIES AND OUR DATASET

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import cufflinks as cf

cf.go_offline()

import seaborn as sns

import ipywidgets as widgets

from ipywidgets import interact, interactive, fixed, interact_manual

sns.set_style('darkgrid')

df1 = pd.read_csv('../input/Movie-Ratings.csv')
##EXPLORING OUR DATAFRAME

##Lets look at the size of our dataframe

df1.shape

#It has 599 Rows and 6 Columns
df1.info()
df1.head()
##FOR CONSISTENCY LETS RENAME OUR COLUMNS AND SET CATEGORICAL VALUES WHERE NECESSARY

df1.columns = ['Film', 'Genre', 'CriticRating', 'AudienceRating',

       'BudgetInMillions','Year']

df1.Film = df1.Film.astype('category')

df1.Genre = df1.Genre.astype('category')

df1.Year = df1.Year.astype('category')
#LETS TAKE A LOOK AT THE BASIC STATS OF OUR DATASET

df1.describe()
sns.heatmap(df1.corr(), annot = True , cmap = 'Greens',linewidths=2)
##LETS START PLOTTING

viz1 = sns.distplot(df1.AudienceRating)
viz2 = sns.distplot(df1.CriticRating)
@interact

def k9(x = 'Year' , y = ['AudienceRating','CriticRating'], Genre = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance',

       'Thriller']):

    sns.boxplot(data = df1[df1.Genre == Genre], x=x,y=y)
@interact

def exp1(x = ['CriticRating', 'AudienceRating', 'BudgetInMillions'],

         y = ['AudienceRating','BudgetInMillions'],):

    df1.iplot(x=x, y=y , kind = 'scatter' , mode='markers',theme = 'ggplot', colorscale = 'ggplot', size  = 5)
##########Lets Take a look at the Kernel Density Estimate

f , axes = plt.subplots(figsize = (12,6) , sharey = True)

plt.subplot(1,2,1)

k1 = sns.kdeplot(df1.BudgetInMillions, df1.AudienceRating )

plt.subplot(1,2,2)

k2 = sns.kdeplot(df1.BudgetInMillions, df1.CriticRating)

plt.show()
##################FACET GRIDS##############

g = sns.FacetGrid(df1 , row = 'Genre' , col = 'Year', hue = 'Genre')

kws = dict(s=50 , linewidth = 0.5, edgecolor = 'black')

g = g.map(plt.scatter, 'CriticRating' , 'AudienceRating' , **kws)

#Setting Axes Limits

g.set(xlim = (0,100) , ylim = (0,100))

#Adding Diagnols

for ax in g.axes.flat:

    ax.plot((0,100),(0,100) , c = 'gray' , ls = '--')

g.add_legend()

plt.show()