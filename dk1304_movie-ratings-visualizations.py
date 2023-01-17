# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import os
print(os.listdir("../input"))
movies = pd.read_csv("../input/Movie-Ratings.csv")
movies
len(movies)
movies.columns
movies.columns = ['Film', 'Genre', 'CriticRatings', 'AudienceRatings',
       'BudgetMillions', 'Year']
movies.Film = movies.Film.astype("category")
movies.Genre = movies.Genre.astype("category")
movies.Year = movies.Year.astype("category")
movies.info()
movies.Genre.cat.categories
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#It is a type of scatter plot
j = sns.jointplot(data=movies, x='CriticRatings', y='AudienceRatings')
#Chart 1.2
#Cluster the above plotted dots
j = sns.jointplot(data=movies, x='CriticRatings', y='AudienceRatings',kind='hex')
#Histogram
#Chart 2
m1 = sns.distplot(movies.AudienceRatings, bins=16)
m2 = sns.distplot(movies.CriticRatings, bins=16)
#Chart 3
#Plotting budget of all the movies
plt.hist(movies.BudgetMillions)
plt.show()
#Plotting budget of movies based on Genre
plt.hist(movies[movies.Genre=='Action'].BudgetMillions)
plt.show()
plt.hist(movies[movies.Genre=='Drama'].BudgetMillions)
plt.show()
#Plotting all the data separately in one graph
plt.hist(movies[movies.Genre=='Action'].BudgetMillions,bins=10)
plt.hist(movies[movies.Genre=='Drama'].BudgetMillions,bins=10)
plt.hist(movies[movies.Genre=='Thriller'].BudgetMillions,bins=10)
plt.show()#
#Chart 4.1
list1=list()
mylabels = list()
for gen in movies.Genre.cat.categories:
    list1.append(movies[movies.Genre==gen].BudgetMillions)
    mylabels.append(gen)
h=plt.hist(list1, label=mylabels)
plt.legend()
plt.show()
#Chart 4.1
list2=list()
mylabels = list()

for gen in movies.Genre.cat.categories:
    list2.append(movies[movies.Genre==gen].BudgetMillions)
    mylabels.append(gen)
h=plt.hist(list2, bins=30,stacked=True,label=mylabels)
plt.legend()
plt.show()
#Chart 5 KDE
visual = sns.lmplot(data=movies, x='CriticRatings', y='AudienceRatings', \
                    fit_reg=False,hue='Genre')
#KDE Plot kernel density estimate
k1=sns.kdeplot(movies.CriticRatings,movies.AudienceRatings)
#compare density of data in above 2 
k1=sns.kdeplot(movies.CriticRatings,movies.AudienceRatings,shade=True)
#combined above 2
k1=sns.kdeplot(movies.CriticRatings,movies.AudienceRatings)
k1=sns.kdeplot(movies.CriticRatings,movies.AudienceRatings,shade=True)
k1=sns.kdeplot(movies.BudgetMillions,movies.AudienceRatings)
k1=sns.kdeplot(movies.BudgetMillions,movies.AudienceRatings,shade=True)
k1=sns.kdeplot(movies.BudgetMillions,movies.CriticRatings)
k1=sns.kdeplot(movies.BudgetMillions,movies.CriticRatings,shade=True)
f,axes=plt.subplots(1,2)
k1=sns.kdeplot(movies.BudgetMillions,movies.AudienceRatings,ax=axes[0])
k2=sns.kdeplot(movies.BudgetMillions,movies.CriticRatings,ax=axes[1])
#chart 6 box plot and violinplots
w = sns.boxplot(data=movies, x='Genre',y='CriticRatings')
z=sns.violinplot(data=movies, x='Genre',y='CriticRatings')
#Chart 7 Facet Grids
q=sns.FacetGrid(movies,row='Genre',col='Year', hue='Genre')
q=q.map(plt.scatter,'CriticRatings', 'AudienceRatings')
#Similarly different types of plots can be made