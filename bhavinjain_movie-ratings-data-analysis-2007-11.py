#importing packages



import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
movies = pd.read_csv('/kaggle/input/movieratings-20072011/Movie-Ratings.csv')
movies.head()
movies.info()
movies.describe()
movies.columns
#Renaming the columns



movies.columns = ['Film', 'Genre', 'CriticRatings', 'AudienceRatings', 'BudgetInMillions', 'ReleaseYear']
movies.head()
#Convert object columns to categories



movies.Film = movies.Film.astype('category') 

movies.Genre = movies.Genre.astype('category')

movies.ReleaseYear = movies.ReleaseYear.astype('category')
movies.info()
#List of Genre's



movies.Genre.cat.categories
#Histograms
#Critics v/s Audience rating pattern



fig, ax = plt.subplots(1,2,figsize= (10,6))

h1 = sns.distplot(movies.CriticRatings, ax=ax[0])

h2 = sns.distplot(movies.AudienceRatings, ax=ax[1])
#Displaying rows with movies of Action genre along with its Budget(million $)



movies[movies.Genre == "Action"].BudgetInMillions
#Stacked histogram



list1 = []

label1 = []

for i in movies.Genre.cat.categories:

    list1.append(movies[movies.Genre == i].BudgetInMillions)

    label1.append(i)



fig, ax = plt.subplots(figsize= (10,7))

plt.hist(list1, bins = 20, rwidth = 1, stacked = True, label = label1)

plt.xlabel('Budget In million ($)')

plt.legend()

plt.show()
#Scatter Plot



g = sns.lmplot(data = movies, x = 'CriticRatings', y = 'AudienceRatings', hue = 'Genre', height = 6, fit_reg = False)

g.fig.set_figwidth(10)
#KDE Plot
k1a = sns.kdeplot(movies.CriticRatings, movies.AudienceRatings, shade = True, shade_lowest=False, cmap = 'Blues')

k1b = sns.kdeplot(movies.CriticRatings, movies.AudienceRatings, cmap = 'Blues')
#Subplots for sophisticated visualization
sns.set_style('white')

f, axes = plt.subplots(1, 2, figsize = (12,5), sharex=True, sharey=True)

k1 = sns.kdeplot(movies.BudgetInMillions, movies.CriticRatings, ax = axes[0])

k2 = sns.kdeplot(movies.BudgetInMillions, movies.AudienceRatings, ax = axes[1])

k1.set(xlim = (-20,200), ylim = (-20,120))
#Violin Plots
f, axes = plt.subplots(1, 2, figsize = (12,5), sharex=True, sharey=True)

V1 = sns.violinplot(data = movies, x = 'Genre', y = 'CriticRatings', ax=axes[0])

V2 = sns.violinplot(data = movies, x = 'Genre', y = 'AudienceRatings', ax=axes[1])
f, axes = plt.subplots(1, 2, figsize = (12,5), sharex=True, sharey=True)

V3 = sns.violinplot(data = movies[movies.Genre=='Action'], x = 'ReleaseYear', y = 'CriticRatings', ax = axes[0])

v4 = sns.violinplot(data = movies[movies.Genre=='Action'], x = 'ReleaseYear', y = 'AudienceRatings', ax = axes[1])
#Facet Grid
#Scatter plot Grid



fg = sns.FacetGrid(movies, row = 'Genre', col = 'ReleaseYear', hue = 'Genre')

kws = dict(s = 50, linewidth = '0.5', edgecolor = 'black')

fg = fg.map(plt.scatter, 'CriticRatings', 'AudienceRatings', **kws)





#Adding diagonal for clear picture of Critics v/s Audience ratings

for i in fg.axes.flat:

    i.plot((0,100), (0,100), c='gray', ls = '--')
#Basic Dashboard
import warnings

warnings.filterwarnings('ignore')
sns.set_style('dark', {'axes.facecolor':'white'})

f, axes = plt.subplots(2, 2, figsize = (13,13))



#Plot 1

k1 = sns.kdeplot(movies.BudgetInMillions, movies.CriticRatings, shade = True,shade_lowest=False, cmap= 'inferno', ax = axes[0,0])

kk1 = sns.kdeplot(movies.BudgetInMillions, movies.CriticRatings, cmap = 'gray', ax =axes[0,0])



#Plot 2

k2 = sns.kdeplot(movies.BudgetInMillions, movies.AudienceRatings,shade = True,shade_lowest=False, cmap= 'inferno', ax = axes[0,1])

kk2 = sns.kdeplot(movies.BudgetInMillions, movies.AudienceRatings,cmap = 'gray', ax = axes[0,1])



k1.set(xlim=(-20,160))

k2.set(xlim=(-20,160))



#Plot 3

V2 = sns.violinplot(data = movies, x = 'ReleaseYear', y = 'BudgetInMillions', ax = axes[1,0], palette = 'Reds')



#Plot 4

k1a = sns.kdeplot(movies.CriticRatings, movies.AudienceRatings, shade = True, shade_lowest=False, cmap = 'Blues_r', ax =axes[1,1])

k1b = sns.kdeplot(movies.CriticRatings, movies.AudienceRatings, cmap = 'Blues', ax =axes[1,1])