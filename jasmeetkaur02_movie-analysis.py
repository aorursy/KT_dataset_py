import pandas as pd #importing panda library
import os 
os.chdir('../input') #setting the working directory
movies = pd.read_csv("Movie-Ratings.csv")
movies
len(movies) #gives the no. of rows ie in this case is the 559 hollywood movies that we are analyzing
movies.head() #displays the first 5 rows of the data set
movies.columns #displays all the column names
movies.columns = ['Film', 'Genre', 'CriticRatings', 'AudienceRatings',
       'BudgetMillions', 'Year'] # replacing some of the column names
movies.head()# again displays all the column names with the changed headings
movies.Film = movies.Film.astype("category") #film, genre, year we are changing into categorical variables
movies.Genre = movies.Genre.astype("category")
movies.Year = movies.Year.astype("category")
movies.info() #tells the detailed information about the dataset
movies.Genre.cat.categories #tells the kind of genre
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore') #use to filter out unnecessary warnings as ignore
#Chart 1
#Joint plots
#It is a type of scatter plot
j = sns.jointplot(data=movies, x='CriticRatings', y='AudienceRatings') #different types of plotting done
#Chart 1.2
#Cluster the above plotted dots
j = sns.jointplot(data=movies, x='CriticRatings', y='AudienceRatings',kind='hex')
#Histogram
#Chart 2
m1 = sns.distplot(movies.AudienceRatings, bins=16)#Audience rating is uniformally distributed, sometimes audiences like big dumb comedies, and critics don't.
m2 = sns.distplot(movies.CriticRatings, bins=16)#There are a lot of factors on which the critic rating actually depends on, critic relies on the judgement, they rely on the quality of the movie by looking at the work of the director, actors, story and certain other parameters like camera work, audio, video whereas the audience does not go into that much detail,therefore we can see that critic rating is not uniformally distributed.
#Chart 3
#Plotting budget of all the movies
plt.hist(movies.BudgetMillions)
plt.show()
#Plotting budget of movies based on Genre
plt.hist(movies[movies.Genre=='Action'].BudgetMillions)
plt.show() #with action genre movies, budget varies like this
plt.hist(movies[movies.Genre=='Drama'].BudgetMillions)
plt.show()#with drama genre movies,budget varies like this
#Plotting all the data separately in one graph
plt.hist(movies[movies.Genre=='Action'].BudgetMillions,bins=10)
plt.hist(movies[movies.Genre=='Drama'].BudgetMillions,bins=10)
plt.hist(movies[movies.Genre=='Thriller'].BudgetMillions,bins=10)
plt.show() #combined all the 3 histograms  of action,drama and thriller movies
#Chart 4.1
list1=list()
mylabels = list()
for gen in movies.Genre.cat.categories: #for loop used to plot all the genre related budget movies
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
                    fit_reg=False,hue='Genre')# linear plotting of audience and critic ratings
#KDE Plot kernel density estimate
k1=sns.kdeplot(movies.CriticRatings,movies.AudienceRatings)
#compare density of data in above 2 
k1=sns.kdeplot(movies.CriticRatings,movies.AudienceRatings,shade=True)
#combined above 2
k1=sns.kdeplot(movies.CriticRatings,movies.AudienceRatings)
k1=sns.kdeplot(movies.CriticRatings,movies.AudienceRatings,shade=True)
k1=sns.kdeplot(movies.BudgetMillions,movies.AudienceRatings) 
k1=sns.kdeplot(movies.CriticRatings,movies.AudienceRatings,shade=True)
k1=sns.kdeplot(movies.BudgetMillions,movies.AudienceRatings,shade=True)#•	Audience ratings does not depend on the budget of the movies,
#therefore we can see that it has got the audience rating as minimum as 25 and maximum as 90.
k1=sns.kdeplot(movies.BudgetMillions,movies.CriticRatings)
k1=sns.kdeplot(movies.BudgetMillions,movies.CriticRatings,shade=True)
f,axes=plt.subplots(1,2)
k1=sns.kdeplot(movies.BudgetMillions,movies.AudienceRatings,ax=axes[0])
k2=sns.kdeplot(movies.BudgetMillions,movies.CriticRatings,ax=axes[1])# plotting the two graphs together
#data visulization using 6 box plot and violinplots
w = sns.boxplot(data=movies, x='Genre',y='CriticRatings')
z=sns.violinplot(data=movies, x='Genre',y='CriticRatings')
#data visulization using 7 Facet Grids
q=sns.FacetGrid(movies,row='Genre',col='Year', hue='Genre')
q=q.map(plt.scatter,'CriticRatings', 'AudienceRatings')
#Similarly different types of plots can be made
