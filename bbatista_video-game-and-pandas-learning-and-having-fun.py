# Reading the data

import pandas as pd

ign = pd.read_csv(r'../input/ign.csv')

ign.head()
# Let's see some descriptions of our file

# Starting with data types

ign.dtypes
# We can use .info () to check more detailed dataset information such as number of columns, number of rows, column names ...

ign.info()
# It seems that the 'url' and 'unnamed: 0' columns will not tell us much in an exploratory analysis. So let's take them off.

ign.drop( columns = ['url', 'Unnamed: 0'], inplace = True)



ign.head(10)
#Checking empty cells in df. We have 36 null values  36 in the 'genre' column.

ign.isnull().sum()


# To find the games without genre fill in these fields with a value of 0

ign['genre'].fillna(0, inplace = True)
# Let's confirm that all the fields in the 'genre' column are now filled...


ign['genre'].isnull().sum()
# Let's look at the average score for non-genre games

ign[ign.genre == 0]['score'].mean()
## Lets see the df qualitative variables

ign[ign.genre == 0].describe(include = object)

# They seem to be games on a low note. 
# And how few are, we ignore them for now
# Let's see a statistical summary of DF using the .describe () method.

ign.describe()

# The ten worst games

ign[['title','score','genre', 'release_year', 'platform']].sort_values(by = 'score').head(10)
# Top ten games

ign[['title','score','genre', 'release_year', 'platform']].sort_values(by = 'score', ascending = False).head(10)
# The method describes, it also allows you to extract very useful information from the data in text format (object, string)
ign.describe(include = object)


top_plataformas = pd.value_counts(ign['platform'], ascending = False).head(10)

top_plataformas

ios = ign[ign['platform'] == 'iPhone']

ios.head(10)
# The top ten iphone games

ios.sort_values(by = 'score', ascending = False).head(10)
# Let's take a look at how is the proportional distribution of iphone games according to the classification sentence


cla = pd.value_counts(ios['score_phrase'], ascending = False, normalize = True)

cla
# To get more visual, I think it would be a good idea to show this distribution on a chart!
# we can do this using the matplotlib library

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib notebook

cla.plot(kind = 'bar', title = 'Score Phrase', fontsize = 8, rot = 45, figsize = (10,10), grid = False)
# Most Popular Genres on iPhone

%matplotlib notebook

igen = pd.value_counts(ios['genre'], ascending = False).head(10)

igen.plot(kind = 'barh', title = 'Most Popular Genres',rot = 45, fontsize = 7, figsize = (10,10), grid = False)

# here you can view the average score per year of iphone games

%matplotlib notebook

year = ios.groupby('release_year')['score'].mean().transpose()

year.plot(kind = 'bar', title = 'Average Score per Year', fontsize = 7, rot = 45, grid = False)

# plotting the amount of iphone ratings per year

%matplotlib notebook


ios_resumo = pd.pivot_table(ios, values = 'score', index = 'score_phrase', 
                             aggfunc = 'count', columns = 'release_year')


ios_resumo.plot(kind = 'bar', subplots = True, figsize = (10,8), legend = True, title = 'iPhone', layout = (3,3),
sharey = True, sharex = False, grid = False, fontsize = 5, rot = 45)
ios['score'].mean()
# First, we created a DF for each platform 

xb360 = ign[ign.platform == 'Xbox 360']
ps3 = ign[ign.platform == 'PlayStation 3']

# Then we use the pd.concat () function to join all in a single DataFrame

duelo = pd.concat([xb360,ps3], axis = 0)

duelo.head(10)
    

# Let's see the average rating of each platform

%matplotlib notebook

duelo.groupby(['platform'])['score'].mean().plot(kind = 'bar', title = 'Average Score', fontsize = 8, rot = 0, grid = False,
                                                figsize = (10,10))

# The games have basically the same average
%matplotlib notebook
duelo.pivot_table('score', index = ['score_phrase'], columns = ['platform'], aggfunc = 'count').plot(kind = 'bar',
grid = False, title = "Classificação Por Platafroma", rot = 45, fontsize = 8, figsize = (10,7))
                                                                                                     
pd.crosstab(duelo.editors_choice, duelo.platform).plot(kind = 'bar', title = 'Editors Choice', rot = 0, figsize = (10,10), grid = False)
# I disregarded the Walking Dead for xbox, its release year is 1970 
duelo[duelo['release_year'] == 1970]
duelo.drop(516, inplace = True)
%matplotlib inline

editores2 = duelo[duelo['editors_choice'] == 'Y'][['platform', 'release_year','score']]

pd.pivot_table(editores2, values = 'score', index = 'release_year', columns = 'platform', aggfunc = 'count',
              fill_value = 0).plot(kind = 'bar', figsize = (15,8), grid = False, rot= 45, title = "Editors Choice by year")
%matplotlib notebook

duelo.platform.value_counts().plot(kind = 'bar', title = 'Released Games', grid = False, figsize = (15,8), rot = 0)
pd.crosstab(duelo.release_year, duelo.platform).plot(kind = 'bar', grid = False, title = 'Released Games by year',
                                                    figsize = (17,8))
duelo.groupby(by = 'platform')['editors_choice'].value_counts(normalize = True)

pd.crosstab(duelo.platform, duelo.editors_choice, normalize = True, margins = True)
