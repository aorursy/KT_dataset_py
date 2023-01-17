# imports



import pandas as pd

import numpy as np

import os

import urllib

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
# Was required for Colab



# !pip install -q kaggle
# Was required for Colab



# Download Dataset from kaggle



# os.environ['KAGGLE_USERNAME'] = 'tj2807'

# os.environ['KAGGLE_KEY'] = '9af863cab313e18021aa8051e9c6ded1'



# !kaggle datasets download -d zynicide/wine-reviews
# Was required for Colab

# importing required modules 

# from zipfile import ZipFile 

  

# # specifying the zip file name 

# file_name = "/content/wine-reviews.zip"

  

# # opening the zip file in READ mode 

# with ZipFile(file_name, 'r') as zip: 

#     # printing all the contents of the zip file 

#     zip.printdir() 

  

#     # extracting all the files 

#     print('Extracting all the files now...') 

#     zip.extractall() 

#     print('Done!') 
reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col = 0)
reviews.head(3)
# Get absolute numbers for each province

reviews.province.value_counts().head(10).plot.bar()
# what percentage though



# Each bar is a category - Thus suitable for representing stats about

# categorical data.



# Bar chart height can represent anything, as long as it is a number.



# Here province is a nominal variable.



(reviews.province.value_counts().head()/len(reviews)*100).plot.bar()
# Bar Charts however can also be used for ordinal categories. In our case

# scores for each wine are between 80-100, thus even though it's a number, 

# they are ordinal category variables.



reviews.points.value_counts().sort_index().plot.bar()
# Line charts are typically useful when there are too many categories

# Line charts however do not make sense for nominal data. Line charts

# mush the values together and order is implicit in line charts. 



reviews.points.value_counts().sort_index().plot.line()
# Area charts are just line charts with area shaded in case of single variable plot.



reviews.points.value_counts().sort_index().plot.area()
reviews[reviews.price < 200]['price'].plot.hist()
# Histograms however face a problem with skewed data. When the data is skewed, 

# since histogram divides input space in uniform intervals, distribution may

# not fir right without normalization. 



# This also makes histograms a very good way to see if data is skewed and decide

# how to normalize.



reviews.price.plot.hist()
# Histograms work really well with ordinal variables as well.



reviews.points.plot.hist()
reviews[reviews.price<100].sample(100).plot.scatter(x='price',y='points')
# From above plot, we can estimate that there is a weak correlation between

# price and points. We had to sample 100 values because there's too much 

# overlapping data



reviews[reviews.price<100].plot.scatter(x='price', y= 'points')
# We took price < 100 because there are too many outliars which will make the 

# scale problematic



reviews.plot.scatter(x='price',y='points')
reviews[reviews.price < 100].plot.hexbin(x='price',y='points', gridsize = 15)
# Converting the data to 2 dimensions and counts. This is a standard format to

# many multivariate plot functions in pandas.



filtered = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]

wine_counts = filtered.groupby(['points','variety']).country.count().unstack()
wine_counts.plot.bar(stacked=True)
wine_counts.plot.area()
wine_counts.plot.line()
# regular bar chart



reviews.points.value_counts().sort_index().plot.bar()
# figsize for overall plot size



reviews.points.value_counts().sort_index().plot.bar(figsize = (12,6))



# figsize takes the image size in inches, it takes (width,height) values.
# color and legend font size







reviews.points.value_counts().sort_index().plot.bar(figsize = (12,6),

                                                    color='mediumvioletred',

                                                    fontsize = 16)
# Add a title



reviews.points.value_counts().sort_index().plot.bar(figsize = (12,6),

                                                    color='mediumvioletred',

                                                    fontsize = 16,

                                                   title = "No. of reviews for each score")
# Pandas plot functions and its paramters are built on matplotlib and act as 

# an abstraction layer. Plot can also be modified using matplotlib directly.





myPlot = reviews.points.value_counts().sort_index().plot.bar(figsize = (12,6),

                                                    color='mediumvioletred',

                                                    fontsize = 16)



myPlot.set_title('No. of reviews for each score', fontsize = 16)



# This is useful since pandas hasn't included all the customization

# functionality that matplotlib provides. For eg. only with pandas we cannot 

# set the font size of title.
# seaborn works along with these libraries



myPlot = reviews.points.value_counts().sort_index().plot.bar(figsize = (12,6),

                                                    color='mediumvioletred',

                                                    fontsize = 16)



myPlot.set_title('No. of reviews for each score', fontsize = 16)

sns.despine(bottom=True, left=True)
fig, axrr = plt.subplots(2,1, figsize = (12,8))



# sibplots method takes rows and columns as argument.

# fig has the figure object now.

# axrr is an array of axes subplot objects for both figures.

# axrr is an array of both axes subplot objects



axrr
# in order to tell pandas where to plot, we need to specify ax attribute which

# takes axws subplot object type.

fig, axrr = plt.subplots(2,1, figsize = (12,8))

reviews.points.value_counts().sort_index().plot.bar(ax = axrr[0])

axrr[0].set_title('No. of reviews with given points')
# Thus each individual subplot can be referred by considering the top left point as origin.



fig, axarr = plt.subplots(2,2, figsize = (12,10))



reviews['points'].value_counts().sort_index().plot.bar(

    ax=axarr[0][0], fontsize=12, color='mediumvioletred'

)

axarr[0][0].set_title("Wine Scores", fontsize=18)



reviews['variety'].value_counts().head(20).plot.bar(

    ax=axarr[1][0], fontsize=12, color='mediumvioletred'

)

axarr[1][0].set_title("Wine Varieties", fontsize=18)



reviews['province'].value_counts().head(20).plot.bar(

    ax=axarr[1][1], fontsize=12, color='mediumvioletred'

)

axarr[1][1].set_title("Wine Origins", fontsize=18)



reviews['price'].value_counts().plot.hist(

    ax=axarr[0][1], fontsize=12, color='mediumvioletred'

)

axarr[0][1].set_title("Wine Prices", fontsize=18)



plt.subplots_adjust(hspace=0.3) # Gap between rows



sns.despine()
# pandas bar plot is basically count plot in seaborn



sns.countplot(reviews.points)



# Note that you do not need to pass value counts to the plot, it automatically

# takes care of the same. Very intuitive.
sns.kdeplot(reviews.query('price < 200').price)
# alternative line plot :



reviews[reviews.price < 200].price.value_counts().sort_index().plot.line()



# Thus it can be observed that kde plot give the true shape line chart data.
# Kde plot can also be plotted for 2d data



# Note that bivariate KDE plots are very computationally expensive. 

# This is the reason why we sample 5000 points here.

# sns.kdeplot(reviews[reviews['price'] < 200].loc[:, ['price', 'points']].dropna().sample(5000))



# Better way to do bivariate

tempFrame = reviews[reviews.price < 200].dropna().sample(5000)

sns.kdeplot(tempFrame.price, tempFrame.points )
sns.distplot(reviews['points'], bins=10, kde=False)
sns.jointplot(x='price', y='points', data = reviews[reviews.price < 100])
sns.jointplot(x='price', y='points', data = reviews[reviews.price < 100], kind='hex', gridsize=20)
myData = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]

myData.variety.value_counts()
sns.boxplot(x = 'variety', y = 'points', data = myData)
# Violin plot shows the same data that boxplot doesn, but it replaces the box in boxplot with kernel density estimation of the data.

myData = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]

sns.violinplot(x = 'variety', y= 'points', data = myData)
pokemons = pd.read_csv('../input/pokemon/Pokemon.csv', index_col = 0)

pokemons.head()
sns.countplot(pokemons.Generation)
sns.distplot(pokemons.HP, kde = True)
sns.jointplot(x = 'Attack', y = 'Defense', data = pokemons)
sns.jointplot(x = 'Attack', y = 'Defense', data = pokemons, kind = 'hex', gridsize = 20)
sns.kdeplot(pokemons['HP'], pokemons['Attack'])
sns.boxplot(x = 'Legendary', y= 'Attack', data = pokemons)
sns.violinplot(x = 'Legendary', y= 'Attack', data = pokemons)
# Loading footballers stats



df = pd.read_csv('../input/fifa-18-demo-player-dataset/CompleteDataset.csv', index_col=0)



footballers = df.copy()

footballers['Unit'] = df['Value'].str[-1]

footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0, 

                                    footballers['Value'].str[1:-1].replace(r'[a-zA-Z]',''))

footballers['Value (M)'] = footballers['Value (M)'].astype(float)

footballers['Value (M)'] = np.where(footballers['Unit'] == 'M', 

                                    footballers['Value (M)'], 

                                    footballers['Value (M)']/1000)

footballers = footballers.assign(Value=footballers['Value (M)'],

                                 Position=footballers['Preferred Positions'].str.split().str[0])

footballers.head()
# Suppose we want to get a kde plot for overall score. This will basically show us the count (actually KDE of probability mass function) of players getting a particular score.



sns.kdeplot(footballers.Overall)
# in this example we are gonna plot multiple facets for overall variable split based on position of the player.



# Let us take only two positions for now



df = footballers[footballers.Position.isin(['ST', 'GK'])]

g = sns.FacetGrid(df, col = 'Position')



# This makes a FacetGrid object which keeps blank facets ready for any particular dataframe to plot any x axis variable.

# Now we use FacetGrid map method to map facetGrid object with plotting function and x axis variable.

df = footballers[footballers.Position.isin(['ST', 'GK'])]

g = sns.FacetGrid(df, col = 'Position')

g.map(sns.kdeplot, 'Overall')
# That's super simple and super useful! We can plot all the positions as well, with wrap on.



g = sns.FacetGrid(footballers, col = 'Position', col_wrap=6)

g.map(sns.kdeplot, 'Overall')
# Facegrid actually let's us do the splitting of facets according to different combinations of two 

# categorical variables. These variables are mentioned in row and col attribute while 

# creating facetGrid object.

df = footballers[footballers.Position.isin(['ST', 'GK'])]

df = df[df.Club.isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]



# We can specify row and column order as well.



g = sns.FacetGrid(df, row = 'Position', col = 'Club', row_order=['GK', 'ST'],

                  col_order=['Atlético Madrid', 'FC Barcelona', 'Real Madrid CF'])

g.map(sns.violinplot, 'Overall')



# Only problem with faceting is that in order to avoid plots becoming too small, we can split facets

# only for about 2 categorical variables with limited no. of categories each.
sns.pairplot(footballers.loc[:, ['Overall', 'Potential', 'Value']])



# At the diagonal, every variable gets plotted against itself, so it's a histogram.

# At other place it has scatter plot.



# Pairplot is often the first visualization tool DS uses to visualize the data.
g = sns.FacetGrid(pokemons, row = 'Legendary')

g.map(sns.kdeplot, 'Attack')
g = sns.FacetGrid(pokemons, row = 'Generation', col = 'Legendary')

g.map(sns.kdeplot, 'Attack')
sns.pairplot(pokemons.loc[:, ['Defense', 'Attack', 'HP']])
footballers.head()
footballers.head()
# Suppose we want to see how different kinds of offensive players are paid [Value] 

# as per their overall rating [Score]]



# sns.lmplot(footballers, x = 'Value', y = 'Overall', hue = 'Position', 

#           data = footballers[footballers.Position.isin(['ST', 'RW', 'LW'])])



sns.lmplot('Value','Overall', footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])],

           fit_reg=False, hue='Position')
# We can also use different markers to distinguish



sns.lmplot('Value','Overall', footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])],

           fit_reg=False, hue='Position', markers = ['*','x', 'o'])
# Suppose we want to see how goalkeepers are scored on agression as compared to Strikers.

# We also want to know this information for player with different overall scores.



f = (footballers

        .loc[ footballers.Position.isin(['GK', 'ST'])]

        .loc[:, ['Overall', 'Aggression', 'Position']])

f = f.loc[f.Overall >= 80]

f = f.loc[f.Overall < 85]

f['Aggression'] = f['Aggression'].astype(float)

sns.boxplot(x = 'Overall', y = 'Aggression', hue = 'Position', data = f)



# This is using basically another visual variable "Grouping"
# Let's find correlation between certain columns



f = (footballers.loc[:, ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control']]

                .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)

                .dropna()).corr()

sns.heatmap(f, annot = True)
from pandas.plotting import parallel_coordinates



f = (

    footballers.iloc[:, 12:17]

        .loc[footballers['Position'].isin(['ST', 'GK'])]

        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)

        .dropna()

)

f['Position'] = footballers['Position']

f = f.sample(200)



parallel_coordinates(f, 'Position')
sns.lmplot('Attack', 'Defense', pokemons, hue = 'Legendary', markers = ['x','o'], fit_reg=False)
sns.boxplot(x = 'Generation', y = 'Total', hue = 'Legendary', data = pokemons)
pokemons.head()

sns.heatmap(pokemons.loc[:, ['HP', 'Attack', 'Sp. Atk', 'Defense', 'Sp. Def', 'Speed']].corr(), annot = True)
from pandas.plotting import parallel_coordinates



p = (pokemons.loc[pokemons['Type 1'].isin(['Fighting', 'Psychic'])]

     .loc[:,['Attack', 'Sp. Atk', 'Defense', 'Sp. Def', 'Type 1']])



parallel_coordinates(p, 'Type 1', color = ['Green', 'orange'])
reviews.head()
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go
# graph_objs go makes the graph object while iplot composes these objects and generates the plot.

# The only issue with plotly like interactive libraries is the amount of data it can plot,

# since they are very resource intensive.

iplot([go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])
# Kde plot + Scatter



# The graph_objs - iplot design also makes it easy to plot multiple plots on each other.



iplot([go.Histogram2dContour(x=reviews.head(500)['points'], 

                             y=reviews.head(500)['price'], 

                             contours=go.Contours(coloring='heatmap')),

       go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])
# Plotly surface is one of the best applications

df = reviews.assign(n=0).groupby(['points', 'price'])['n'].count().reset_index()

df = df[df["price"] < 100]

v = df.pivot(index='price', columns='points', values='n').fillna(0).values.tolist()

iplot([go.Surface(z=v)])
# On kaggle plotly is mostly used to make choropleths. Choropleth is a kind of map where every 

# region of the map is covered as per some variable.

df = reviews['country'].replace("US", "United States").value_counts()



iplot([go.Choropleth(

    locationmode='country names',

    locations=df.index.values,

    text=df.index,

    z=df.values

)])





# It is important to decide when to and when not to use plotly. While plotly is extremely attractive,

# it has less documentation, and also sometimes overly complex. It is very rarely useful as compared

# to equivalent plots in pandas and matplotlib.
# Get the data ready



top_wines = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]

top_wines.head()
# simple scatter plot

from plotnine import *



df = top_wines.head(1000).dropna()



( ggplot(df)

    + aes('points', 'price')

    + geom_point() )



# as it can be seen, it's super simple! ggplot takes in the data, aes takes in the aesthetics related

# details including the axes information. Finally a function is added to see what kind of plot it is.

# it is also super easy to just add another plot to this, add a function!



( ggplot(df)

    + aes('points', 'price')

    + geom_point()

    + stat_smooth())

# to add color, just add one more aes with color!



( ggplot(df)

    + aes('points', 'price')

     + aes(color = 'points')

    + geom_point()

    + stat_smooth())

# To add faceting to this, just add facet wrap!



( ggplot(df)

    + aes('points', 'price')

     + aes(color = 'points')

    + geom_point()

    + stat_smooth()

    + facet_wrap('~variety')

)





# If we wanted to add or remove faceting in seaborn or matplotlib, we would have to change the

# entrie structure of the code! With grammar of graphics based libraries, it's super easy.
# Bar plot



( ggplot(df)

+ aes('points')

 + geom_bar()

)
# Hex plot



( ggplot(top_wines)

 + aes('points', 'variety')

 + geom_bin2d(bins = 20)

)
# Non geometric functions can be mixed to make changes



(ggplot(top_wines)

         + aes('points', 'variety')

         + geom_bin2d(bins=20)

         + coord_fixed(ratio=1)

         + ggtitle("Top Five Most Common Wine Variety Points Awarded")

)
pd.set_option('max_columns', None)



stocks = pd.read_csv("../input/nyse/prices.csv", parse_dates=['date'])

stocks = stocks[stocks['symbol'] == "GOOG"].set_index('date')

stocks.head()
shelter_outcomes = pd.read_csv(

    "../input/austin-animal-center-shelter-outcomes-and/aac_shelter_outcomes.csv", 

    parse_dates=['date_of_birth', 'datetime']

)

shelter_outcomes = shelter_outcomes[

    ['outcome_type', 'age_upon_outcome', 'datetime', 'animal_type', 'breed', 

     'color', 'sex_upon_outcome', 'date_of_birth']

]

shelter_outcomes.head()
# Line plot across time



shelter_outcomes.date_of_birth.value_counts().sort_values().plot.line()
# In above plot it looks like data peaked around 2014, but we can't say for sure since data is rather noisy.

# To solve this problem, we resample the data so that it is yearly rather than daily.



# In pandas, resampling works in similar way like groupby.



shelter_outcomes.date_of_birth.value_counts().resample('Y').sum().plot.line()



# Pandas find x axis labels automatically, it's date-time aware.
stocks.volume.resample('Y').mean().plot.bar()
from pandas.plotting import lag_plot



lag_plot(stocks.volume.tail(250))



# following plot shows that two consecutive days of trading may be highly correlated. High volume on one 

# day also corresponds to higher volume on next day. It can also be seen however, this phenemenon seems 

# to be more prevalant in case of lower first day values. 
from pandas.plotting import autocorrelation_plot



autocorrelation_plot(stocks.volume)