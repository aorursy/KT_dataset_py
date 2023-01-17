# open libraries

import pandas as pd

import seaborn as sns

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np
#to get the kaggle path

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
# open df

df = pd.read_csv('/kaggle/input/disney-plus-shows/disney_plus_shows.csv', encoding = 'latin1', engine = 'python', delimiter = ',')
# print 4 first rows 

df.head().T
# print info about 

df.info()
# check if there's repetead values

## as expected, there aren't repetead values.



df.imdb_id.value_counts()
df.title.value_counts() 
#filter to only movies

df_disney = df[(df.type == "movie")]
df_disney['plot']
# plot the most frequent words in the movies

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# combine multiple rows into one object

text = df_disney['plot'].str.cat(sep='/ ')



# Create stopword list:

stopwords = set(STOPWORDS)



# Create and generate a word cloud image:

wc= WordCloud(stopwords = stopwords, background_color="white").generate(text)



# Display the generated image:

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()



#wc.to_file("disney.png")

#adapted code from: https://www.datacamp.com/community/tutorials/wordcloud-python
# show's type percentage

df.type.value_counts()/len(df)*100
ax = sns.countplot(y = df.type, palette = 'colorblind')

ax.set(xlabel = 'Frequency', ylabel= 'Type')

plt.title("Disney Show's Type")
df_disney.rated.value_counts()/len(df_disney)*100
# set order by value_counts

ax = sns.countplot(y = df_disney.rated, palette = 'colorblind', order = df_disney.rated.value_counts().index)

ax.set(xlabel = 'Frequency', ylabel= 'Rate')

plt.title("Disney Rate")
df_disney.year.value_counts()/len(df_disney)*100

# this column need to be treated
df_disney.released_at.value_counts()/len(df_disney)*100
## tranform in date

df_disney['released']= df_disney.released_at.astype('datetime64') 



## get only the year

df_disney['released_year'] = df_disney.released.dt.year



## get only the month

df_disney['released_month'] = df_disney.released.dt.month
## plot a graph line

sns.distplot(df_disney.released_year)

plt.title('Released Year')

plt.xlabel("Year")
# plot released_month

sns.countplot(y = df_disney.released_month, palette = 'colorblind')

plt.title('Disney Shows Released Month')

plt.xlabel('Frequency')

plt.ylabel('Month')
## tranform in date

df_disney['added'] = df_disney.added_at.astype('datetime64')



## get only the year

df_disney['added_year'] = df_disney.added.dt.year

df_disney['added_month'] = df_disney.added.dt.month
# explore added_year

df_disney.added_year.value_counts()/len(df_disney) * 100
# plot added_month

df_disney.added_month.value_counts()/len(df_disney)*100
# plot added_month

sns.countplot(y = df_disney.added_month, palette = 'colorblind')

plt.title('Disney Shows Added Month')

plt.xlabel('Frequency')

plt.ylabel('Month')
## convert to a float

df_disney['runtime'] = df_disney.runtime.str.rstrip('min')

df_disney['runtime'] = df_disney.runtime.str.rstrip('h').astype(float)
df_disney.runtime.describe()
sns.distplot(df_disney.runtime)

plt.title('Runtime')
sns.boxplot(df_disney.runtime)
df_disney.genre.value_counts()/len(df_disney)*100
#create a df that split the director column

df_director = df_disney.assign(var1 = df_disney.director.str.split(',')).explode('var1').reset_index(drop = True)



#To remove white space at the beginning of string:

df_director['var1'] = df_director.var1.str.lstrip()
sns.countplot(y = df_director.var1, order=df_director.var1.value_counts().iloc[:15].index, palette = 'colorblind')

plt.title('Top 15 Directors')

plt.xlabel('')

plt.ylabel('')
#create a df that split the writer column

df_writer = df_disney.assign(var1 = df_disney.writer.str.split(',')).explode('var1').reset_index(drop = True)



#To remove white space at the beginning of string:

df_writer['var1'] = df_writer.var1.str.lstrip()
#plot top 15 writers with the most movies produced

sns.countplot(y = df_writer.var1, order=df_writer.var1.value_counts().iloc[:15].index, palette = 'colorblind')

plt.title('Top 15 Writers')

plt.xlabel('')

plt.ylabel('')
#create a df that split the writer column

df_actors = df_disney.assign(var1 = df_disney.actors.str.split(',')).explode('var1').reset_index(drop = True)



#To remove white space at the beginning of string:

df_actors['var1'] = df_actors.var1.str.lstrip()
#plot top 15 actors with the most movies produced

sns.countplot(y = df_actors.var1, order=df_actors.var1.value_counts().iloc[:15].index, palette = 'colorblind')

plt.title('Top 15 Actors')

plt.xlabel('')

plt.ylabel('')
#create a df that split the writer column

df_lang= df_disney.assign(var1 = df_disney.language.str.split(',')).explode('var1').reset_index(drop = True)



#To remove white space at the beginning of string:

df_lang['var1'] = df_lang.var1.str.lstrip()
#plot top 15 actors with the most movies produced

sns.countplot(y = df_lang.var1, order=df_lang.var1.value_counts().iloc[:15].index, palette = 'colorblind')

plt.title('Top 15 Languages')

plt.xlabel('')

plt.ylabel('')
#create a df that split the writer column

df_countries= df_disney.assign(var1 = df_disney.country.str.split(',')).explode('var1').reset_index(drop = True)



#To remove white space at the beginning of string:

df_countries['var1'] = df_countries.var1.str.lstrip()
#plot top 15 actors with the most movies produced

sns.countplot(y = df_countries.var1, order=df_countries.var1.value_counts().iloc[:5].index, palette = 'colorblind')

plt.title('Top 5 Countries')

plt.xlabel('')

plt.ylabel('')
df_disney.awards.value_counts()/len(df_disney)*100
df_disney.awards.describe()
df_disney.metascore.describe()
# metascore density plot

sns.distplot(df_disney.metascore)
df_disney.imdb_rating.describe()
sns.distplot(df_disney.imdb_rating)
# replace ',' with ''

df_disney['imdb_votes'] = df_disney.imdb_votes.str.replace(',', '').astype(float)
df_disney.imdb_votes.describe()
sns.distplot(df_disney.imdb_votes)