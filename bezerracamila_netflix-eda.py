# open libraries

import pandas as pd

import seaborn as sns

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#to get the kaggle path

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# open df

df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv', encoding = 'latin1', engine = 'python', delimiter = ',')
# print 4 first rows 

df.head()
# print info about 

df.info()
# check if there's repetead values

## as expected, there aren't repetead values.



df.show_id.value_counts()
df.type.value_counts()/len(df)*100
## a new dataset with only movies.

df_netflix = df[(df.type == 'Movie')]
df_netflix.title.describe()
df_netflix[(df_netflix.title == "The Silence")]

#there's one movie repetead, thus I'll drop one row.
df_netflix = df_netflix[(df_netflix.show_id != '80238292')]
df_netflix.director.describe()
#print the movies directed by RaÃºl Campos and Jan Suter together.

df_netflix[(df_netflix.director == "RaÃºl Campos, Jan Suter")]
#create a df that split the director column

df_director = df_netflix.assign(var1 = df_netflix.director.str.split(',')).explode('var1').reset_index(drop = True)



#To remove white space at the beginning of string:

df_director['var1'] = df_director.var1.str.lstrip()
sns.countplot(y = df_director.var1, order=df_director.var1.value_counts().iloc[:15].index, palette = 'colorblind')

plt.title('Top 15 Directors')

plt.xlabel('')

plt.ylabel('')
df_netflix.cast.describe()
#create a df that split the cast column

df_cast = df_netflix.assign(var1=df_netflix.cast.str.split(',')).explode('var1').reset_index(drop=True)



#To remove white space at the beginning of string:

df_cast['var1'] = df_cast.var1.str.lstrip()
#plot the top 15 actors with the most movies.

sns.countplot(y = df_cast.var1, order=df_cast.var1.value_counts().iloc[:15].index, palette = 'colorblind')

plt.xlabel('Number of Movies')

plt.ylabel('Actor')

plt.title('Top 15 Actors')
df_netflix.country.describe()
#create a df that split the country column

df_country = df_netflix.assign(var1=df_netflix.country.str.split(',')).explode('var1').reset_index(drop=True)



#To remove white space at the beginning of string:

df_country['var1'] = df_country.var1.str.lstrip()
df_country.var1.value_counts()/len(df_country)*100
#plot the top 15 countries with the most movies produced.

sns.countplot(y = df_country.var1, order=df_country.var1.value_counts().iloc[:15].index, palette = 'colorblind')

plt.xlabel('Number of Movies')

plt.ylabel('Country')

plt.title('Top 15 Countries')
## convert to date

df_netflix['date_added']= df_netflix.date_added.astype('datetime64') 



## get only the year

df_netflix['date_added_year'] = df_netflix.date_added.dt.year



## get only the month

df_netflix['date_added_month'] = df_netflix.date_added.dt.month
## plot a density plot

sns.distplot(df_netflix.date_added_year)

plt.title('Added Year')

plt.xlabel("Year")
## plot a density plot

sns.distplot(df_netflix.date_added_month)

plt.title('Added Month')

plt.xlabel("Month")
# plot the number of movies added grouped by month 

#set order by the number of movies descending

sns.countplot(y=df_netflix.date_added_month, palette = 'colorblind', order = df_netflix.date_added_month.value_counts().index)
sns.distplot(df_netflix.release_year)

plt.title('Release Year')

plt.xlabel('Year')
df_netflix.rating.unique()
sns.countplot(y = df_netflix.rating, palette = 'colorblind', order = df_netflix.rating.value_counts().index)

plt.title('Rating')

plt.xlabel('')

plt.ylabel('')
df_netflix['duration'] = df_netflix.duration.str.rstrip('min').astype(float)



#df_disney.runtime.str.rstrip('min')
# duration density plot

sns.distplot(df_netflix.duration)

plt.title('Duration Density Plot')

plt.xlabel('Duration')
df_netflix.listed_in.describe()
#create a df that split the listed column

df_list = df_netflix.assign(var1 = df_netflix.listed_in.str.split(',')).explode('var1').reset_index(drop = True)



#To remove white space at the beginning of string:

df_list['var1'] = df_list.var1.str.lstrip()
df_list.var1.value_counts()/len(df_list)*100
sns.countplot(y = df_list.var1, palette = 'colorblind', order = df_list.var1.value_counts().index)

plt.title('Movies Genres')

plt.ylabel('')
# plot the most frequent words in the movies



# combine multiple rows into one object

text = df_netflix['description'].str.cat(sep='/ ')



# Create stopword list:

stopwords = set(STOPWORDS)

stopwords.update(["one", "two", "three", "four", "five"])



# Create and generate a word cloud image:

wc= WordCloud(stopwords = stopwords, background_color="white").generate(text)



# Display the generated image:

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()



#wc.to_file("netflix.png")

#adapted code from: https://www.datacamp.com/community/tutorials/wordcloud-python