#Import relevant libraries

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import seaborn as sns



sns.set(style="white")



#Read CSV file

df = pd.read_csv('../input/movie_metadata.csv')



#Preview the data

df.head(n=5)
#View information on data

df.info()
#Check for null values

print (df.isnull().any())



#Drop null values

df = df.dropna()
#Searching for duplicate rows

num_duplicates = 0

for i in df.duplicated():

    if i == True:

        num_duplicates += 1

print ('Number of Duplicates:',num_duplicates)



#Drop duplicates

df.drop_duplicates(inplace = True)
#Descriptive statistics

df.describe()
#Plotting distribution of gross profit

gross = df['gross'].tolist()

plt.hist(gross,75,color = "#74a9cf")

plt.title('Distribution of Gross Profit')

plt.ylabel('Number of Movies')

plt.xlabel('Gross Profit')

plt.show()
#Plotting distribution of IMDB scores

imdb_score = df['imdb_score'].tolist()

plt.hist(imdb_score,75,color = "#74a9cf")

plt.title('Distribution of IMDB Ratings')

plt.ylabel('Number of Movies')

plt.xlabel('IMDB Rating')

plt.show()
#Top 10 gross movies

df.loc[:,['movie_title','gross','director_name','title_year']].sort_values(by = 'gross', ascending = False)[:10]
#Top 10 imdb-rated movies

df.loc[:,['movie_title','imdb_score','director_name','title_year']].sort_values(by = 'imdb_score', ascending = False)[:10]
#Computing the correlation matrix

corr = df.corr()



#Generating a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



#Setting up the matplotlib figure

f, ax = plt.subplots(figsize=(8,8))



#Generating a custom diverging colormap

cmap = sns.diverging_palette(220,10, as_cmap=True)



#Drawing the heatmap with the mask

sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)



plt.title("Correlation Matrix")

plt.show()
#Countries that release the most movies

dist_countries_df = df.groupby(['country']).count()['imdb_score'].sort_values(ascending=False)[:10]



plt.figure(figsize=(8,4))

sns.barplot(y = dist_countries_df.index, x = dist_countries_df.values, palette = "PuBu_r")

plt.title('Countries that Release the Most Movies')

plt.xlabel('Movies Released')

plt.ylabel('Country')

plt.show()
#Plotting average imdb ratings of movies from 10 countries who release the most movies

top_dist_countries = df[df['country'].isin((df.groupby(['country']).count()['imdb_score'].sort_values(ascending=False)[:10]).index)]



plt.figure(figsize=(8,4))

sns.boxplot(x = 'country', y = 'imdb_score', data = top_dist_countries, palette = 'PuBu_r')

plt.xticks(rotation = 45)

plt.title('IMDB Scores for Countries that Release the Most Movies')

plt.xlabel('Country')

plt.ylabel('IMDB Score')

plt.show()
#Directors who release the most movies

dist_directors_df = df.groupby(['director_name']).count()['imdb_score'].sort_values(ascending=False)[:10]



plt.figure(figsize=(8,5))

sns.barplot(y = dist_directors_df.index, x = dist_directors_df.values, palette = "PuBu_r")

plt.title('Directors Who Released the Most Movies')

plt.xlabel('Movies Released')

plt.ylabel('Director')

plt.show()
#Plotting average imdb rating for directors who have released the most movies

top_dist_directors = df[df['director_name'].isin((df.groupby(['director_name']).count()['imdb_score'].sort_values(ascending=False)[:10]).index)]



plt.figure(figsize=(8,4))

sns.boxplot(x = 'director_name', y = 'imdb_score', data = top_dist_directors, palette = 'PuBu_r')

plt.xticks(rotation = 45)

plt.title('IMDB Scores for Directors Who Release the Most Movies')

plt.xlabel('Director')

plt.ylabel('IMDB Score')

plt.show()
#Directors with the highest average imdb rating

imdb_director_df = df.groupby(['director_name']).mean()['imdb_score'].sort_values(ascending=False)[:10]



plt.figure(figsize=(8,5))

sns.barplot(y = imdb_director_df.index, x = imdb_director_df.values, palette = "PuBu_r")

plt.title('Directors with the Highest Average IMDB Rating')

plt.xlabel('Average Rating')

plt.ylabel('Director')

plt.show()
#Directors with the highest gross profit

gross_director_df = df.groupby(['director_name']).sum()['gross'].sort_values(ascending=False)[:10]



plt.figure(figsize=(8,5))

sns.barplot(y = gross_director_df.index, x = gross_director_df.values, palette = "PuBu_r")

plt.title('Directors with the Highest Total Gross')

plt.xlabel('Total Gross')

plt.ylabel('Director')

plt.show()
#Plotting average gross for top directors

top_directors_gross = df[df['director_name'].isin((df.groupby(['director_name']).sum()['gross'].sort_values(ascending=False)[:10]).index)]



plt.figure(figsize=(8,4))

sns.boxplot(x = 'director_name', y = 'gross', data = top_directors_gross, palette = 'PuBu_r')

plt.xticks(rotation = 45)

plt.title('Gross for Directors with Highest Total Gross')

plt.xlabel('Director')

plt.ylabel('Gross')

plt.show()
#Heatmap of gross by director and year

top_10_dir = df.groupby(['director_name']).sum()['gross'].sort_values(ascending = False)[:10]

director_year_table = pd.pivot_table(df[df['director_name'].isin(top_10_dir.index)], values = ['gross'], index = ['title_year'], columns = ['director_name'], aggfunc = 'sum')



plt.figure(figsize=(8,6))

sns.heatmap(director_year_table['gross'], annot_kws = {"size": 8}, fmt = 'g', cmap = "PuBu")

plt.xticks(rotation = 70)

plt.title('Gross by Year and Director (with Highest Total Gross)')

plt.xlabel('Director')

plt.ylabel('Year')

plt.show()
#Actors who are in the most movies

actors_df = df.groupby(['actor_1_name']).count()['imdb_score'].sort_values(ascending=False)[:10]



plt.figure(figsize=(8,5))

sns.barplot(y = actors_df.index, x = actors_df.values, palette = "PuBu_r")

plt.title('Actors in the Most Movies')

plt.xlabel('Number of Movies')

plt.ylabel('Actor')

plt.show()
#Plotting average imdb rating for actors who appear in the most movies

top_actors = df[df['actor_1_name'].isin((df.groupby(['actor_1_name']).count()['imdb_score'].sort_values(ascending=False)[:10]).index)]



plt.figure(figsize=(8,4))

sns.boxplot(x = 'actor_1_name', y = 'imdb_score', data = top_actors, palette = 'PuBu_r')

plt.xticks(rotation = 45)

plt.title('IMDB Score for Actors Who Appear in the Most Movies')

plt.xlabel('Actor')

plt.ylabel('IMDB Score')

plt.show()
#Actors with the highest gross profit

gross_actor_df = df.groupby(['actor_1_name']).sum()['gross'].sort_values(ascending=False)[:10]



plt.figure(figsize=(8,5))

sns.barplot(y = gross_actor_df.index, x = gross_actor_df.values, palette = "PuBu_r")

plt.title('Actors with the Highest Total Gross Profit')

plt.xlabel('Total Gross')

plt.ylabel('Actor')

plt.show()
#Plotting average gross for top actors

top_actors_gross = df[df['actor_1_name'].isin((df.groupby(['actor_1_name']).sum()['gross'].sort_values(ascending=False)[:10]).index)]



plt.figure(figsize=(8,4))

sns.boxplot(x = 'actor_1_name', y = 'gross', data = top_actors_gross, palette = 'PuBu_r')

plt.xticks(rotation = 45)

plt.title('Gross for Actors with Highest Total Gross')

plt.xlabel('Actors')

plt.ylabel('Gross')

plt.show()