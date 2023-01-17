# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")

df.info()
df.head()
dropped_df = df.drop(["homepage", "keywords", "overview", "production_companies", "production_countries", "original_language", "status", "spoken_languages", "tagline", "release_date", "original_title"], axis=1)
dropped_df.head()
# I wanted to look at the relationship between popularity and movie genres,
# but I had to give it up as 4 different movie genres were entered for a single movie in the dataset.

#dropped_df["genres"].describe()

#d_genres = dropped_df.loc[:,"genres"]
#print(d_genres)
#print(" ")
#data_dict = d_genres.to_dict()
#print(data_dict)

# That's why I am gonna extract "genres" too.

dropped_df2 = dropped_df.drop(["genres"], axis= 1)
dropped_df2.head(10)
dropped_df2.corr()
#Let's visualize the correlation
#correlation map
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(dropped_df2.corr(), annot = True, linewidths= .2, fmt= ".2f",ax=ax )
plt.show()
dropped_df2.plot(kind="scatter",x="popularity",y="revenue",alpha=.5,color="b",grid=True,figsize=(13,13))
plt.xlabel("popularity")
plt.ylabel("revenue")
plt.title("popularity vs. revenue")
plt.show()
dropped_df2.plot(kind="scatter",y="popularity",x="vote_count",alpha=.5,color="g",grid=True,figsize=(13,13))
plt.ylabel("popularity")
plt.xlabel("vote_count")
plt.title("popularity vs. vote_count")
plt.show()
dropped_df.plot(kind="scatter",y="revenue",x="vote_count",alpha=.5,color="r",grid=True,figsize=(13,13))
plt.ylabel("revenue")
plt.xlabel("vote_count")
plt.title("revenue vs. vote_count")
plt.show()
dropped_df2.popularity.plot(kind="line",color="r", label="popularity",lw=1,ls=":",alpha=.9, grid=True,figsize=(12,12))
plt.xlabel("index")
plt.ylabel("popularity")
plt.legend(loc="upper right")
plt.title("Popularity")
plt.show()
pop= dropped_df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(10),pop['popularity'].head(10),color='orange')
plt.gca()
plt.xlabel("Popularity")
plt.title("Popular Movies")
dropped_df2.vote_average.plot(kind= "hist", color='crimson', bins=100, grid=True, figsize= (12,12))
plt.xlabel('vote_avarage')
plt.title('Vote avarage histogram')
plt.show()
dropped_df2.runtime.plot(kind= "hist", color='purple', bins=100, grid=True, figsize= (12,12))
plt.xlabel('runtime')
plt.title('Runtime histogram')
plt.show()
#Top 10 most popular films
pop.head(6)
#title and vote_avarage
dictionary = {'Minions':'6.4', 'Interstellar':'8.1', 'Deadpool':'7.4', 'Guardians of the Galaxy':'7.9', 'Mad Max:Fury Road':'7.2', 'Jurassic World':'6.5'}
dictionary
pop.tail(3)
dictionary_tail = {'Penitentiary':'4.9','Alien Zone':'4.0','America Is Still the Place':'0.0'}
dictionary.update(dictionary_tail)
dictionary
print(dictionary.keys())
print(dictionary.values())
dictionary ['America Is Still the Place'] = "2.0" #update existing entry
dictionary
print('Big Hero 6'in dictionary ) #check include or not
#comparisson operators
p=dropped_df2["popularity"] > 200 #The popularity of only the 11 most popular films is greater than 200
dropped_df2[p]
x=pop["vote_average"] > 7.5 #The avarage vote of most popular films is greater than 7.0
pop[x]
len(pop[x]) #The average rating of 271 of 4803 films is more than 7.5
pop[np.logical_and(pop['vote_average']>7.5, pop['budget']>150000000 )]
liste = [724,481,203,187,167]
for index, value in enumerate(liste):
    print(index,':', value)

dict2 = {'Interstellar':'724','Guardians of the Galaxy':'481','Big Hero 6':'203','The Dark Night':'187','Inception':'167'}
for key, value in dict2.items():
    print(key,':', value)
for index,value in pop[['title']][0:1].iterrows():
    print(index," : ",value)
for index,value in pop[['vote_average']][0:1].iterrows():
    print(index," : ",value)
for index,value in pop[['popularity']][0:1].iterrows():
    print(index," : ",value)
#title and vote_avarage
def f(**kwargs):
    for key, value in kwargs.items():               
        print(key, " ", value)
f( Minions = 6.4, Interstellar = 8.1, Deadpool = 7.4, GuardiansoftheGalaxy = 7.9, MadMaxFuryRoad = 7.2, JurassicWorld = 6.5)