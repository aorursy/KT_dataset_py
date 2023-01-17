import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
movies=pd.read_csv ("../input/Movie_Movies.csv")

genres=pd.read_csv ("../input/Movie_Genres.csv")

ratings=pd.read_csv ("../input/Movie_AdditionalRating.csv")
movies.info()
ratings.info()

len(ratings.imdbID.unique())
ratings.drop_duplicates(subset='imdbID',inplace=True)

#Only one rating per film is needed for the filtering
#makings sure there are no identicals

movies.drop_duplicates(subset='imdbID',inplace=True)





ratings.drop_duplicates(subset='imdbID',inplace=True)

#Only one rating per film is needed for the filtering



#the following definitely won't be used





movies.drop(['Awards','DVD', 'Poster', 'Plot','Website', 'Production','Released','imdbVotes','Type','Rated'], axis = 1, inplace = True)

genres.Genre= [x.strip()for x in genres.Genre]
#without any filtering

rawdirectors=movies["Director"].value_counts()[:10]

pd.DataFrame({'Director':rawdirectors.index, 'NumberOfMoviesProduced':rawdirectors.values})
#using rated films only



rated=pd.merge(movies,ratings, on = 'imdbID')

rated.drop_duplicates(subset='imdbID',inplace=True)



rateddirectors=rated['Director'].value_counts()[:10]

pd.DataFrame({'Director':rateddirectors.index, 'NumberOfMoviesProduced':rateddirectors.values})
#using films from specified genres



updated=pd.merge(movies,genres, on = 'imdbID',how="left")



notwanted=["Short", "Adult"]

notwantedids=updated[updated.Genre.isin(notwanted)]["imdbID"]



#dropping the ids that have at least a genre belonging in the notwanted

updated.drop(updated[updated.imdbID.isin(notwantedids)].index, inplace=True)

updated.drop_duplicates(subset='imdbID',inplace=True)

updated = updated.reset_index(drop=True)
updateddirectors=updated['Director'].value_counts()[:10].index.tolist()

filmstopten=updated[updated.Director.isin(updateddirectors)]



genredirectors=updated['Director'].value_counts()[:10]

pd.DataFrame({'Director':genredirectors.index, 'NumberOfMoviesProduced':genredirectors.values})
#using both





rated.drop(rated[rated.imdbID.isin(notwantedids)].index, inplace=True)





finaldirectors=rated['Director'].value_counts()[:10]

DoubleFilteredDirectors=pd.DataFrame({'Director':finaldirectors.index, 'NumberOfMoviesProduced':finaldirectors.values})

DoubleFilteredDirectors

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams

%matplotlib inline

# figure size in inches



plt.figure(figsize=(15,10))

boom=sns.barplot(x="Director", y='NumberOfMoviesProduced', palette="ch:.25", data=DoubleFilteredDirectors)

plt.xticks(rotation= 45)

plt.xlabel('Directors')

plt.ylabel('Number of Films')

plt.title('Films per Director')

plt.show()