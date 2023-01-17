# I am trying to find the highest rated movies by the top content ratings, such as PG, PG-13, and R. 
#I improved on the previous versions of more tedius codes. I welcome your feedback on more improvement.

import numpy as np 
import pandas as pd 

movies = pd.read_csv('../input/imdb_1000.csv', header=0)
movies.head()
#First, I am going to drop the messy actors_list column. 
movies.drop(['actors_list'],  axis=1, inplace=True)
movies.head()
# The total number of movies per each existing content_ratings.
# I will keep the most frequent ratings and get rid of the rest.
movies.content_rating.value_counts()
# I will keep the first 3: R, PG-13 and PG.
#Here I got a new dataframe with movies only in 3 groups of content. I used .isin to achieve this.
movies_trimmed=movies[movies.content_rating.isin(['R', 'PG', 'PG-13'])].copy()
#In the first version I wrote lengthy codes, which got rid of the less popular genres in two steps-
#first resetting the idex to ensure that there are no duplications, and then using .drop along with | operator.
movies_trimmed.content_rating.value_counts()
#a snapshot of the new trimmed df
movies_trimmed.sample(10)
#I'll sort the movies by content_rating and then star_rating
movies_trimmed.sort_values(['content_rating','star_rating'], ascending=False, inplace=True)
movies_trimmed.head()

#Now I am going to separate the sorted movies by content group, and save them in 3 dataframes 
movies_1=movies_trimmed[movies_trimmed.content_rating=='R']
movies_2=movies_trimmed[movies_trimmed.content_rating=='PG']
movies_3=movies_trimmed[movies_trimmed.content_rating=='PG-13']
#the first df's tail-end
movies_1.tail()
# Now, finding the 4 largest values in each dataframe, then combining the 3 dataframes into one, the quicker way
result=pd.concat((movies_1.nlargest(4,'star_rating', keep='first'), movies_2.nlargest(3,'star_rating', keep='first'),movies_3.nlargest(3,'star_rating', keep='first')),ignore_index=True)
# In the last version, I did the same job in a more tedius way, see those codes below:
#group=movies_1[(movies_1.star_rating>=9) & (movies_1.content_rating=='R')]
#group=group.append(movies_1[(movies_1.star_rating>=8.8) & (movies_1.content_rating=='PG-13')], ignore_index=True)
#group=group.append(movies_1[(movies_1.star_rating>=8.7) & (movies_1.content_rating=='PG')] , ignore_index=True)
#group=group.append(movies_1[(movies_1.star_rating>=8.5) & (movies_1.content_rating=='APPROVED')] , ignore_index=True)
#group

#the final df
result
result.to_csv('imdb_output.csv')





