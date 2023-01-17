import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
anime = pd.read_csv('/kaggle/input/anime-recommendations-database/anime.csv')
rating = pd.read_csv('/kaggle/input/anime-recommendations-database/rating.csv')
anime.head()
rating.head()
anime.describe()
rating.describe()
# Lets have a look the distribution of ratings, because those "-1" are suspicious
rating.rating.value_counts()
# I'm not sure what the "-1" mean because the ratings goes from 1 up to 10. 
# Maybe it means, that no rating are available. 
# Therefore we will delete the rows with "-1" in rating
rating = rating[rating["rating"] != -1]
print(f"anime.csv - rows: {anime.shape[0]}, columns: {anime.shape[1]}")
print(f"rating.csv - rows: {rating.shape[0]}, columns: {rating.shape[1]}")
plt.figure(figsize=(8,6))
sns.heatmap(anime.isnull())
plt.title("Missing values in anime?", fontsize = 15)
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(rating.isnull())
plt.title("Missing values in rating?", fontsize = 15)
plt.show()
# Merge anime and rating using "anime_id" as reference
# Keep only the columns we will use
df = pd.merge(rating,anime[["anime_id","name"]], left_on = "anime_id", right_on = "anime_id").drop("anime_id", axis = 1)
df.head()
# Count the number of ratings for each anime
count_rating = df.groupby("name")["rating"].count().sort_values(ascending = False)
count_rating
# Some animes have only 1 rating, therefore it is better for the recommender system to ignore them
# We will keep only the animes with at least r ratings
r = 5000
more_than_r_ratings = count_rating[count_rating.apply(lambda x: x >= r)].index

# Keep only the animes with at least r ratings in the DataFrame
df_r = df[df['name'].apply(lambda x: x in more_than_r_ratings)]
before = len(df.name.unique())
after = len(df_r.name.unique())
rows_before = df.shape[0]
rows_after = df_r.shape[0]
print(f'''There are {before} animes in the dataset before filtering and {after} animes after the filtering.

{before} animes => {after} animes
{rows_before} rows before filtering => {rows_after} rows after filtering''')
# Create a matrix with userId as rows and the titles of the movies as column.
# Each cell will have the rating given by the user to the animes.
# There will be a lot of NaN values, because each user hasn't watched most of the animes
df_recom = df_r.pivot_table(index='user_id',columns='name',values='rating')
df_recom.iloc[:5,:5]
df_r.name.value_counts().head(10)
def find_corr(df, name):
    '''
    Get the correlation of one anime with the others
    
    Args
        df (DataFrame):  with user_id as rows and movie titles as column and ratings as values
        name (str): Name of the anime
    
    Return
        DataFrame with the correlation of the anime with all others
    '''
    
    similar_to_movie = df.corrwith(df[name])
    similar_to_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])
    similar_to_movie = similar_to_movie.sort_values(by = 'Correlation', ascending = False)
    return similar_to_movie
# Let's choose an anime
anime1 = 'Naruto'

# Let's try with "Naruto"

# Recommendations
find_corr(df_recom, anime1).head(40)
# Not recommended
find_corr(df_recom, anime1).tail(40)
# Let's choose an anime
anime2 = 'Death Note'

# Recommendations
find_corr(df_recom, anime2).head(40)
# Not recommended
find_corr(df_recom, anime2).tail(40)