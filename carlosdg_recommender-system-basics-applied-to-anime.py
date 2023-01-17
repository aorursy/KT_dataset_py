import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



from surprise import Dataset, Reader, NormalPredictor, KNNBasic, KNNWithMeans, SVD, accuracy

from surprise.model_selection import cross_validate



from sklearn.neighbors import NearestNeighbors



plt.style.use('seaborn-colorblind')

np.random.seed(8)

warnings.filterwarnings(action="ignore")
df_ratings = pd.read_csv("/kaggle/input/anime-recommendations-database/rating.csv")



display(df_ratings.sample(5, random_state=8))



print(f"""Number of total ratings: {df_ratings.shape[0]}. 

Number of different users: {df_ratings.user_id.nunique()}. 

Number of different animes: {df_ratings.anime_id.nunique()}.""")
def countplot_with_percentages(serie):

    ax = sns.countplot(serie)

    total_count = serie.count()



    for p in ax.patches:

        x = p.get_bbox().get_points()[:, 0]

        y = p.get_bbox().get_points()[1, 1]

        percentage = p.get_height() / total_count * 100

        ax.annotate(f'\n{percentage: .2f}%',

                    (x.mean(), y), ha='center', size=14)





plt.figure(figsize=(12, 6))

plt.axes(yscale="log")



countplot_with_percentages(df_ratings.rating)



plt.title("Distribution of Ratings in log scale")

plt.ylabel("Count in log scale")

plt.xlabel("Ratings");
cleaned_df_ratings = df_ratings[df_ratings.rating > 0].iloc[:50000, :]

reader = Reader(line_format='user item rating')

data = Dataset.load_from_df(cleaned_df_ratings, reader)
plt.figure(figsize=(12, 6))

#plt.axes(yscale="log")



countplot_with_percentages(cleaned_df_ratings.rating)



plt.title("Distribution of Ratings after cleaning the data")

plt.ylabel("Count")

plt.xlabel("Ratings");
algorithm = KNNBasic(k=40, random_state=8, sim_options={

    'name': 'pearson',

    'user_based': True

})



cross_validate(algorithm, data, measures=['RMSE'], cv=5, verbose=True);
algorithm = KNNWithMeans(k=40, random_state=8, sim_options={

    'name': 'pearson',

    'user_based': True

})



cross_validate(algorithm, data, measures=['RMSE'], cv=5, verbose=True);
algorithm = SVD(random_state=8, n_factors=100)



cross_validate(algorithm, data, measures=['RMSE'], cv=5, verbose=True);
df_animes = pd.read_csv("/kaggle/input/anime-recommendations-database/anime.csv", index_col="anime_id")

df_animes.head()
# Drop the animes with null values

df_clean_animes = df_animes[df_animes.genre.notna() & df_animes.type.notna()]



# First, split the genre column by comma and expand the list so there is

# a column for each genre. Now we have 13 columns, because the anime with

# most genres tags has 13 tags

genres = df_clean_animes.genre.str.split(", ", expand=True)



# Now we can get the list of unique genres. We "convert" the dataframe into

# a single dimension array and take the unique values

unique_genres = pd.Series(genres.values.ravel('K')).dropna().unique()



# Getting the dummy variables will result in having a lot more columns

# than unique genres

dummies = pd.get_dummies(genres)



# So we sum up the columns with the same genre to have a single column for

# each genre

for genre in unique_genres:

    df_clean_animes["Genre: " + genre] = dummies.loc[:, dummies.columns.str.endswith(genre)].sum(axis=1)

    

# Add the type dummies

type_dummies = pd.get_dummies(df_clean_animes.type, prefix="Type:", prefix_sep=" ")

df_clean_animes = pd.concat([df_clean_animes, type_dummies], axis=1)



df_clean_animes = df_clean_animes.drop(columns=["name", "type", "genre", "episodes", "rating", "members"])

df_clean_animes.head()
# Helper function to get the features of an anime given its name

def get_features_from_anime_name(name):

    return df_clean_animes.loc[df_animes[df_animes.name == name].index]





# Build and "train" the model

neigh = NearestNeighbors(15)

neigh.fit(df_clean_animes.values)



# Get the features of this anime

item_to_compare = get_features_from_anime_name("Dragon Ball Z")



# Get the indices of the most similar items found

# Note: these are ignoring the dataframe indices and starting from 0

index = neigh.kneighbors(item_to_compare, return_distance=False)



# Show the details of the items found

df_animes.loc[df_animes.index[index][0]]