import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plotting
from scipy import stats # statistics
from sklearn.metrics.pairwise import cosine_similarity
anime = pd.read_csv("../input/anime.csv") # load data
anime.head(10)
anime.describe()
# anime.episodes.describe() # dtype: object
# anime['episodes'] = pd.to_numeric(anime['episodes']) # ValueError: invalid literal for int() with base 10: 'Unknown'

anime["episodes"] = anime["episodes"].map(lambda x:np.nan if x=="Unknown" else x)
anime["episodes"].fillna(anime["episodes"].median(),inplace = True)
anime['episodes'] = pd.to_numeric(anime['episodes'])
anime.describe()
sns.jointplot(x='episodes', y='rating', data=anime)
anime[(anime.episodes > 200)].sort_values(by='episodes', ascending=False)
sns.jointplot(x='episodes', y='rating', data=anime[(anime.episodes < 100)])
anime.episodes.value_counts().head(10)
anime.loc[(anime.rating > 9)].sort_values(by='rating', ascending=False)
sns.distplot(anime.rating.dropna(), bins=20) # overall distribution
sns.jointplot(x='members', y='rating', data=anime) # scatterplot of rating vs members
sns.distplot(anime[(anime.members >= 100)].rating.dropna(), bins=20) # distribution for animes with at least 100 members
sns.jointplot(x='members', y='rating', data=anime[(anime.members) > 100]) # scatterplot of rating vs members > 100
sns.distplot(anime[(anime.members < 100)].rating.dropna(), bins=20) # distribution for animes with less than 100 members
sns.jointplot(x='members', y='rating', data=anime[(anime.members < 100)]) # scatterplot of rating vs members < 100
low_members = anime[(anime.members < 100)].rating.dropna().sample(100, random_state = 0)
all_members = anime.rating.dropna().sample(100, random_state = 0)
stats.ks_2samp(low_members, all_members)
normal_members = anime[(anime.members >= 100)].rating.dropna().sample(100, random_state = 0)
stats.ks_2samp(normal_members, all_members)
low = []
norm = []
for i in range(1000):
    low_members = anime[(anime.members < 100)].rating.dropna().sample(100)
    all_members = anime.rating.dropna().sample(100)
    norm_members = anime[(anime.members >= 100)].rating.dropna().sample(100)
    low_stat, low_p = stats.ks_2samp(low_members, all_members)
    norm_stat, norm_p = stats.ks_2samp(norm_members, all_members)
    low.append((low_stat, low_p))
    norm.append((norm_stat, norm_p))
low_values = pd.DataFrame.from_records(low, columns = ['KS_stat', 'p_value'])
norm_values = pd.DataFrame.from_records(norm, columns = ['KS_stat', 'p_value'])
low_values[low_values.p_value >= 0.05].p_value.count() / len(low_values)
norm_values[norm_values.p_value < 0.05].p_value.count() / len(norm_values)
genre_features = anime.genre.str.get_dummies(sep=', ')
print(genre_features.shape)
genre_features.head()
genre_counts = pd.DataFrame(genre_features.sum()).reset_index().rename(columns={'index':'genre', 0:'count'})
genre_counts.sort_values('count', ascending = False, inplace = True)
sns.barplot(x = 'count', y = 'genre', data = genre_counts.head(20))
genre_list = [genre for genre in genre_counts.genre.head(20)]
sns.heatmap(genre_features[genre_list].corr(), vmax = 0.6)
cosine_matrix = cosine_similarity(genre_features)
print(cosine_matrix.shape)
def recommend_anime(anime_index):
    cos_similarity = pd.DataFrame(cosine_matrix[anime_index]).rename({0: 'cos_sim'}, axis = 'columns')
    weighted_score = pd.DataFrame(anime.rating * cos_similarity.cos_sim, columns = ['cos_score'])
    result = pd.concat([anime.name, anime.genre, cos_similarity.cos_sim, weighted_score.cos_score], axis = 'columns')
    title, genres = anime.loc[anime_index, ['name', 'genre']]
    print("Shows similar to '%s', which has genres '%s' are:" % (title, genres))
    return result.drop([anime_index]).sort_values(by='cos_score', ascending = False)[['name', 'cos_score']].head(15)
my_favorite_animes = ['Monster', 'Full Metal Panic? Fumoffu', 'Juuni Kokuki', 'Mononoke Hime', 'Ghost in the Shell: Stand Alone Complex', 'Nana', 'Higurashi no Naku Koro ni', 'Clannad', 'Tengen Toppa Gurren Lagann', 'Steins;Gate', 'Kimi no Na wa.']
#my_favorite_animes = ['Boku no Hero Academia', 'Shingeki no Kyojin', 'One Punch Man', 'Fullmetal Alchemist', 'Nanatsu no Taizai', 'One Piece']

anime_list_indices = anime.loc[anime.name.isin(my_favorite_animes)].index
for index in anime_list_indices:
    output = recommend_anime(index)
    print(output)
my_favorite_animes = ['Clannad: After Story', 'Clannad']

anime_list_indices = anime.loc[anime.name.isin(my_favorite_animes)].index
for index in anime_list_indices:
    output = recommend_anime(index)
    print(output)
user = pd.read_csv('../input/rating.csv')
user.head()
user.rating.replace({-1: np.nan}, regex=True, inplace = True)
user.head()
user.describe()
user.rating.value_counts()
user[(user.rating.isin([7,8,9]))].rating.value_counts().sum() / user.rating.value_counts().sum()
sns.countplot(user.rating)
avg = user.groupby('user_id').rating.mean().sort_values(ascending=False).dropna()
print(avg.head())
print(avg.tail())
user.groupby('user_id').rating.mean().isin([float(i+1) for i in range(10)]).value_counts()
sns.distplot(user.groupby('user_id').rating.mean().dropna(), bins = 20)
