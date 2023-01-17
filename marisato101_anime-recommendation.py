import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
ratings = pd.read_csv('/Users/admin/Desktop/anime-recommendations-database/rating.csv')
anime = pd.read_csv('/Users/admin/Desktop/anime-recommendations-database/anime.csv')
ratings.head()
anime.head()
#'members'でソート
anime.sort_values('members', ascending=False).head()
round(anime.describe(), 2)
round(ratings.describe(), 2)
ratings['rating'].hist(bins=11, figsize=(5, 5))
anime['members'].hist(bins=20, figsize=(10, 5))
#members 10000以上のものだけを選択
popu_anime = anime[anime['members'] > 10000]
round(popu_anime.describe(), 2)
#欠損データを確認
popu_anime.isnull().sum()
#欠損データ削除
popu_anime = popu_anime.dropna()
#評価ついてるものだけを選択
modi_ratings = ratings[ratings.rating >= 0]
round(modi_ratings.describe(), 2)
modi_ratings.isnull().sum()
#animeとratingをマージ
mergeddf = modi_ratings.merge(popu_anime, left_on='anime_id', right_on='anime_id', suffixes=['_user', ''])

mergeddf.head()
round(mergeddf.describe(), 2)
mergeddf = mergeddf[['user_id', 'name', 'rating_user']]
mergeddf = mergeddf.drop_duplicates(['user_id', 'name' ])
mergeddf.head()
#データフレームをピボット
anime_pivot = mergeddf.pivot(index='name', columns='user_id', values='rating_user').fillna(0)
anime_pivot_sparse = csr_matrix(anime_pivot.values)

anime_pivot.head()
#Learning model
knn = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='cosine')
model_knn = knn.fit(anime_pivot_sparse)
def searchanime(string):
    print(anime_pivot[anime_pivot.index.str.contains(string)].index[0:])

searchanime('Hajime')
# 「はじめの一歩」に対してのオススメのアニメ10個
Anime = 'Hajime no Ippo'
distance, indice = model_knn.kneighbors(anime_pivot.iloc[anime_pivot.index== Anime].values.reshape(1,-1), n_neighbors=11)
for i in range(0, len(distance.flatten())):
    if  i == 0:
        print('Recommendations if you like the anime {0}:\n'.format(anime_pivot[anime_pivot.index== Anime].index[0]))
    else:
        print('{0}: {1} with distance: {2}'.format(i,anime_pivot.index[indice.flatten()[i]],distance.flatten()[i]))

print(distance.flatten())
print(indice.flatten())
# 「君の名は」を見たことがあるあなたにオススメのアニメは・・・
Anime = 'Kimi no Na wa.'
distance, indice = model_knn.kneighbors(anime_pivot.iloc[anime_pivot.index== Anime].values.reshape(1,-1),n_neighbors=11)
for i in range(0, len(distance.flatten())):
    if  i == 0:
        print('Recommendations if you like the anime {0}:\n'.format(anime_pivot[anime_pivot.index== Anime].index[0]))
    else:
        print('{0}: {1} with distance: {2}'.format(i,anime_pivot.index[indice.flatten()[i]],distance.flatten()[i]))
