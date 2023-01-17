# ref: https://www.codexa.net/collaborative-filtering-k-nearest-neighbor/
# ref: https://www.kaggle.com/ajmichelutti/collaborative-filtering-on-anime-data

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# data analysis and wrangling
import pandas as pd
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20
import numpy as np
import random as rnd
import operator
import itertools
import collections

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# machine learning
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_similarity_score
from xgboost import XGBClassifier

# warnings
import warnings
warnings.filterwarnings('ignore')
# アニメ一覧
df_anime = pd.read_csv('../input/anime.csv')
# ユーザー評価
df_rating = pd.read_csv('../input/rating.csv')
df_anime.head()
# 欠損値を確認
df_anime.isnull().sum() 
# 欠損値を確認
df_rating.isnull().sum() 
df_anime.describe()
df_anime.describe(include=['O'])
df_rating.head()
df_rating.describe()
# 高評価トップ10
df_anime.sort_values('rating', ascending=False).loc[:10]
# メンバー数トップ10
df_anime.sort_values('members', ascending=False)[:10]
# メンバー数が少ない（人気が少ない）アニメはお勧め対象外
df_anime = df_anime[df_anime['members'] >= 10000]

df_anime.shape[0]
# アニメ名で部分一致検索
def search_anime(name):
    print(df_anime.loc[df_anime['name'].str.contains(name, case=False), 'name'].values)
    
search_anime('madoka')
# genreを配列にする
df_anime['genre'] = df_anime['genre'].apply(lambda x: x.split(', ') if type(x) is str else [])
genre_data = itertools.chain(*df_anime['genre'].values.tolist()) # フラットな配列に変更
genre_counter = collections.Counter(genre_data) # ジャンル別カウント

df_genre = pd.DataFrame.from_dict(genre_counter, orient='index').reset_index().rename(columns={'index': 'genre', 0:'count'})
df_genre.sort_values('count', ascending=False, inplace=True)
# ジャンル数ランキングを表示
figure, ax = plt.subplots(figsize=(8, 12))
sns.barplot(x='count', y='genre', data=df_genre, color='b')
ax.set(ylabel='Genre', xlabel='Anime Count')
genre_map = {genre: idx for idx, genre in enumerate(genre_counter.keys())} # ジャンル名: Index

# ジャンルをOne-hot表現に変換
def extract_feature(genre):
    feature = np.zeros(len(genre_map.keys()), dtype=int)
    feature[[genre_map[idx] for idx in genre]] += 1
    return feature

df_anime_feature = pd.concat([df_anime['anime_id'], df_anime['name'], df_anime['genre']], axis=1)
df_anime_feature['genre'] = df_anime_feature['genre'].apply(lambda x: extract_feature(x))
df_anime_feature.head()
# ジャンルが似ているアニメを取得
def similar_genre_animes(anime_name, num=100, verbose=False):
    anime_index = df_anime_feature[df_anime_feature['name'] == anime_name].index[0]
    s_anime = df_anime_feature[df_anime_feature.index== anime_index]
    anime_name = s_anime['name'].values[0]
    anime_genre = s_anime['genre'].values[0]
    df_search = df_anime_feature.drop(anime_index)
    # Jaccard 係数が高いアニメを取得
    # 集合 X と集合 Y がどれくらい似ているか
    # A または B に含まれている要素のうち A にも B にも含まれている要素の割合
    # ref: https://mathwords.net/jaccardkeisu
    df_search['jaccard'] = df_search['genre'].apply(lambda x: jaccard_similarity_score(anime_genre, x))
    df_result = df_search.sort_values('jaccard', ascending=False).head(num)
    if verbose:
        print('【{}】　にジャンルが似ているアニメ'.format(anime_name))
        for idx, res in df_result.iterrows():
            print('\t{}'.format(res['name']))
        print()
    return df_result
    
# 【魔法少女まどか☆マギカ】にジャンルが似ているアニメトップ10
_ = similar_genre_animes('Mahou Shoujo Madoka★Magica', num=10, verbose=True)
# ユーザー評価にアニメ情報を結合する
df_merge = df_rating.merge(df_anime, left_on='anime_id', right_on='anime_id', suffixes=['_user', ''])
round(df_merge.describe(), 2)
# 使わないカラムを削除
df_merge = df_merge[['user_id', 'anime_id', 'name', 'rating_user']]
# 重複している評価を削除
df_merge = df_merge.drop_duplicates(['user_id', 'name'])

df_merge.head()
# 行をアニメ、カラムをユーザーに変形
df_anime_pivot = df_merge.pivot(index='anime_id', columns='user_id', values='rating_user').fillna(0)
# 疎行列に変換
# ref: http://hamukazu.com/2014/09/26/scipy-sparse-basics/
anime_pivot_sparse = csr_matrix(df_anime_pivot)
# K近傍法で学習
# ref: https://qiita.com/yshi12/items/26771139672d40a0be32
# brute:  力任せ検索
# ref: https://ja.wikipedia.org/wiki/%E5%8A%9B%E3%81%BE%E3%81%8B%E3%81%9B%E6%8E%A2%E7%B4%A2
# cosine: コサイン類似度
# ref: http://www.cse.kyoto-su.ac.jp/~g0846020/keywords/cosinSimilarity.html
knn = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='cosine')
model_knn = knn.fit(anime_pivot_sparse)
# ユーザー評価が似ているアニメを取得
def similar_rating_animes(anime_name, num=100, verbose=False):
    anime_id = df_anime.loc[df_anime['name'] == anime_name, 'anime_id'].values[0]
    users_rating = df_anime_pivot.iloc[df_anime_pivot.index == anime_id].values.reshape(1, -1)
    # コサイン類似度が近いアニメを取得
    # イメージ的にはユーザーが同じような評価を付けているアニメ
    distance, indice = model_knn.kneighbors(users_rating, n_neighbors=num)
    df_result = df_anime_pivot.iloc[indice.flatten()]
    df_result['distance'] = distance.flatten()
    df_result['name'] = df_result.index.map(lambda x: df_anime.loc[df_anime['anime_id'] == x, 'name'].values[0])
    df_result = df_result.drop(anime_id)
    df_result = df_result.sort_values('distance', ascending=True).head(num)
    df_result = df_result.reset_index()
    df_result = df_result[['anime_id', 'name', 'distance']]
    df_result.columns = ['anime_id', 'name', 'distance']
    if verbose:
        print('【{0}】 とユーザー評価が似ているアニメ'.format(anime_name))
        for idx, res in df_result.iterrows():
            print('\t{0}'.format(res['name']))
    return df_result
    
# 【魔法少女まどか☆マギカ】にユーザー評価が似ているアニメトップ10
_ = similar_rating_animes('Mahou Shoujo Madoka★Magica', num=10, verbose=True)
# ユーザー評価とジャンル評価を組み合わせてお勧めなアニメを取得する
def similar_animes(anime_name, genre_weight=0.5, rating_weight=0.5, num=20, verbose=False):
    df_genre_similar = similar_genre_animes(anime_name) # ジャンル高評価
    df_rating_similar = similar_rating_animes(anime_name) # ユーザー高評価
    #  ジャンル評価とユーザー評価を外部結合でマージしたテーブルを作成
    df_similar_merge = df_rating_similar.merge(
        df_genre_similar, left_on=['anime_id', 'name'], right_on=['anime_id', 'name'], how='outer')
    # ジャンル高評価に引っかかっていない場合はJaccard 係数は0（最低評価）
    df_similar_merge['jaccard'].fillna(0, inplace=True)
    # ユーザー高評価に引っかかっていない場合はコサイン類似度は1（最低評価）
    df_similar_merge['distance'].fillna(1, inplace=True) 
    # 各評価を0~1で正規化
    df_score_genre= df_similar_merge['jaccard'] / df_similar_merge['jaccard'].max()
    df_score_rating = (1.0 - df_similar_merge['distance']) / df_similar_merge['distance'].max()
    #　総合評価（最大1）を取得
    df_similar_merge['score'] =  (genre_weight * df_score_genre + rating_weight * df_score_rating) / 2
    # 総合評価が高いアニメ一覧を取得
    df_result = df_similar_merge.sort_values('score', ascending=False).head(num)
    df_result.reset_index(inplace=True)
    df_result = df_result[['anime_id', 'name', 'score']]
    if verbose:
        print('【{0}】 が好きな人におすすめなアニメ'.format(anime_name))
        for idx, res in df_result.iterrows():
            print('\t{0}'.format(res['name']))
    return df_result
    
# 【魔法少女まどか☆マギカ】が好きな人にお勧めなアニメトップ10
_ = similar_animes('Mahou Shoujo Madoka★Magica', verbose=True)