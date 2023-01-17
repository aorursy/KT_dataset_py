import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



%matplotlib inline





plt.rcParams['figure.figsize'] = (6, 4)

plt.style.use('ggplot')

%config InlineBackend.figure_formats = {'png', 'retina'}

from surprise import Reader, Dataset, SVD, evaluate
anime = pd.read_csv('../input/anime.csv')
anime.head()
#df_title = anime.loc['anime_id','name']

df_title = anime.drop(['genre','type','episodes','rating','members'], axis=1)

df_title.set_index('anime_id', inplace = True)

df_title.head()
print(anime.shape)
user = pd.read_csv('../input/rating.csv')
user.head(10)
print(user.shape)
# merge 2 dataset

df = pd.merge(anime,user,on=['anime_id','anime_id'])

df= df[df.user_id <= 20000]

df.head(10)
f = ['count','mean']



df_movie_summary = df.groupby('anime_id')['rating_y'].agg(f)

df_movie_summary.index = df_movie_summary.index.map(int)

movie_benchmark = round(df_movie_summary['count'].quantile(0.8),0)

drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index



print('Movie minimum times of review: {}'.format(movie_benchmark))



df_cust_summary = df.groupby('user_id')['rating_y'].agg(f)

df_cust_summary.index = df_cust_summary.index.map(int)

cust_benchmark = round(df_cust_summary['count'].quantile(0.8),0)

drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index



print('Customer minimum times of review: {}'.format(cust_benchmark))
print('Original Shape: {}'.format(df.shape))

df = df[~df['anime_id'].isin(drop_movie_list)]

df = df[~df['user_id'].isin(drop_cust_list)]

print('After Trim Shape: {}'.format(df.shape))

print('-Data Examples-')

print(df.iloc[::5000000, :])
df_p = pd.pivot_table(df,values='rating_y',index='user_id',columns='anime_id')



print(df_p.shape)
reader = Reader()



# get just top 100K rows for faster run time

data = Dataset.load_from_df(df[['user_id', 'anime_id', 'rating_y']][:100000], reader)

data.split(n_folds=3)



svd = SVD()

evaluate(svd, data, measures=['RMSE', 'MAE'])
df_244 = df[(df['user_id'] == 244) & (df['rating_y'] == 10)]

#df_152 = df_152.set_index('anime_id')

#df_152 = df_152.join(df_title)['name']

df_244.head(5)
user_244 = df_title.copy()

user_244 = user_244.reset_index()

user_244 = user_244[~user_244['anime_id'].isin(drop_movie_list)]



# getting full dataset

data = Dataset.load_from_df(df[['user_id', 'anime_id', 'rating_y']], reader)



trainset = data.build_full_trainset()

svd.train(trainset)



user_244['Estimate_Score'] = user_244['anime_id'].apply(lambda x: svd.predict(144, x).est)



user_244 = user_244.drop('anime_id', axis = 1)



user_244 = user_244.sort_values('Estimate_Score', ascending=False)

user_244.head(10)

df_title['name']
def recommend(movie_title, min_count):

    print("For anime ({})".format(movie_title))

    print("- Top 10 anime recommended based on Pearsons'R correlation - ")

    i = int(df_title.index[df_title['name'] == movie_title][0])

    target = df_p[i]

    similar_to_target = df_p.corrwith(target)

    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])

    corr_target.dropna(inplace = True)

    corr_target = corr_target.sort_values('PearsonR', ascending = False)

    corr_target.index = corr_target.index.map(int)

    corr_target = corr_target.join(df_title).join(df_movie_summary)[['PearsonR', 'name', 'count', 'mean']]

    print(corr_target[corr_target['count']>min_count][:10].to_string(index=False))
recommend("Howl no Ugoku Shiro", 0)
recommend("Kimi no Iru Machi: Tasogare Kousaten", 0)