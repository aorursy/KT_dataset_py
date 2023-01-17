import pandas as pd
import time

df_tags= pd.read_csv('../input/tags.csv')
df_movies = pd.read_csv('../input/movies.csv')
df_movies['genres'] = df_movies['genres'].apply(lambda x: x.split('|'))
df_tags_combined = df_tags.groupby('movieId').apply(lambda x: list(x['tag'])).reset_index().rename(columns={0:'tags'})
df_movies = pd.merge(df_movies, df_tags_combined, on = 'movieId', how = 'left')

df_movies['tags'] = df_movies['tags'].apply(lambda x: x if isinstance(x,list) else [])
df_movies['keywords'] = df_movies['genres']+df_movies['tags']
df_movies['keywords'] = df_movies['keywords'].apply(lambda x: set([str.lower(i.replace(" ", "")) for i in x]))
df_movies.set_index('movieId', inplace= True)

all_keywords = set()
for this_movie_keywords in df_movies['keywords']:
    all_keywords = all_keywords.union(this_movie_keywords)


df_movies
df_ratings = pd.read_csv('../input/ratings.csv')
df_mxk = pd.DataFrame(0, index = df_movies.reset_index()['movieId'].unique(), columns = all_keywords)
df_mxk['mean_rating'] = df_ratings.groupby('movieId')['rating'].mean()

for index,row in df_mxk.iterrows():
    df_mxk.loc[index,df_movies.loc[index]['keywords']] = 1

df_mxk['mean_rating'].fillna(df_mxk['mean_rating'].mean(), inplace=True)
df_mxk = df_mxk.loc[:,df_mxk.sum() > 5]
df_mxk
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(random_state=42)
X = df_mxk.drop('mean_rating', axis = 1).as_matrix()
y = df_mxk['mean_rating'].as_matrix()

reg.fit(X,y)
keyword_scores = pd.Series(reg.feature_importances_ , index = df_mxk.drop('mean_rating', axis=1).columns)
keyword_frequency = df_mxk.sum()
df_movies['chief_keyword'] = df_movies['keywords'].apply(lambda x: (keyword_scores[x]/keyword_frequency).idxmax())
df_movies
all_chief_keywords = df_movies['chief_keyword'].unique()
df_uxk = pd.DataFrame(0, index = df_ratings['userId'].unique(), columns = all_chief_keywords)


for row in df_ratings.itertuples(index=True, name='Pandas'):
    this_movie_chief_keyword = df_movies.loc[getattr(row, 'movieId'), 'chief_keyword']
    this_user_this_movie_rating = getattr(row, 'rating')
    this_user_id = getattr(row, 'userId')
    df_uxk.loc[this_user_id,this_movie_chief_keyword] += this_user_this_movie_rating




df_uxk

nok = len(all_chief_keywords)
df_co_rating = pd.DataFrame(0, index = all_chief_keywords, columns = all_chief_keywords)


for index,row in df_uxk.iterrows():
    print (index)
    for i, first_keyword in enumerate(all_chief_keywords):
        for j in range(i+1,nok):
            second_keyword = all_chief_keywords[j]
            df_co_rating.loc[first_keyword,second_keyword] += min(row[first_keyword],row[second_keyword])
            df_co_rating.loc[second_keyword,first_keyword] = df_co_rating.loc[first_keyword,second_keyword]
         


import scipy.stats


def sim_matrix(co): # returns the similarity matrix for the given co-occurence matrix
    chief_keywords = co.columns
    df_sim = pd.DataFrame(index = co.index, columns = co.columns)
    f = co.sum()
    n = sum(f)

    for first_chief_keyword in chief_keywords:
        for second_chief_keyword in chief_keywords:
            k11 = co.loc[first_chief_keyword][second_chief_keyword]
            k12 = f[first_chief_keyword]-k11
            k21 = f[second_chief_keyword]-k11
            k22 = n - k12 - k21 + k11
            df_sim.loc[first_chief_keyword][second_chief_keyword], p, dof, expctd= scipy.stats.chi2_contingency([[k11,k12],[k21,k22]], lambda_="log-likelihood")
            if ((k11/k21) < f[first_chief_keyword]/(n-f[first_chief_keyword])):
                df_sim.loc[first_chief_keyword][second_chief_keyword] = 0
                
    return df_sim

df_sim_chief_keyword = sim_matrix(df_co_rating)
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
df_sim_chief_keyword
df_sim_chief_keyword['children'].sort_values(ascending = False)


from surprise import SVD, Reader, Dataset

reader = Reader()
data = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
svd = SVD()
trainset = data.build_full_trainset()
svd.train(trainset)
def collaborative(userId):
    df_movies['est'] = df_movies.reset_index()['movieId'].apply(lambda x: svd.predict(userId,x).est)
    return df_movies.sort_values('est', ascending=False).head(10)
    
collaborative(1)
df_movies[df_movies['chief_keyword'] == 'superhero']

title_to_id = df_movies.reset_index()[['movieId', 'title']].set_index('title')

def hybrid(userId, title):
    this_movie_id = title_to_id.loc[title]
    all_movieids = list(df_movies.index)
    sim_scores_series = pd.Series(0,index = all_movieids)
    for movieid in all_movieids:
        sim_scores_series.loc[movieid] = df_sim_chief_keyword.loc[df_movies.loc[this_movie_id,'chief_keyword'],df_movies.loc[movieid,'chief_keyword']].iloc[0]
        
    top_25_ids = sim_scores_series.sort_values(ascending=False)[:26].index
    df_movies_top25 = df_movies.loc[top_25_ids].reset_index()
    
    df_movies_top25['est'] = df_movies_top25['index'].apply(lambda x: svd.predict(userId,x).est)
    
    #Sort the movies in decreasing order of predicted rating
    df_movies_top25 = df_movies_top25.sort_values('est', ascending=False)
    
    #Return the top 10 movies as recommendations
    return df_movies_top25.head(10)

hybrid(1, 'Spider-Man (2002)')
hybrid(1, 'Spider-Man (2002)')
