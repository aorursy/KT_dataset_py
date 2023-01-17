from PIL import Image

Image.open('../input/imagenes-propias/ejemplos.png')
import pandas as pd

import numpy as np

from scipy import stats

from scipy import sparse

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input/the-movies-dataset"))
movies = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')

movies = movies.drop([19730, 29503, 35587])

#Check EDA Notebook for how and why I got these indices.

movies['id'] = movies['id'].astype('int')

movies.head(2)
# Load the movielens-100k dataset (download it if needed),

data = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

data = data[data['movieId'].isin(movies['id'])]

data = data[['userId','movieId','rating']]

data.head()
Image.open('../input/imagenes-propias/filtros_colaborativos_intro.png')
# Import surprise modules

from surprise import Dataset

from surprise import Reader

from surprise import Dataset

from surprise import accuracy

from surprise.model_selection import cross_validate

from surprise.model_selection import train_test_split

from surprise.model_selection import GridSearchCV

from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering, KNNBaseline, KNNWithZScore, KNNWithMeans, KNNBasic, BaselineOnly, NormalPredictor

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# Dictionaty of user and items

users = list(data['userId'].unique())

items = list(data['movieId'].unique())



def movie_id_to_name(id):

    return movies.loc[movies['id']==id].original_title

movie_id_to_name(1371)
# Specific reader for surpirse to work

reader = Reader(rating_scale= (1,5))

data_sp = Dataset.load_from_df(data, reader=reader)
Image.open('../input/imagenes-propias/objetivo_filtros.png')
# We'll use the famous SVD algorithm.

algo =  [SVD(), NMF(), BaselineOnly(), SlopeOne(), KNNBasic()]



names = [algo[i].__class__.__name__ for i in range(len(algo))]



result = np.zeros((len(algo),4))            

for i in range(len(algo)):

    print('Procesando ', names[i], '...')

    tab = cross_validate(algo[i], data_sp,cv=5, verbose = False)

    rmse = np.mean(tab['test_rmse']).round(3)

    mae = np.mean(tab['test_mae']).round(3)

    ft = np.mean(tab['fit_time']).round(3)

    tt = np.mean(tab['test_time']).round(3)

    result[i]=[rmse, mae, ft, tt]

    

result = result.round(3)

result = pd.DataFrame(result, index = names, columns =['RMSE','MAE','fit_time','test_time'])
plt.figure(figsize=(12,8))

sns.heatmap(result,

            annot=True,linecolor="w",

            linewidth=2,cmap=sns.color_palette("Blues"))

plt.title("Data summary")

plt.show()
param_grid = {'n_epochs': [5, 10, 20], 'lr_all': [0.002, 0.005],

              'reg_all': [0.4, 0.6], 'n_factors' : [15, 30, 100]}



gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)



gs.fit(data_sp)



# best RMSE score

print(gs.best_score['rmse'])



# combination of parameters that gave the best RMSE score

print(gs.best_params['rmse'])
Image.open('../input/imagenes-propias/NMF_large.png')
Image.open('../input/imagenes-propias/NMF_explicacion_l.png')
Image.open('../input/imagenes-propias/NFM_final.png')
#Crea la matriz de predicciones

#n = len(users)

#m = len(items)

#recomendation = np.zeros((n,m))



#for k in users:

 #   u = users.index(k)

  #  for l in items:

   #     i = items.index(l)

    #    recomendation[u,i] = algo.predict(k,l)[3]
#algo.pu # Users

#algo.qi # Items

#sim_users = cosine_similarity(algo.pu, algo.pu)

#sim_items = cosine_similarity(algo.qi, algo.qi)
from lightfm import LightFM

from lightfm.evaluation import precision_at_k
# Function to create an interaction matrix dataframe from transactional type interactions

interactions = data.groupby(['userId', 'movieId'])['rating'].sum().unstack().reset_index().fillna(0).set_index('userId')

    

interactions.head()

interactions.shape
# Function to create a user dictionary based on their index and number in interaction dataset

user_id = list(interactions.index)

user_dict = {}

counter = 0 

for i in user_id:

    user_dict[i] = counter

    counter += 1



# Function to create an item dictionary based on their item_id and item name

movies = movies.reset_index()

item_dict ={}

for i in range(movies.shape[0]):

    item_dict[(movies.loc[i,'id'])] = movies.loc[i,'original_title']
item_dict[5]
# Function to run matrix-factorization algorithm

x = sparse.csr_matrix(interactions.values)

model = LightFM(no_components= 150, loss='warp')

model.fit(x,epochs=3000,num_threads = 4)
# Evaluate the trained model

k = 10

print('Train precision at k={}:\t{:.4f}'.format(k, precision_at_k(model, x, k=k).mean()))

#print('Test precision at k={}:\t\t{:.4f}'.format(k, precision_at_k(model, test_matrix, k=k).mean()))
def sample_recommendation_user(model, interactions, user_id, user_dict, 

                               item_dict,threshold = 0,nrec_items = 10, show = True):

    '''

    Function to produce user recommendations

    Required Input - 

        - model = Trained matrix factorization model

        - interactions = dataset used for training the model

        - user_id = user ID for which we need to generate recommendation

        - user_dict = Dictionary type input containing interaction_index as key and user_id as value

        - item_dict = Dictionary type input containing item_id as key and item_name as value

        - threshold = value above which the rating is favorable in new interaction matrix

        - nrec_items = Number of output recommendation needed

    Expected Output - 

        - Prints list of items the given user has already bought

        - Prints list of N recommended items  which user hopefully will be interested in

    '''

    n_users, n_items = interactions.shape

    user_x = user_dict[user_id]

    scores = pd.Series(model.predict(user_x,np.arange(n_items)))

    scores.index = interactions.columns

    scores = list(pd.Series(scores.sort_values(ascending=False).index))

    

    known_items = list(pd.Series(interactions.loc[user_id,:] \

                                 [interactions.loc[user_id,:] > threshold].index) \

                                .sort_values(ascending=False))

    #print(known_items)

    

    scores = [x for x in scores if x not in known_items]

    return_score_list = scores[0:nrec_items]

    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))

    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))

    if show == True:

        print("Known Likes:")

        counter = 1

        for i in known_items:

            #print(i)

            print(str(counter) + '- ' + i)

            counter+=1



        print("\n Recommended Items:")

        counter = 1

        for i in scores:

            #print(i)

            print(str(counter) + '- ' + i)

            counter+=1

    return return_score_list
## Calling 10 movie recommendation for user id 11

rec_list = sample_recommendation_user(model = model, 

                                      interactions = interactions, 

                                      user_id = 11, 

                                      user_dict = user_dict,

                                      item_dict = item_dict, 

                                      threshold = 4,

                                      nrec_items = 10,

                                      show = True)
def sample_recommendation_item(model,interactions,item_id,user_dict,item_dict,number_of_user):

    '''

    Funnction to produce a list of top N interested users for a given item

    Required Input -

        - model = Trained matrix factorization model

        - interactions = dataset used for training the model

        - item_id = item ID for which we need to generate recommended users

        - user_dict =  Dictionary type input containing interaction_index as key and user_id as value

        - item_dict = Dictionary type input containing item_id as key and item_name as value

        - number_of_user = Number of users needed as an output

    Expected Output -

        - user_list = List of recommended users 

    '''

    n_users, n_items = interactions.shape

    x = np.array(interactions.columns)

    scores = pd.Series(model.predict(np.arange(n_users), np.repeat(x.searchsorted(item_id),n_users)))

    user_list = list(interactions.index[scores.sort_values(ascending=False).head(number_of_user).index])

    return user_list 





## Calling 15 user recommendation for item id 1

sample_recommendation_item(model = model,

                           interactions = interactions,

                           item_id = 1,

                           user_dict = user_dict,

                           item_dict = item_dict,

                           number_of_user = 15)
def create_item_emdedding_distance_matrix(model,interactions):

    '''

    Function to create item-item distance embedding matrix

    Required Input -

        - model = Trained matrix factorization model

        - interactions = dataset used for training the model

    Expected Output -

        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items

    '''

    df_item_norm_sparse = sparse.csr_matrix(model.item_embeddings)

    similarities = cosine_similarity(df_item_norm_sparse)

    item_emdedding_distance_matrix = pd.DataFrame(similarities)

    item_emdedding_distance_matrix.columns = interactions.columns

    item_emdedding_distance_matrix.index = interactions.columns

    return item_emdedding_distance_matrix



## Creating item-item distance matrix

item_item_dist = create_item_emdedding_distance_matrix(model = model,

                                                       interactions = interactions)

## Checking item embedding distance matrix

#item_item_dist.head()
def item_item_recommendation(item_emdedding_distance_matrix, item_id, 

                             item_dict, n_items = 10, show = True):

    '''

    Function to create item-item recommendation

    Required Input - 

        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items

        - item_id  = item ID for which we need to generate recommended items

        - item_dict = Dictionary type input containing item_id as key and item_name as value

        - n_items = Number of items needed as an output

    Expected Output -

        - recommended_items = List of recommended items

    '''

    recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \

                                  sort_values(ascending = False).head(n_items+1). \

                                  index[1:n_items+1]))

    if show == True:

        print("Item of interest :{0}".format(item_dict[item_id]))

        print("Item similar to the above item:")

        counter = 1

        for i in recommended_items:

            print(str(counter) + '- ' +  item_dict[i])

            counter+=1

    return recommended_items



## Calling 10 recommended items for item id 

rec_list = item_item_recommendation(item_emdedding_distance_matrix = item_item_dist,

                                    item_id = 6,

                                    item_dict = item_dict,

                                    n_items = 10)
Image.open('../input/imagenes-propias/filtros_colaborativos_intro.png')
from sklearn.metrics.pairwise import cosine_similarity

from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import wordnet



import warnings; warnings.simplefilter('ignore')
md = movies



md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

links_small = pd.read_csv('../input/the-movies-dataset/links_small.csv')

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')



#Check EDA Notebook for how and why I got these indices.

smd = md[md['id'].isin(links_small)]



credits = pd.read_csv('../input/the-movies-dataset/credits.csv')

keywords = pd.read_csv('../input/the-movies-dataset/keywords.csv')

keywords['id'] = keywords['id'].astype('int')

credits['id'] = credits['id'].astype('int')

md['id'] = md['id'].astype('int')



md = md.merge(credits, on='id')

md = md.merge(keywords, on='id')



smd = md[md['id'].isin(links_small)]



smd['cast'] = smd['cast'].apply(literal_eval)

smd['crew'] = smd['crew'].apply(literal_eval)

smd['keywords'] = smd['keywords'].apply(literal_eval)

smd['cast_size'] = smd['cast'].apply(lambda x: len(x))

smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan

smd['director'] = smd['crew'].apply(get_director)

smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])



smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))

smd['director'] = smd['director'].apply(lambda x: [x,x, x])



#Keywords

s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'keyword'

s = s.value_counts()

s = s[s > 1]



stemmer = SnowballStemmer('english')



def filter_keywords(x):

    words = []

    for i in x:

        if i in s:

            words.append(i)

    return words



smd['keywords'] = smd['keywords'].apply(filter_keywords)

smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])

smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']

smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))



count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

count_matrix = count.fit_transform(smd['soup'])



item_features = count_matrix
Image.open('../input/imagenes-propias/espacio_vectorial_rec.png')
cosine_sim = linear_kernel(item_features, item_features)



smd = smd.reset_index()

titles = smd['title']

indices = pd.Series(smd.index, index=smd['title'])



def get_recommendations(title):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices]



get_recommendations('The Godfather').head(10)
Image.open('../input/imagenes-propias/contenido_3.png')
x = sparse.csr_matrix(interactions.values)

model = LightFM(no_components= 15, loss='warp',k=5)

model.fit(x,item_features=item_features,epochs=4,num_threads = 2)
## Calling 10 movie recommendation for user id 11

rec_list = sample_recommendation_user(model = model, 

                                      interactions = interactions, 

                                      user_id = 11, 

                                      user_dict = user_dict,

                                      item_dict = item_dict, 

                                      threshold = 4,

                                      nrec_items = 10,

                                      show = True)