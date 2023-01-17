import os

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 10)

pd.set_option('display.width', 1000)
from plotly.offline import init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)
train_df = pd.read_csv('../input/restaurantrecommendationdata/train_100k.csv')
test_df = pd.read_csv('../input/restaurantrecommendationdata/test_100k.csv')
print(train_df.shape)

print(test_df.shape)
train_df.restaurent_rating = train_df.restaurent_rating.astype('float')

test_df.restaurent_rating = test_df.restaurent_rating.astype('float')
train_df.head()
test_df.head()
train_df.info()
test_df.info()
train_explicit = train_df[['customer_id','restaurant_id','restaurent_rating']]

train_implicit = train_df[['customer_id','restaurant_id','restaurent_rating','restaurent_tag_name']]
test_explicit = test_df[['customer_id','restaurant_id','restaurent_rating']]

test_implicit = test_df[['customer_id','restaurant_id','restaurent_rating','restaurent_tag_name']]
train_explicit.head()
train_implicit.head()
train_explicit.restaurant_id.value_counts().sort_values(ascending=True)[:10]
train_explicit.customer_id.value_counts().sort_values(ascending=False)[:10]
train_explicit.restaurent_rating.value_counts()
test_explicit.restaurant_id = test_explicit.restaurant_id.astype('int64')
test_explicit.restaurant_id.value_counts()[:10]
test_explicit.customer_id.value_counts()[:10]
test_explicit.restaurent_rating.value_counts()
rating_average = train_explicit.groupby('restaurant_id')['restaurent_rating'].mean()

rating_count = train_explicit.groupby('restaurant_id')['restaurent_rating'].count()
rating_average = rating_average.to_frame()

rating_average.rename(columns={'restaurent_rating':'rating_average'}, inplace= True)
rating_count = rating_count.to_frame()

rating_count.rename(columns={'restaurent_rating':'rating_count'}, inplace= True)
df2= rating_average.merge(rating_count,on='restaurant_id')
C= df2['rating_average'].mean()

C
m= df2['rating_count'].quantile(0.3)

m
restaurants = df2.copy().loc[df2['rating_count'] >= m]

restaurants.shape
def weighted_rating(x, m=m, C=C):

    v = x['rating_count']

    R = x['rating_average']

    return (v/(v+m) * R) + (m/(m+v) * C)
restaurants['score'] = restaurants.apply(weighted_rating, axis=1)
restaurants = restaurants.sort_values('score', ascending=False)

restaurants.index.to_list()[:5]
print('>> Importing Libraries')



import pandas as pd



from surprise import Reader, Dataset, SVD



from surprise.accuracy import rmse, mae

from surprise.model_selection import cross_validate



print('>> Libraries imported.')
train_explicit.isna().sum()
n_restaurants = train_explicit['restaurant_id'].nunique()

n_customers = train_explicit['customer_id'].nunique()

print(f'Number of unique restaurants: {n_restaurants}')

print(f'Number of unique n_customers: {n_customers}')
available_ratings = train_explicit['restaurent_rating'].count()

total_ratings = n_restaurants*n_customers

missing_ratings = total_ratings - available_ratings

sparsity = (missing_ratings/total_ratings) * 100

print(f'Sparsity: {sparsity}')
train_explicit['restaurent_rating'].value_counts().plot(kind='bar')
filter_restaurants = train_explicit['restaurant_id'].value_counts() > 5

filter_restaurants = filter_restaurants[filter_restaurants].index.tolist()
filter_customers = train_explicit['customer_id'].value_counts() > 5

filter_customers = filter_customers[filter_customers].index.tolist()
print(f'Original shape: {train_explicit.shape}')

df = train_explicit[(train_explicit['restaurant_id'].isin(filter_restaurants)) & (train_explicit['customer_id'].isin(filter_customers))]

print(f'New shape: {df.shape}')
df.head()
cols = ['customer_id','restaurant_id','restaurent_rating']
reader = Reader(rating_scale = (0.5, 5))

data = Dataset.load_from_df(df[cols], reader)
trainset = data.build_full_trainset()

antiset = trainset.build_anti_testset()
algo = SVD(n_epochs =25, verbose = True)
cross_validate(algo, data, measures = ['RMSE', 'MAE'], cv=5, verbose= True)

print('>> Training Done')
predictions = algo.test(antiset)
predictions[0]
from collections import defaultdict

def get_top_n(predictions, n=5):

    top_n = defaultdict(list)

    for uid, iid, _, est, _ in predictions:

        top_n[uid].append((iid, est))

        

    for uid, user_ratings in top_n.items():

        user_ratings.sort(key = lambda x: x[1], reverse = True)

        top_n[uid] = user_ratings[:n]

        

    return top_n



top_n = get_top_n(predictions, n=5)



for uid, user_ratings in top_n.items():

    print(uid, [iid for (iid, rating) in user_ratings])
from surprise import Reader

from surprise import Dataset

from surprise.model_selection import cross_validate

from surprise import NormalPredictor

from surprise import KNNBasic

from surprise import KNNWithMeans

from surprise import KNNWithZScore

from surprise import KNNBaseline

from surprise import SVD

from surprise import BaselineOnly

from surprise import SVDpp

from surprise import NMF

from surprise import SlopeOne

from surprise import CoClustering

from surprise.accuracy import rmse

from surprise import accuracy

from surprise.model_selection import train_test_split

from surprise.model_selection import GridSearchCV
df = train_explicit
data = df['restaurent_rating'].value_counts().sort_index(ascending=False)

trace = go.Bar(x = data.index,

               text = ['{:.1f} %'.format(val) for val in (data.values / df.shape[0] * 100)],

               textposition = 'auto',

               textfont = dict(color = '#000000'),

               y = data.values,

               )

# Create layout

layout = dict(title = 'Distribution Of {} restaurant-ratings'.format(df.shape[0]),

              xaxis = dict(title = 'Rating'),

              yaxis = dict(title = 'Count'))

# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
data = df.groupby('restaurant_id')['restaurent_rating'].count().clip(lower=5000)



# Create trace

trace = go.Histogram(x = data.values,

                     name = 'Ratings',

                     xbins = dict(start = 5000,

                                  end = 6000,

                                  size = 100))

# Create layout

layout = go.Layout(title = 'Distribution Of Number of Ratings Per restaurant',

                   xaxis = dict(title = 'Number of Ratings Per Restaurant'),

                   yaxis = dict(title = 'Count'),

                   bargap = 0.2)



# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
df.groupby('restaurant_id')['restaurent_rating'].count().reset_index().sort_values('restaurent_rating', ascending=False)[:10]
data = df.groupby('customer_id')['restaurent_rating'].count()



# Create trace

trace = go.Histogram(x = data.values,

                     name = 'Ratings',

                     xbins = dict(start = 0,

                                  end = 50,

                                  size = 2))

# Create layout

layout = go.Layout(title = 'Distribution Of Number of Ratings Per Customer',

                   xaxis = dict(title = 'Ratings Per Customer'),

                   yaxis = dict(title = 'Count'),

                   bargap = 0.2)



# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
df.groupby('customer_id')['restaurent_rating'].count().reset_index().sort_values('restaurent_rating', ascending=False)[:10]
filter_restaurants = train_explicit['restaurant_id'].value_counts() > 5

filter_restaurants = filter_restaurants[filter_restaurants].index.tolist()
filter_customers = train_explicit['customer_id'].value_counts() > 5

filter_customers = filter_customers[filter_customers].index.tolist()
print(f'Original shape: {train_explicit.shape}')

df = train_explicit[(train_explicit['restaurant_id'].isin(filter_restaurants)) & (train_explicit['customer_id'].isin(filter_customers))]

print(f'New shape: {df.shape}')
reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(df[['customer_id', 'restaurant_id', 'restaurent_rating']], reader)
benchmark = []

for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:

    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)

    tmp = pd.DataFrame.from_dict(results).mean(axis=0)

    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))

    benchmark.append(tmp)

    

pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')  
print('Using ALS')

bsl_options = {'method': 'als',

               'n_epochs': 5,

               'reg_u': 12,

               'reg_i': 5}

algo = BaselineOnly(bsl_options=bsl_options)

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)
trainset, testset = train_test_split(data, test_size=0.25)

algo = BaselineOnly(bsl_options=bsl_options)

predictions = algo.fit(trainset).test(testset)

accuracy.rmse(predictions)


def get_Iu(uid):

    """ return the number of items rated by given user

    args: 

      uid: the id of the user

    returns: 

      the number of items rated by the user

    """

    try:

        return len(trainset.ur[trainset.to_inner_uid(uid)])

    except ValueError: # user was not part of the trainset

        return 0

    

def get_Ui(iid):

    """ return number of users that have rated given item

    args:

      iid: the raw id of the item

    returns:

      the number of users that have rated the item.

    """

    try: 

        return len(trainset.ir[trainset.to_inner_iid(iid)])

    except ValueError:

        return 0

    

df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])

df['Iu'] = df.uid.apply(get_Iu)

df['Ui'] = df.iid.apply(get_Ui)

df['err'] = abs(df.est - df.rui)

best_predictions = df.sort_values(by='err')[:10]

worst_predictions = df.sort_values(by='err')[-10:]
best_predictions
worst_predictions
df = train_implicit[(train_implicit['restaurant_id'].isin(filter_restaurants)) & (train_implicit['customer_id'].isin(filter_customers))]
df = df.sample(n=10000)
df['restaurent_tag_name'] = df['restaurent_tag_name'].apply(lambda x: x.replace(',',' '))
df['restaurent_tag_name'].head()
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(stop_words='english')

df['restaurent_tag_name'] = df['restaurent_tag_name'].fillna('')

tfidf_matrix = tfidf.fit_transform(df['restaurent_tag_name'])

tfidf_matrix.shape
from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

df['restaurent_tag_name'] = df['restaurent_tag_name'].fillna('')

count_matrix = count.fit_transform(df['restaurent_tag_name'])

count_matrix.shape
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
df = df.reset_index()

indices = pd.Series(df.index, index=df['restaurent_tag_name']).drop_duplicates()
indices.index.unique()
def get_recommendations(variable,indices,df,cosine_sim):

    idx = indices[variable]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1].all(), reverse=True)

    sim_scores = sim_scores[1:5]

    df_indices = [i[0] for i in sim_scores]

    return df['restaurant_id'].iloc[df_indices]
get_recommendations('Asian Dimsum Grills Japanese Rice Soups',indices,df,cosine_sim)
get_recommendations('Asian Dimsum Grills Japanese Rice Soups',indices,df,cosine_sim2)