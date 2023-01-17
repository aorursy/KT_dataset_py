# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
books= pd.read_csv('/kaggle/input/goodbooks-10k/books.csv',error_bad_lines = False)
books.head()
books.shape
ratings = pd.read_csv('/kaggle/input/goodbooks-10k/ratings.csv')



ratings.head()
tags = pd.read_csv('/kaggle/input/goodbooks-10k/book_tags.csv')

tags.tail()
btags = pd.read_csv('/kaggle/input/goodbooks-10k/tags.csv')

btags.tail()
ratings=ratings.sort_values("user_id")

ratings.shape
ratings.drop_duplicates(subset =["user_id","book_id"], 

                     keep = False, inplace = True) 

ratings.shape
print(books.shape)

books.drop_duplicates(subset='original_title',keep=False,inplace=True)

print(books.shape)
print(btags.shape)

btags.drop_duplicates(subset='tag_id',keep=False,inplace=True)

print(btags.shape)
print(tags.shape)

tags.drop_duplicates(subset=['tag_id','goodreads_book_id'],keep=False,inplace=True)

print(tags.shape)
joint_tags=pd.merge(tags,btags,left_on='tag_id',right_on='tag_id',how='inner')
top_rated=books.sort_values('average_rating', ascending=False)

top10=top_rated.head(10)

f=['title','small_image_url']

displ=(top10[f])

displ.set_index('title', inplace=True)

from IPython.display import Image, HTML



def path_to_image_html(path):

    '''

     This function essentially convert the image url to 

     '<img src="'+ path + '"/>' format. And one can put any

     formatting adjustments to control the height, aspect ratio, size etc.

     within as in the below example. 

    '''



    return '<img src="'+ path + '""/>'



HTML(displ.to_html(escape=False ,formatters=dict(small_image_url=path_to_image_html),justify='center'))
pop10=books.sort_values(by='ratings_count', ascending=False)

f=['title','small_image_url']

pop10=pop10.head(10)



pop10=(pop10[f])

pop10=pop10.set_index('title')
HTML(pop10.to_html(escape=False ,formatters=dict(small_image_url=path_to_image_html),justify='center'))
import seaborn as sns

plt.figure(figsize=(16,8))

sns.distplot(a=books['average_rating'], kde=True, color='r')
no_of_ratings_per_book=ratings.groupby('book_id').count()
no_of_ratings_per_book
plt.figure(figsize=(16,8))

sns.distplot(a=no_of_ratings_per_book['rating'], color='g')
books.head(2)
f=['authors', 'average_rating']

top_authors=top_rated[f]

top_authors=top_authors.head(20)
fig = px.bar(top_authors, x='authors', y='average_rating', color ='average_rating')

fig.show()
p=joint_tags.groupby('tag_name').count()
p=p.sort_values(by='count', ascending=False)

p
genres=["Art", "Biography", "Business", "Chick Lit", "Children's", "Christian", "Classics", "Comics", "Contemporary", "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction", "Gay and Lesbian", "Graphic Novels", "Historical Fiction", "History", "Horror", "Humor and Comedy", "Manga", "Memoir", "Music", "Mystery", "Nonfiction", "Paranormal", "Philosophy", "Poetry", "Psychology", "Religion", "Romance", "Science", "Science Fiction", "Self Help", "Suspense", "Spirituality", "Sports", "Thriller", "Travel", "Young Adult"]

for i in range(len(genres)):

    genres[i]=genres[i].lower()
new_tags=p[p.index.isin(genres)]
import plotly.graph_objects as go



fig = go.Figure(go.Bar(

            x=new_tags['count'],

            y=new_tags.index,

            orientation='h'))



fig.show()
fig = px.line(books, y="books_count", x="average_rating", title='Book Count VS Average Rating')

fig.show()
dropna= books.dropna()

fig = px.treemap(dropna, path=['original_publication_year','language_code', "average_rating"],

                  color='average_rating')

fig.show()
books['length-title']=books['original_title'].str.len()
plt.figure(figsize=(16,8))

sns.regplot(x=books['length-title'], y=books['average_rating'])
from wordcloud import WordCloud, STOPWORDS 

text = new_tags.index.values 



wordcloud = WordCloud().generate(str(text))

plt.figure(figsize = (8, 8), facecolor = None)

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
books.columns
to_read=pd.read_csv("/kaggle/input/goodbooks-10k/to_read.csv")

to_r= books.merge(to_read, left_on='book_id', right_on='book_id', how='inner')
to_r=to_r.groupby('original_title').count()
to_r=to_r.sort_values(by='id', ascending=False)

to_r20=to_r.head(20)
fig = px.bar(to_r20, x=to_r20.index, y='id', color ='id')

fig.show()
to_read1=to_read.groupby('user_id').count()
from collections import Counter

c=Counter(list(to_read1['book_id']))
import plotly.graph_objects as go



fig = go.Figure(data=[go.Scatter(

    x=list(c.keys()), y=list(c.values()),

    mode='markers')

])



fig.show()
fillnabooks= books.fillna('')
def clean_data(x):

        return str.lower(x.replace(" ", ""))
features=['original_title','authors','average_rating']

fillednabooks=fillnabooks[features]
fillednabooks = fillednabooks.astype(str)

fillednabooks.dtypes
for feature in features:

    fillednabooks[feature] = fillednabooks[feature].apply(clean_data)

    

fillednabooks.head(2)
def create_soup(x):

    return x['original_title']+ ' ' + x['authors'] + ' ' + x['average_rating']
fillednabooks['soup'] = fillednabooks.apply(create_soup, axis=1)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity



count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(fillednabooks['soup'])



cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
fillednabooks=fillednabooks.reset_index()

indices = pd.Series(fillednabooks.index, index=fillednabooks['original_title'])
def get_recommendations_new(title, cosine_sim=cosine_sim2):

    title=title.replace(' ','').lower()

    idx = indices[title]



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return list(books['original_title'].iloc[movie_indices])
l=get_recommendations_new('The Hobbit', cosine_sim2)

fig = go.Figure(data=[go.Table(header=dict(values=l,fill_color='orange'))

                     ])

fig.show()
l=get_recommendations_new('Harry Potter and The Chamber of Secrets', cosine_sim2)

fig = go.Figure(data=[go.Table(header=dict(values=l,fill_color='orange'))

                     ])

fig.show()
usecols=['book_id', 'original_title']

books_col=books[usecols]
books_col.dropna()
from scipy.sparse import csr_matrix

# pivot ratings into movie features

df_book_features = ratings.pivot(index='book_id',columns='user_id',values='rating').fillna(0)

mat_book_features = csr_matrix(df_book_features.values)
df_book_features.head()
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)





num_users = len(ratings.user_id.unique())

num_items = len(ratings.book_id.unique())

print('There are {} unique users and {} unique movies in this data set'.format(num_users, num_items))
ratings=ratings.dropna()
df_ratings_cnt_tmp = pd.DataFrame(ratings.groupby('rating').size(), columns=['count'])

df_ratings_cnt_tmp.head(10)
total_cnt = num_users * num_items

rating_zero_cnt = total_cnt - ratings.shape[0]



df_ratings_cnt = df_ratings_cnt_tmp.append(

    pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),

    verify_integrity=True,

).sort_index()

df_ratings_cnt
import numpy as np

df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])

df_ratings_cnt



import matplotlib.pyplot as plt





get_ipython().run_line_magic('matplotlib', 'inline')

ax = df_ratings_cnt[['count']].reset_index().rename(columns={'index': 'rating score'}).plot(

    x='rating score',

    y='count',

    kind='bar',

    figsize=(12, 8),

    title='Count for Each Rating Score (in Log Scale)',

    logy=True,

    fontsize=12,color='black'

)

ax.set_xlabel("book rating score")

ax.set_ylabel("number of ratings")



df_books_cnt = pd.DataFrame(ratings.groupby('book_id').size(), columns=['count'])

df_books_cnt.head()
#now we need to take only books that have been rated atleast 60 times to get some idea of the reactions of users towards it



popularity_thres = 60

popular_movies = list(set(df_books_cnt.query('count >= @popularity_thres').index))

df_ratings_drop = ratings[ratings.book_id.isin(popular_movies)]

print('shape of original ratings data: ', ratings.shape)

print('shape of ratings data after dropping unpopular movies: ', df_ratings_drop.shape)
# get number of ratings given by every user

df_users_cnt = pd.DataFrame(df_ratings_drop.groupby('user_id').size(), columns=['count'])

df_users_cnt.head()
ratings_thres = 50

active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))

df_ratings_drop_users = df_ratings_drop[df_ratings_drop.user_id.isin(active_users)]

print('shape of original ratings data: ', ratings.shape)

print('shape of ratings data after dropping both unpopular movies and inactive users: ', df_ratings_drop_users.shape)

book_user_mat = df_ratings_drop_users.pivot(index='book_id', columns='user_id', values='rating').fillna(0)

book_user_mat


book_user_mat_sparse = csr_matrix(book_user_mat.values)

book_user_mat_sparse
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

# fit

model_knn.fit(book_user_mat_sparse)
from fuzzywuzzy import fuzz





# In[24]:





def fuzzy_matching(mapper, fav_book, verbose=True):

    """

    return the closest match via fuzzy ratio. 

    

    Parameters

    ----------    

    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie

    

    verbose: bool, print log if True

    Return

    ------

    index of the closest match

    """

    match_tuple = []

    # get match

    for title, idx in mapper.items():

        ratio = fuzz.ratio(title.lower(), fav_book.lower())

        if ratio >= 60:

            match_tuple.append((title, idx, ratio))

    # sort

    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]

    if not match_tuple:

        print('Oops! No match is found')

        return

    if verbose:

        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))

    return match_tuple[0][1]
def make_recommendation(model_knn, data, mapper, fav_book, n_recommendations):

    """

    return top n similar book recommendations based on user's input book

    Parameters

    ----------

    model_knn: sklearn model, knn model

    data: book-user matrix

    mapper: dict, map book title name to index of the book in data

    fav_book: str, name of user input book

    n_recommendations: int, top n recommendations

    Return

    ------

    list of top n similar book recommendations

    """

    # fit

    model_knn.fit(data)

    # get input movie index

    print('You have input book:', fav_book)

    idx = fuzzy_matching(mapper, fav_book, verbose=True)

    

    print('Recommendation system starting to make inference')

    print('......\n')

    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)

    

    raw_recommends =         sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

    # get reverse mapper

    reverse_mapper = {v: k for k, v in mapper.items()}

    # print recommendations

    print('Recommendations for {}:'.format(fav_book))

    rec=[]

    for i, (idx, dist) in enumerate(raw_recommends):

        if idx not in reverse_mapper.keys():

            continue

        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))

        rec.append(reverse_mapper[idx])

    return rec
my_favorite = 'To Kill a Mockingbird'

indices = pd.Series(books_col.index, index=books_col['original_title'])
make_recommendation(

    model_knn=model_knn,

    data=book_user_mat_sparse,

    fav_book=my_favorite,

    mapper=indices,

    n_recommendations=10)
make_recommendation(

    model_knn=model_knn,

    data=book_user_mat_sparse,

    fav_book='Harry Potter and the Chamber of Secrets',

    mapper=indices,

    n_recommendations=10)
rec=make_recommendation(

    model_knn=model_knn,

    data=book_user_mat_sparse,

    fav_book='Gone Girl',

    mapper=indices,

    n_recommendations=10)
rec=make_recommendation(

    model_knn=model_knn,

    data=book_user_mat_sparse,

    fav_book='Divergent',

    mapper=indices,

    n_recommendations=10)
rec=make_recommendation(

    model_knn=model_knn,

    data=book_user_mat_sparse,

    fav_book='Kafka on the Shore',

    mapper=indices,

    n_recommendations=10)
rec=make_recommendation(

    model_knn=model_knn,

    data=book_user_mat_sparse,

    fav_book='Kafka on the Shore',

    mapper=indices,

    n_recommendations=10)
netflix=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

netflix.head()
netflix.shape
books['original_title']=books['original_title'].str.lower()

netflix['title']=netflix['title'].str.lower()
t=netflix.merge(books, left_on='title', right_on='original_title', how="inner")
t.shape
import plotly.graph_objects as go



labels = ['Shows from books','Shows not from books']

values = [193,6234]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()