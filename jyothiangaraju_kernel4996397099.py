# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib

matplotlib.use('Agg')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split 

from keras.layers import Input, Embedding, Flatten, Dot, Dense

from keras.models import Model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# Any results you write to the current directory are saved as output.
books = pd.read_csv("../input/goodbooks-10k/books.csv")

ratings = pd.read_csv("../input/goodbooks-10k/ratings.csv")
books.columns
ratings.columns
ratings.describe()
books.describe()
unique_users = ratings.user_id.unique()

unique_books = ratings.book_id.unique()

print(unique_users)

print(unique_books)

n_users=len(unique_users)

n_books=len(unique_books)

print("number of unique users: ",n_users)

print("number of unique books: ",n_books)
from plotly.offline import init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



data = ratings['rating'].value_counts().sort_index(ascending=False)

trace = go.Bar(x = data.index,

               text = ['{:.1f} %'.format(val) for val in (data.values / ratings.shape[0] * 100)],

               textposition = 'auto',

               textfont = dict(color = '#000000'),

               y = data.values,

               )

# Create layout

layout = dict(title = 'Distribution Of {} book-ratings'.format(ratings.shape[0]),

              xaxis = dict(title = 'Rating'),

              yaxis = dict(title = 'Count'))

# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)

combined_data=pd.merge(ratings,books,on="book_id")

combined_data.head()
heavily_rated_books=pd.DataFrame(combined_data.groupby('book_id')['rating'].mean())

heavily_rated_books['total_ratings']=pd.DataFrame(combined_data.groupby('book_id')['rating'].count())

heavily_rated_books.sort_values('total_ratings',ascending=False).head(10)
heavily_rated_books.sort_values('rating',ascending=False).head(10)
book_data={}

for id in unique_books:

    book_data[id]=[0,0]

for x in ratings.index:

    id=ratings['book_id'][x]

    book_data[id][1]+=1

    book_data[id][0]+=ratings['rating'][x]

    

    

    

    
user_deviation={}

rating_dict={}

for id in unique_users:

    user_deviation[id]=[0,0]

    rating_dict[id]=[]

for x in ratings.index:

    user_id=ratings['user_id'][x]

    book_id=ratings['book_id'][x]

    rating=ratings['rating'][x]

    rating_dict[user_id].append(rating)

    user_deviation[user_id][0]+=(abs(rating-(book_data[book_id][0]))/book_data[book_id][1])/book_data[book_id][1]

    user_deviation[user_id][1]+=1

    

    
rdma={}

standard_deviation={}

avg_rdma=0

for user_id in user_deviation.keys():

    val=user_deviation[user_id][0]/user_deviation[user_id][1]

    avg_rdma+=val

    rdma[user_id]=val

    standard_deviation[user_id]=np.std(rating_dict[user_id])

avg_rdma/=n_users

print(avg_rdma)

print(standard_deviation)

    
def f(x):

    p=10*(x-avg_rdma)/(1-avg_rdma)

    y=np.exp(p)

    m=np.exp(10)

    return (y-1)/(m-1)
unique_users
shilling_attackers=[]

f_x=[]

for user_id in unique_users:

    x=rdma[user_id]

    if x>avg_rdma:

        p=f(x)

        f_x.append(p)

        if p>0.5:

           shilling_attackers.append(user_id)
max([user_deviation[x][1] for x in unique_users ])

ratings.shape
for user_id in unique_users:

    if(standard_deviation[user_id]<0.00000001 and user_deviation[user_id][1]>100):

         shilling_attackers.append(user_id)

len(shilling_attackers)
ratings.shape

shilling_attackers
ratings_without_shilling_attackers = ratings

for ind in ratings_without_shilling_attackers.index:

    if(ratings_without_shilling_attackers['user_id'][ind] in shilling_attackers):

        ratings_without_shilling_attackers=ratings_without_shilling_attackers.drop(ind,axis=0)

ratings_without_shilling_attackers.head()        
ratings_without_shilling_attackers.shape

ratings_without_shilling_attackers=ratings_without_shilling_attackers.reset_index()
from keras.layers import Input, Embedding, Flatten, Dot, Dense

from keras.models import Model

book_input = Input(shape=[1], name="Book-Input")

book_embedding = Embedding(n_books+1, 5, name="Book-Embedding")(book_input)

book_vec = Flatten(name="Flatten-Books")(book_embedding)

user_input = Input(shape=[1], name="User-Input")

user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)

user_vec = Flatten(name="Flatten-Users")(user_embedding)

prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec])

model_with_shilling_attackers = Model([user_input, book_input], prod)

model_with_shilling_attackers.compile('adam', 'mean_squared_error')

book_input1 = Input(shape=[1], name="Book-Input1")

book_embedding1 = Embedding(n_books+1, 5, name="Book-Embedding1")(book_input1)

book_vec1 = Flatten(name="Flatten-Books1")(book_embedding1)

user_input1 = Input(shape=[1], name="User-Input1")

user_embedding1 = Embedding(n_users+1, 5, name="User-Embedding")(user_input1)

user_vec1 = Flatten(name="Flatten-Users")(user_embedding1)

prod1 = Dot(name="Dot-Product", axes=1)([book_vec1, user_vec1])

model_without_shilling_attackers = Model([user_input1, book_input1], prod1)

model_without_shilling_attackers.compile('adam', 'mean_squared_error')

model_with_shilling_attackers.fit([ratings.user_id, ratings.book_id], ratings.rating, epochs=10, verbose=1)

model_with_shilling_attackers.save('regression_model.h5')
model_without_shilling_attackers.fit([ratings_without_shilling_attackers.user_id, ratings_without_shilling_attackers.book_id], ratings_without_shilling_attackers.rating, epochs=10, verbose=1)

model_without_shilling_attackers.save('regression_model.h5')
def recommendations(user_id):

    book_data1 = np.array(list(set(ratings.book_id)))

    user = np.array([user_id for i in range(len(book_data1))])

    if(user_id in shilling_attackers):

        predictions = model_with_shilling_attackers.predict([user, book_data1])

    else:

        predictions = model_without_shilling_attackers.predict([user, book_data1])

    predictions = np.array([a[0] for a in predictions])

    recommended_book_ids = (-predictions).argsort()[:10]

    return recommended_book_ids

predictions_array=[]

for user_id in unique_users[:20]:

    predictions_array.append(list(recommendations(user_id)))

predictions_array    
bookembeddings = model_with_shilling_attackers.get_layer('Book-Embedding')

bookembeddings_weights = bookembeddings.get_weights()[0]
from sklearn.decomposition import PCA

import seaborn as sns

pca = PCA(n_components=2)

pca_transformed_result = pca.fit_transform(bookembeddings_weights)

sns.scatterplot(x=pca_transformed_result[:,0], y=pca_transformed_result[:,1])