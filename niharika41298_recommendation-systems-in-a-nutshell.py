import plotly.graph_objects as go

import plotly.offline as py

autosize =False





# Use `hole` to create a donut-like pie chart

values=[35, 65]

labels=['Recommended Purchases',"Original Purchases"]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=15,

                  marker=dict(colors=['#00008b','#000'], line=dict(color='#FFFFFF', width=2.5)))

fig.update_layout(

    title='Recommended Purchases VS Original')

py.iplot(fig)
# Use `hole` to create a donut-like pie chart

values=[75, 25]

labels=['Recommended Content',"Originals"]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=15,

                  marker=dict(colors=['#DAA520','#800000'], line=dict(color='#FFFFFF', width=2.5)))

fig.update_layout(

    title='Recommended Views VS Original')

austosize=False

py.iplot(fig)
import numpy as np

from lightfm.data import Dataset

from lightfm import LightFM

from lightfm.evaluation import precision_at_k

from lightfm.evaluation import auc_score

from lightfm.datasets import fetch_movielens

from lightfm.cross_validation import random_train_test_split

from scipy.sparse import coo_matrix as sp
data = fetch_movielens(min_rating = 4.0)
print(repr(data['train']))

print(repr(data['test']))
model = LightFM(loss = 'warp')
model.fit(data["train"], epochs=30, num_threads=2)
def sample_recommendation(model, data, user_ids):

    n_users, n_items = data['train'].shape

    for user_id in user_ids:

        known_positives = data['item_labels'][data['train'].tocsr()                                    

                          [user_id].indices]

        

        scores = model.predict(user_id, np.arange(n_items))



        top_items = data['item_labels'][np.argsort(-scores)]



        print("User %s" % user_id)

        print("     Known positives:")

        

        for x in known_positives[:3]:

            print("        %s" % x)

        

        print("     Recommended:")

        

        for x in top_items[:3]:

            print("        %s" % x)
sample_recommendation(model, data, [6, 25, 451])
import pandas as pd

import numpy as np
books=pd.read_csv("../input/goodbooks-10k-updated/books.csv")
books
import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))

sns.distplot(a=books['average_rating'], kde=True, color='r')
dropna= books.dropna()

fig = px.treemap(dropna, path=['original_publication_year','language_code', "average_rating"],

                  color='average_rating')

fig.show()
fig = px.line(books, y="books_count", x="average_rating", title='Book Count VS Average Rating')

fig.show()
books.columns
books_metadata_selected = books[['book_id', 'average_rating', 

'original_publication_year', 'ratings_count', 'language_code']]

books_metadata_selected
import pandas_profiling



books_metadata_selected.replace('', np.nan, inplace=True)

profile = pandas_profiling.ProfileReport(books_metadata_selected[['average_rating',

                                                                  'original_publication_year', 'ratings_count']])

profile
#rounding the average rating to nearest 0.5 score

books_metadata_selected['average_rating'] = books_metadata_selected['average_rating'].apply(lambda x: round(x*2)/2)



#replacing missing values to the year 

books_metadata_selected['original_publication_year'].replace(np.nan, 2100, inplace=True)
# using pandas qcut method to convert fields into quantile-based discrete intervals

books_metadata_selected['ratings_count'] = pd.qcut(books_metadata_selected['ratings_count'], 25)
profile = pandas_profiling.ProfileReport(books_metadata_selected[['average_rating',

                                                                  'original_publication_year', 'ratings_count']])

profile
#importing ratings data for creating utility matrix

interactions=pd.read_csv("../input/goodbooks-10k-updated/ratings.csv")
from scipy.sparse import *

from scipy import *

item_dict ={}

df = books[['book_id', 'original_title']].sort_values('book_id').reset_index()

for i in range(df.shape[0]):

    item_dict[(df.loc[i,'book_id'])] = df.loc[i,'original_title']

# dummify categorical features

books_metadata_selected_transformed = pd.get_dummies(books_metadata_selected, columns = ['average_rating','original_publication_year', 'ratings_count', 'language_code'])

books_metadata_selected_transformed = books_metadata_selected_transformed.sort_values('book_id').reset_index().drop('index', axis=1)

books_metadata_selected_transformed.head(5)

# convert to csr matrix

books_metadata_csr = csr_matrix(books_metadata_selected_transformed.drop('book_id', axis=1).values)
user_book_interaction = pd.pivot_table(interactions, index='user_id', columns='book_id', values='rating')

# fill missing values with 0

user_book_interaction = user_book_interaction.fillna(0)

user_id = list(user_book_interaction.index)

user_dict = {}

counter = 0 

for i in user_id:

    user_dict[i] = counter

    counter += 1

# convert to csr matrix

user_book_interaction_csr = csr_matrix(user_book_interaction.values)

user_book_interaction_csr
model = LightFM(loss='warp',

                random_state=2016,

                learning_rate=0.90,

                no_components=150,

                user_alpha=0.000005)

model = model.fit(user_book_interaction_csr,

                  epochs=5,

                  num_threads=16, verbose=False)
def sample_recommendation_user(model, interactions, user_id, user_dict, 

                               item_dict,threshold = 0,nrec_items = 5, show = True):

    

    n_users, n_items = interactions.shape

    user_x = user_dict[user_id]

    scores = pd.Series(model.predict(user_x,np.arange(n_items), item_features=books_metadata_csr))

    scores.index = interactions.columns

    scores = list(pd.Series(scores.sort_values(ascending=False).index))

    

    known_items = list(pd.Series(interactions.loc[user_id,:] \

                                 [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))

    

    scores = [x for x in scores if x not in known_items]

    return_score_list = scores[0:nrec_items]

    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))

    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))

    if show == True:

        print ("User: " + str(user_id))

        print("Known Likes:")

        counter = 1

        for i in known_items:

            print(str(counter) + '- ' + str(i))

            counter+=1

    print("\n Recommended Items:")

    counter = 1

    for i in scores:

        print(i)

        print(str(counter) + '- ' + i)

        counter+=1
sample_recommendation_user(model, user_book_interaction, 5, user_dict, item_dict)
sample_recommendation_user(model, user_book_interaction, 500, user_dict, item_dict)