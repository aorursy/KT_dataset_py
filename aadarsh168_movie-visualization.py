import re

import nltk

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

from ast import literal_eval

import matplotlib.pyplot as plt

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



%matplotlib inline

warnings.simplefilter('ignore')

pd.set_option('display.max_columns', 50)
credits = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')

movies = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits.head()
from ast import literal_eval



#Converting the string into list of dictionaries

credits.cast = credits.cast.apply(literal_eval)

credits.crew = credits.crew.apply(literal_eval)



# Extracting the Casts into a list from Dictionaries

credits['cast'] = credits['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])



# Extracting the Director from the Crew

def extract_director(x):

    for crew_mem in x:

        if crew_mem['job'] == 'Director':

            return crew_mem['name']

        else:

            return np.nan



credits['director'] = credits['crew'].apply(extract_director)

credits['director'].fillna('',inplace = True)

credits.drop(['crew'],axis = 1,inplace = True)

credits.drop(['title'],axis = 1,inplace = True)
movies.head()
# Extracting the Genres into a list from Dictionaris

movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# Extracting the Keywords into a list from Dictionaris

movies['keywords'] = movies['keywords'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movies = movies.merge(credits, left_on='id', right_on='movie_id', how = 'left')
# Selecting required columns from the master dataframe

movies = movies[['id','original_title','title','cast', 'director', 'keywords', 'genres', 'release_date', 'overview', 

                 'original_language', 'runtime', 'tagline', 'vote_average', 'vote_count','popularity']]

movies.head()
movies.isna().sum()
movies.tagline.fillna('',inplace = True)

movies = movies.dropna().reset_index()
movies.release_date = pd.to_datetime(movies.release_date,format = '%Y-%m-%d')

movies['release_year'] = movies.release_date.apply(lambda x: x.year)
def get_wordnet_pos(word):

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)



lemmatizer=WordNetLemmatizer()



def clean_plot(txt):

    regex = re.compile(r"[!@%&;?'',.""-]")

    txt_clean = re.sub(regex,'',txt)

    txt_clean = txt_clean.lower()

    txt_clean = txt_clean.split(' ')

    txt_clean = [word for word in txt_clean if word not in stopwords.words('english')]

    txt_clean = ' '.join(txt_clean)

    word_list = nltk.word_tokenize(txt_clean)

    txt_clean = ' '.join([lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in word_list])

    return txt_clean

movies.head()
genres = movies['genres'].apply(lambda x : " ".join(x))

keywords = movies['keywords'].apply(lambda x : " ".join(x))
overview = movies.overview.apply(clean_plot)

genres = genres.apply(clean_plot)

keywords = keywords.apply(clean_plot)
release_year = movies.release_year
genre_keys = genres + ' ' + keywords
tfidf = TfidfVectorizer(analyzer = 'word', ngram_range = (1,1), min_df = 0, stop_words = 'english')

plot_vector = tfidf.fit_transform(overview)
cv = CountVectorizer(analyzer = 'word', ngram_range = (1,1), min_df = 0, stop_words = 'english')

genrekey_vector = cv.fit_transform(genre_keys)
from scipy.spatial import distance

import time

score = []

start_time = time.time()



plot_arr = plot_vector.toarray()

genrekey_arr = genrekey_vector.toarray()



def get_pos(title):

    target_plot_arr = plot_vector[movies[movies.title==title].index.values[0]].toarray()

    target_genre_arr = genrekey_vector[movies[movies.title==title].index.values[0]].toarray()

    pos = {}

    for i in range(plot_arr.shape[0]):

        plot_pos = distance.euclidean(target_plot_arr,plot_arr[i])

        genre_pos = distance.euclidean(target_genre_arr,genrekey_arr[i])

        pos[movies.title[i]] = [release_year[i],plot_pos,genre_pos]

    return pos



print("--- %s seconds ---" % (time.time() - start_time))
import plotly.graph_objects as go

import plotly.express as px

import networkx as nx
def create_graph(title):

    G = nx.Graph()

    G.clear()

    G.add_nodes_from(movies.title.tolist())

    pos = get_pos(title)

    for node in G.nodes:

        G.nodes[node]['pos'] = pos[node]

    return G
def plot_graph(G):

    edge_x = []

    edge_y = []

    for edge in G.edges():

        x0, y0 = G.nodes[edge[0]]['pos']

        x1, y1 = G.nodes[edge[1]]['pos']

        edge_x.append(x0)

        edge_x.append(x1)

        edge_x.append(None)

        edge_y.append(y0)

        edge_y.append(y1)

        edge_y.append(None)



    edge_trace = go.Scatter(

        x=edge_x, y=edge_y,

        line=dict(width=0.5, color='#888'),

        hoverinfo='none',

        mode='lines')



    node_x = []

    node_y = []

    node_z = []

    for node in G.nodes():

        x, y, z = G.nodes[node]['pos']

        node_x.append(x)

        node_y.append(y)

        node_z.append(z)



    node_trace = go.Scatter3d(

        x=node_x, y=node_y,z=node_z,

        mode='markers',

        hoverinfo='text',

        marker=dict(

            # colorscale options

            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |

            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |

            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |

            colorscale='YlGnBu',

            reversescale=True,

            color=z,

            size=10,

            opacity=0.8,

            colorbar=dict(

                thickness=15,

                title='Similarty',

                xanchor='left',

                titleside='right'

            ),

            line_width=2))

    node_adjacencies = []

    node_text = []

    for index,row in movies[['title','genres']].iterrows():

        gen = " | ".join(row['genres'])

        text = 'Title: ' + row['title'] + '\nGenres:' + gen

        node_text.append(text)

#     for node, adjacencies in enumerate(G.adjacency()):

#         node_adjacencies.append(len(adjacencies[1]))

#         node_text.append('# of connections: '+str(len(adjacencies[1])))

#     zipped_nodes = zip(node_x,node_y,node_z)

#     color_node = [round(x + y + z) for (x, y,z) in zipped_nodes]

    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],

             layout=go.Layout(

                title='<br>Movie Similarity Graph with Python',

                titlefont_size=16,

                showlegend=False,

                hovermode='closest',

                margin=dict(b=20,l=5,r=5,t=40),

                annotations=[ dict(

                    text="Python code: <a href='https://www.kaggle.com/aadarsh168/movie-visualization/'> Kaggle Notebook</a>",

                    showarrow=False,

                    xref="paper", yref="paper",

                    x=0.005, y=-0.002 ) ],

                xaxis=dict(showgrid=False, zeroline=True, showticklabels=True),

                yaxis=dict(showgrid=False, zeroline=True, showticklabels=True))

                )

    fig.update_layout(

    scene = dict(

        xaxis = dict(nticks=4, range=[1915,2018],),

        yaxis = dict(nticks=4, range=[-1,10],),

        zaxis = dict(nticks=4, range=[-1,10],),),

        width=700,

        margin=dict(r=20, l=10, b=10, t=10))

    

    fig.update_layout(

        scene = dict(

            xaxis_title='Release Year',

            yaxis_title='Plot Distance',

            zaxis_title='Genres Distance'),

            width=700,

            margin=dict(r=20, b=10, l=10, t=10))

    



#     fig.update_layout(

#         xaxis = dict(

#             tickangle = 90,

#             title_text = "Genre Distance",

#             title_font = {"size": 20},

#             title_standoff = 25),

#         yaxis = dict(

#             title_text = "Plot Distance",

#             title_font = {"size": 20},

#             title_standoff = 25))

    return fig

fig = plot_graph(create_graph('Toy Story'))

fig.show()