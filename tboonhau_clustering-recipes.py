#import necessary libraries

import pandas as pd

import numpy as np

import json

import re

from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#open dataset

data = []

with open('../input/recipes.json') as f:

    for line in f:

        data.append(json.loads(line))
#read dataset

recipe = data[0]

df = pd.DataFrame(recipe)

df.head()
df.shape
df = df.drop_duplicates(['Name'],keep='first')
df.shape
df = df.drop(['Day','Month','Url','Weekday','Year'],axis=1)



df.head()
df['Ingredients'] = df['Ingredients'].apply(lambda x: ' '.join(x))
df.head()
# Define a function to perform both stemming and tokenization

def tokenizer(text):

    

    # Tokenize by sentence, then by word

    tokens = [word for text in sent_tokenize(text) for word in word_tokenize(text)]

    

       

    # Filter out raw tokens to remove noise

    filtered_tokens = [token for token in tokens if re.match(r'[A-ZÄÖÜ][a-zäöüß]+', token)]

    

    return filtered_tokens
# Instantiate TfidfVectorizer object with stopwords and tokenizer parameters for efficient processing of text



tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer,lowercase=False)
# Fit and transform the tfidf_vectorizer with the "plot" of each movie to create a vector representation of the plot summaries



tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in df['Ingredients']])



print(tfidf_matrix.shape)
pca = PCA(n_components=5)

X = pca.fit_transform(tfidf_matrix.todense())
def find_optimal_clusters(data, max_k):

    iters = range(2, max_k+1, 1)

    

    sse = []

    for k in iters:

        sse.append(KMeans(n_clusters=k, random_state=20).fit(data).inertia_)

        print('Fit {} clusters'.format(k))

        

    f, ax = plt.subplots(1, 1)

    ax.plot(iters, sse, marker='o')

    ax.set_xlabel('Cluster Centers')

    ax.set_xticks(iters)

    ax.set_xticklabels(iters)

    ax.set_ylabel('SSE')

    ax.set_title('SSE by Cluster Center Plot')

    

find_optimal_clusters(tfidf_matrix, 20)
clusters = KMeans(n_clusters=5, random_state=20).fit_predict(tfidf_matrix)
df['clusters'] = clusters

df.head()
#plotting k-means clustering



trace_Kmeans = go.Scatter(x=X[:, 0], y= X[:, 1], mode="markers",

                    showlegend=False,

                    text = df['Name'],

                    hoverinfo = 'text',

                    marker=dict(

                            size=8,

                            color = clusters,

                            colorscale = 'Portland',

                            showscale=False, 

                            line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        )

                   ))

layout = dict(title = 'KMeans Clustering',

              hovermode= 'closest',

              yaxis = dict(zeroline = False),

              xaxis = dict(zeroline = False),

              showlegend= False,

              width=600,

              height=600,

             )



data = [trace_Kmeans]

fig1 = dict(data=data, layout= layout)

# fig1.append_trace(contour_list)

py.iplot(fig1, filename="svm")
tsne = TSNE()

tsne_results = tsne.fit_transform(tfidf_matrix.todense()) 
traceTSNE = go.Scatter(

    x = tsne_results[:,0],

    y = tsne_results[:,1],

#    name = Target,

#     hoveron = Target,

    mode = 'markers',

     text = df['Name'],

    hoverinfo = 'text',

    showlegend = True,

    marker = dict(

        size = 8,

        color = clusters,

        colorscale ='Portland',

        showscale = False,

        line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        ),

        opacity = 0.8

    )

)

data = [traceTSNE]



layout = dict(title = 'TSNE (T-Distributed Stochastic Neighbour Embedding)',

              hovermode= 'closest',

              yaxis = dict(zeroline = False),

              xaxis = dict(zeroline = False),

              showlegend= True,

              autosize=False,

              width=600,

              height=600,

             )



fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
def get_top_keywords(data, clusters, labels, n_terms):

    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    

    for i,r in df.iterrows():

        print('\nCluster {}'.format(i))

        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

            

get_top_keywords(tfidf_matrix, clusters, tfidf_vectorizer.get_feature_names(), 5)