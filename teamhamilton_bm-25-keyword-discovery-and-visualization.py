#Import the BM25 index

!pip install rank_bm25

import pickle

import os

!git clone https://github.com/CoronaWhy/CORD-19-QA 

os.chdir('CORD-19-QA')

!wget -O bert_bioasq_final-bm25.pkl https://publicweightsdata.s3.us-east-2.amazonaws.com/bert_bioasq_final-bm25.pkl

from bm25_index import BM25Index

os.chdir('/kaggle/working')



#Import Transformers 

!pip install transformers

!wget -O scibert_uncased.tar https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar

!tar -xvf scibert_uncased.tar

import torch

from transformers import BertTokenizer, BertModel



#Import key libraries 

import numpy as np 

import pandas as pd

import plotly.graph_objects as go

import networkx as nx

import itertools

from random import randrange

import math

import re

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize,word_tokenize

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
#Load metadata.csv (from https://www.kaggle.com/ajrwhite/covid-19-thematic-tagging-with-regular-expressions/notebook)

def load_metadata(metadata_file):

    df = pd.read_csv(metadata_file,

                 dtype={'Microsoft Academic Paper ID': str,

                        'pubmed_id': str},

                 low_memory=False)

    df.doi = df.doi.fillna('').apply(doi_url)

    #df['authors_short'] = df.authors.apply(shorten_authors)

    #df['sorting_date'] = pd.to_datetime(df.publish_time)

    print(f'loaded DataFrame with {len(df)} records')

    #return df.sort_values('sorting_date', ascending=False)

    return df



# Fix DOI links

def doi_url(d):

    if d.startswith('http'):

        return d

    elif d.startswith('doi.org'):

        return f'http://{d}'

    else:

        return f'http://doi.org/{d}'



metadata = load_metadata('../input/CORD-19-research-challenge/metadata.csv')
#Search results for a query

the_index = pickle.load(open('CORD-19-QA/bert_bioasq_final-bm25.pkl', 'rb'))

the_search_terms = 'coronavirus bat to human transmission'

abstracts = the_index.search(the_search_terms, 10)

dummy_search_results = pd.merge(abstracts, metadata, left_on = ['paper_id'], right_on = ['sha'], how = 'inner')
##Preprocess the search result abstracts

covid_text = dummy_search_results['abstract_y']



## remove special characters from string and set to lowercase

def cleanString(text):

    regex = r'[^a-zA-Z0-9\s]'

    text = re.sub(regex,'',text)

    text = text.rstrip("\n")

    return text.lower()



## remove special characters from 'text' string

for i in range(0, len(dummy_search_results)):

    covid_text.at[i] = cleanString(covid_text[i])

        

## Remove stopwords and words of length 1

stop_words = set(stopwords.words('english'))

for i in range(0, len(covid_text)):

    rowbody = ''

    for word in word_tokenize(covid_text[i]):

        if word not in stop_words and len(word)>1:

            rowbody = rowbody + word + ' '

    covid_text.at[i] = rowbody

    

## Tokenize words

for i in range(0, len(covid_text)):

    rowbody = ''

    for word in word_tokenize(covid_text[i]):

        rowbody = rowbody + word + ' '

    covid_text.at[i] = rowbody    



## Calculate TF-IDF

tfidf_vectorizer=TfidfVectorizer(smooth_idf=True, use_idf=True)

tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(covid_text.to_list())



## Calculate TF-IDF score for every word in a given document

def getTFIDF_byindex(dataset, docnum):

    one_vector=tfidf_vectorizer_vectors[int(docnum)]

    # Create df, where rows are each word and column is tfidf score

    df = pd.DataFrame(one_vector.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

    # Add cord_uid document unique identifier column to df

    df['cord_uid'] = dataset.loc[int(docnum),'cord_uid']

    # Return dataframe sorted by tfidf score

    return (df.sort_values(by='tfidf', ascending=False))



## Calculate TF-IDF for each document in the search results

allres = []

for i in range(0, len(dummy_search_results)):

    corduid = dummy_search_results.loc[i,'cord_uid']

    allres.append(getTFIDF_byindex(dummy_search_results, i))

    

## display dictionary of resulting tf-idf dataframes (one dataframe per document)

allres = pd.concat(allres, sort=False)
G = nx.Graph()

for combo in itertools.combinations(allres['cord_uid'].unique(), 2):

    A = set(allres[(allres['cord_uid']==combo[0]) & (allres['tfidf']>0.1)].index).difference(the_search_terms.split())

    B = set(allres[(allres['cord_uid']==combo[1]) & (allres['tfidf']>0.1)].index).difference(the_search_terms.split())        

    C = A.intersection(B)

    if len(C)>0:

        G.add_edge(combo[0], combo[1], weight = len(C), papers = list(C))   
#Using plotly, draw a chord diagram. Nodes are papers; edges and edge thickness indicates the presence and frequency of shared keywords.   



#Machinery to define an edge as a Bézier curve



#Computes the distance between two 2D points: A, B

def dist (A,B):

    return np.linalg.norm(np.array(A)-np.array(B))



#Threshold distances between nodes on the unit circle and the list of parameters for interior control points of the Bézier curve

Dist=[0, dist([1,0], 2*[np.sqrt(2)/2]), np.sqrt(2), dist([1,0],  [-np.sqrt(2)/2, np.sqrt(2)/2]), 2.0]

params=[1.2, 1.5, 1.8, 2.1]



#Returns the index of the interval the distance d belongs to

def get_idx_interv(d, D):

    k=0

    while(round(d,2)>D[k]):

        k+=1

    return  k-1



#The default number of points evaluated on a Bézier edge is 5. Then setting the Plotly shape of the edge line as spline, the five points are interpolated.

class InvalidInputError(Exception):

    pass

#Returns the point corresponding to the parameter t, on a Bézier curve of control points given in the list b.

def deCasteljau(b,t):

    N=len(b)

    if(N<2):

        raise InvalidInputError("The  control polygon must have at least two points")

    a=np.copy(b) #shallow copy of the list of control points 

    for r in range(1,N):

        a[:N-r,:]=(1-t)*a[:N-r,:]+t*a[1:N-r+1,:]

    return a[0,:]



#Returns an array of shape (nr, 2) containing the coordinates of nr points evaluated on the Bézier curve, at equally spaced parameters in [0,1]

def BezierCv(b, nr=5):

    t=np.linspace(0, 1, nr)

    return np.array([deCasteljau(b, t[k]) for k in range(nr)])





#Create Edges

#Add edges as disconnected lines in individual traces and nodes as a scatter trace, along with hidden edge traces in the middle of each edge for interactivity.

pos=nx.circular_layout(G)  #node locations along the unit circle

    

edge_traces=[]

hidden_edge_traces=[]

edge_colors=['#d4daff','#84a9dd', '#5588c8', '#6d8acf']



for edge in G.edges():

    edge_x = []

    edge_y = []

    hidden_x = []

    hidden_y = []

    x0, y0 = pos[edge[0]] #xy-coordinates of the first node on the edge 

    x1, y1 = pos[edge[1]] #xy-coordinates of the second node on the edge

    d = dist([x0,y0], [x1,y1])

    K=get_idx_interv(d, Dist)

    b=[pos[edge[0]], pos[edge[0]]/params[K], pos[edge[1]]/params[K], pos[edge[1]]]

    color=edge_colors[K]

    pts=BezierCv(b, nr=5)

    mark=deCasteljau(b,0.9)

    edge_x.append(x0)

    edge_x.append(x1)

    edge_y.append(y0)

    edge_y.append(y1)

    hidden_x.append(mark[0])

    hidden_y.append(mark[1])

    

    edge_traces.append(go.Scatter(

        x=pts[:,0], 

        y=pts[:,1],

        mode = 'lines',

        line=dict(color=color, shape='spline', width=G.edges[edge]['weight']), 

        hoverinfo='none'))

    

    hidden_edge_traces.append(go.Scatter(

        x=hidden_x, 

        y=hidden_y, 

        text=' '.join(G.edges[edge]['papers']), 

        mode='markers', 

        hoverinfo='text', 

        marker=go.scatter.Marker(opacity=0)))



node_x = []

node_y = []

for node in G.nodes():

    x, y = pos[node]

    node_x.append(x)

    node_y.append(y)



node_trace = go.Scatter(

    x=node_x, 

    y=node_y,

    hoverinfo = 'text',

    mode='markers',

    text = [metadata.at[metadata[metadata['cord_uid']==node].index[0], 'title'] for node in G.nodes()])
#Create Network Graph



fig = go.Figure(data = [*edge_traces, *hidden_edge_traces, node_trace],

             layout=go.Layout(

                title='<br>Animal host evidence of spill-over to humans',

                titlefont_size=16,

                title_x = 0.5,

                showlegend=False,

                hovermode='closest',

                margin=dict(b=20,l=5,r=5,t=40),

                annotations=[ dict(

                    text='Search Terms: ' + the_search_terms,

                    showarrow=False,

                    xref="paper", yref="paper",

                    x=0.005, y=-0.002 ) ], 

                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),

                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

                )

fig.show()