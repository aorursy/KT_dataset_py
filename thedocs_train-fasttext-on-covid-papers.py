# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import fasttext

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
from pathlib import Path

# metadata

data_dir = Path('/kaggle/input/')



dtypes = {'title': str, 'abstract': str, ' text': str}

df1 = pd.read_csv(data_dir/'cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv', dtype=dtypes)

df2 = pd.read_csv(data_dir/'cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv',dtype=dtypes)

df3 = pd.read_csv(data_dir/'cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv', dtype=dtypes)

df4 = pd.read_csv(data_dir/'cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv', dtype=dtypes)



import gc 



all_data = pd.concat([df1, df2, df3, df4])

del df1, df2, df3, df4

gc.collect()

all_data.tail()



%%time

from collections import Counter    

from nltk.corpus import stopwords



# remove special characters 

def remove_special(x):

    # new line at sentence end for training purposes

    x = x.replace('.', '\n')

    return re.sub('[^A-Za-z \n]+', ' ', x).lower()

    

stop_words = stopwords.words('english')

stopwords_dict = Counter(stop_words)

def remove_stopwords(x):

    return ' '.join([word for word in x.split() if word not in stopwords_dict])

    

title = all_data['title'][~all_data['title'].isna()].apply(remove_stopwords).apply(remove_special).values

text_body = all_data['text'][~all_data['text'].isna()].apply(remove_stopwords).apply(remove_special).values

abstract = all_data['abstract'][~all_data['abstract'].isna()].apply(remove_stopwords).apply(remove_special).values



corpus = np.concatenate([title, text_body, abstract], axis=0)



print('Number of rows in corpus list: ', len(corpus))



# convert to string for model training

corpus = "\n".join(corpus)
%%time



train_file = 'covid_corpus.txt'



with open(train_file, 'w') as f:

    f.write(corpus)

    

"""

Uncomment below to train. Training can take a 2-8 hours based on number of epochs

"""

# model_train = fasttext.train_unsupervised(input=newpath, epoch=5, model="skipgram")

# model_train.save_model("model_1.bin")



# using previously trained model 

model = fasttext.load_model(str(data_dir/'trainedfasttext/model_1.bin'))

"""

Utility visualization functions 

"""



def get_words_from_indices(model, indices: list):

    """

    Gets words from model vocab given the indices

    """

    labels = np.array(model.get_labels())

    

    return labels[indices]



def get_embedding_from_indices(model, indices: list):

    """

    Gets word embeddings of model given the indices

    """



    return model.get_output_matrix()[indices, :]

  



def find_kneighbours(embeddings_matrix, vector, n_neighbors=20):

    """

    Finds k nearest neighbors of vector (1xN) in the (MxN) matrix. 

    """

    from sklearn.neighbors import NearestNeighbors



    if len(vector.shape) == 1:

        vector = vector.reshape(-1,1).T



    assert vector.shape[1] == embeddings_matrix.shape[1]



    nn = NearestNeighbors(n_neighbors=n_neighbors)

    nn.fit(embeddings_matrix)

    indices = nn.kneighbors(vector, return_distance=False)

    

    return indices.squeeze(0)



def get_tnse_embeddings(output_matrix):

    from sklearn.manifold import TSNE 



    tnse = TSNE(n_components=3, perplexity=30.0, early_exaggeration=12.0, 

                            learning_rate=200.0, n_iter=500, n_iter_without_progress=100, 

                            min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, 

                            random_state=None, 

                            method='barnes_hut', angle=0.5)

    embedding_matrix = tnse.fit_transform(output_matrix)

    return embedding_matrix



def plot_tnse_embeddings(embedding_matrix, labels, color=None):

    """

    input: (M x 3) and labels

    output: plotly figure 

    """

    import plotly.io as pio

    import plotly.graph_objs as go

    import math

    plot_mode = 'text+markers'

    if not color:

        color = "#3266c1"



    if embedding_matrix.shape[1] == 3:

        scatter = go.Scatter3d(

            name='None',

            x=embedding_matrix[:,0],

            y=embedding_matrix[:,1],

            z=embedding_matrix[:,2],

            text=labels,

            textposition="top center",

            showlegend=False,

            mode=plot_mode,

            marker=dict(size=5, color=color, symbol="circle"),

        )

        figure = go.Figure(data=[scatter])

        return figure

    else:

        raise ValueError('Unsupported dimensions: must be 3, not {}'.format(embedding_matrix.shape[1])) 

        

    

def plot_tnse_aroud_word(model, word: str, n_neighbors=20):



    """

    creates visualization of tnse showing the input word. 

    """

    if len(word.strip().split()) == 1:

        vector = model.get_word_vector(word)

    else:

        vector = model.get_sentence_vector(word)



    indices = find_kneighbours(model.get_output_matrix(), vector, n_neighbors=n_neighbors)

    embedding_matrix = get_embedding_from_indices(model, indices)

    print('Size of embedding matrix:', embedding_matrix.shape)

    labels = get_words_from_indices(model, indices)

    embeddings_with_target = np.concatenate((vector.reshape(-1,1).T, embedding_matrix),axis=0)

    labels_with_target = [word] + list(labels) # appending to first as embedding also appened to first

    tnse_embeddings_with_target = get_tnse_embeddings(embeddings_with_target)

    

    colors = np.ones(tnse_embeddings_with_target.shape[0])

    colors[0] = 5



    return plot_tnse_embeddings(tnse_embeddings_with_target, labels_with_target, color=colors.tolist())



word= 'virus'

plot_tnse_aroud_word(model, word, n_neighbors=20)              