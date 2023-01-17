#Data Processing
import tensorflow as tf
import numpy as np
import unicodedata
import re
import os
from gensim.utils import simple_preprocess
from gensim.models import FastText
from gensim.models import TfidfModel
from sklearn.neighbors import NearestNeighbors
import pandas as pd

MODE = 'train'
NUMBER_OF_DATASET = 60000


def read_dataset(number):

    twi_data = []
    with open('../input/data-gen-dataset/jw300.en-tw.tw') as file:

        # twi=file.read()
        line = file.readline()
        cnt = 1
        while line:
            twi_data.append(line.strip())
            line = file.readline()
            cnt += 1

    return twi_data[:number]

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalize_twi(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.ƆɔɛƐ!?’]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s


raw_data_twi = read_dataset(NUMBER_OF_DATASET)
raw_data_twi = [normalize_twi(data) for data in raw_data_twi]
data=[]
data_raw = []
for tw in raw_data_twi:
    data_raw.append(tw)
    data.append(tw.split())
    
raw_data_twi = []
print(data_raw[:3])
print(data[:3])

n = 50
ft_model = FastText(data, size=n, window=8, min_count=5, workers=2,sg=1)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data_raw)
print(X.shape)
#To proprely work with scikit's vectorizer
merged_data = [' '.join(question) for question in data]
document_names = ['Doc {:d}'.format(i) for i in range(len(merged_data))]
# print(document_names)

def get_tfidf(docs, ngram_range=(1,1), index=None):
    vect = TfidfVectorizer( ngram_range=ngram_range)
    tfidf = vect.fit_transform(docs).todense()
    return pd.DataFrame(tfidf, columns=vect.get_feature_names(), index=index).T

tfidf = get_tfidf(merged_data, ngram_range=(1,1), index=document_names)
def get_sent_embs(emb_model):
    sent_embs = []
    for desc in range(len(data)):
        sent_emb = np.zeros((1, n))
        if len(data[desc]) > 0:
            sent_emb = np.zeros((1, n))
            div = 0
            model = emb_model
            for word in data[desc]:
                if word in model.wv.vocab and word in tfidf.index:
                    word_emb = model.wv[word]
                    weight = tfidf.loc[word, 'Doc {:d}'.format(desc)]
                    sent_emb = np.add(sent_emb, word_emb * weight)
                    div += weight
                else:
                    div += 1e-13 #to avoid dividing by 0
        if div == 0:
            print(desc)

        sent_emb = np.divide(sent_emb, div)
        sent_embs.append(sent_emb.flatten())
    return sent_embs
ft_sent = get_sent_embs(emb_model = ft_model) 
def get_n_most_similar(interest_index, embeddings, n):
    """
    Takes the embedding vector of interest, the list with all embeddings, and the number of similar questions to 
    retrieve.
    Outputs the disctionary IDs and distances
    """
    nbrs = NearestNeighbors(n_neighbors=n, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    similar_indices = indices[interest_index][1:]
    similar_distances = distances[interest_index][1:]
    return similar_indices, similar_distances

def get_sen_most_similar(sent, embeddings, n):
    """
    Takes the embedding vector of interest, the list with all embeddings, and the number of similar questions to 
    retrieve.
    Outputs the disctionary IDs and distances
    """
   
    
    nbrs = NearestNeighbors(n_neighbors=n, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    similar_indices = indices[interest_index][1:]
    similar_distances = distances[interest_index][1:]
    return similar_indices, similar_distances

def print_similar(interest_index, embeddings, n):
    """
    Convenience function for visual analysis
    """
    closest_ind, closest_dist = get_n_most_similar(interest_index, embeddings, n)
    print("Twi Sentence ===> ",data_raw[interest_index])
    
    print('\n')
    print("Similiar Sentences")
    print("--------------------------------")
    print('\n')
    for question in closest_ind:
        print(data_raw[question])
        print("--------------------------------")

def print_similar_sen(sent, embeddings, n):
    """
    Convenience function for visual analysis
    """
    interest_index = data_raw.index(sent)
        
    closest_ind, closest_dist = get_n_most_similar(interest_index, embeddings, n)
    print("Twi Sentence ===> ",data_raw[interest_index])
    
    print('\n')
    print("Similiar Sentences")
    print("--------------------------------")
    print('\n')
    for question in closest_ind:
        print(data_raw[question])
        print("--------------------------------")
print_similar(200, ft_sent, 5)
sent = 'Yefii yɛn som adwuma ase sɛ akwampaefo atitiriw wɔ Beppu wɔ Kyushu Supɔw so wɔ May mu .'
print_similar_sen(sent, ft_sent, 5)
