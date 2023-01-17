!pip install rank_bm25

!pip install range_key_dict



import os

import re

import time

import pandas as pd

import numpy as np

import gensim

from gensim.summarization.summarizer import summarize

from gensim.utils import simple_preprocess

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial.distance import cosine

from sklearn.feature_extraction.text import TfidfVectorizer

from spacy import displacy

import ipywidgets as widgets

from ipywidgets import Layout

from IPython.display import display, clear_output

from IPython.core.display import display, HTML, Image

from rank_bm25 import BM25Okapi

from rank_bm25 import BM25L

import nltk

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')

nltk.download('averaged_perceptron_tagger')



from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

from nltk.util import ngrams

from nltk.tokenize import word_tokenize, sent_tokenize



from nltk.stem.porter import PorterStemmer

from sklearn.manifold import TSNE





from range_key_dict import RangeKeyDict



from rank_bm25 import BM25Okapi

import ast

import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
# Dataset creation credits : https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv



biorxiv_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv')

clean_comm_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv')

clean_noncomm_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv')

clean_pmc_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv')



# Concat all data files into a single dataframe.

dfs = [biorxiv_data, clean_comm_data, clean_noncomm_data, clean_pmc_data]

data = pd.concat(dfs, ignore_index=True)

# Drop duplicate papers

data.drop_duplicates(subset=["title"],inplace=True)





def preprocess(data):



    def preprocess_(x):

        # Lowercase

        x = x.lower()

        # Remove urls

        x = re.sub(r'https?://\S+|www\.\S+', '', x)

        #x = re.sub(r'\[(\d+)(\,+)\]', '', x).strip()

        x = re.sub(r'\[\d{0,2}\s*(,\d{0,2})*\]*', '', x).strip()

        return x



    data = data.copy()



    # Filling the empty abstracts 

    data['abstract'] = data['abstract'].fillna('')

    # Filling the empty titles 

    data['title'] = data['title'].fillna('')



    # Preprocess

    data['abstract_processed'] = data['abstract'].apply(preprocess_)

    data['title_processed'] = data['title'].apply(preprocess_)

    data['text_processed'] = data['text'].apply(preprocess_)

    data['combined'] = data['title_processed'] + '. ' + data['abstract_processed']



    return data



# Preprocess data

data = preprocess(data)
tokens = data['combined'].apply(nltk.word_tokenize).values

model = gensim.models.Word2Vec(

        tokens,

        size=200,

        window=5,

        min_count=30,

        workers=20,

        iter=10)



def query_synonyms(query_term, embedding_model, top_n=5):

    words = embedding_model.wv.most_similar(query_term, topn=top_n)

    words = [word[0] for word in words]

    return words

    

def plot_tsne(word, model):



    if word == "":

        return

    words = model.wv.most_similar(word, topn=20)

    words = [x[0] for x in words] + [word]

    X = model[words]

    

    tsne = TSNE(n_components=2)

    X_tsne = tsne.fit_transform(X)



    df = pd.DataFrame(X_tsne, index=words, columns=['x', 'y'])

    

    plt.figure(figsize=(15, 10))

    plt.scatter(df.x, df.y)

    for w, pos in df.iterrows():

        if w == word:

            plt.annotate(w, pos, size=40)

        else:

            plt.annotate(w, pos, size=20)
plot_tsne('coronavirus', model)
def process_query(query, expand=False):



    query = query.lower()

    query_list = word_tokenize(query)

    

    if expand == True:

        for word in query_list.copy():

            synonyms = query_synonyms(word, model, 3)

            query_list.extend(synonyms)

    return query_list



def filter_top_docs(data, scores, top_n=300):



    retrieved = data.copy()

    retrieved['score'] = scores

    retrieved = retrieved[retrieved['score'] > 0]

    retrieved.sort_values('score', ascending=False, inplace=True)

    retrieved = retrieved.iloc[0:top_n]

    return retrieved





documents = data['combined'].values

tokenized_corpus = [word_tokenize(doc) for doc in documents]

bm25 = BM25Okapi(tokenized_corpus) 



query = "covid-19 coronavirus sars-cov-2 2019-ncov"

query = process_query(query, expand=False)

scores = bm25.get_scores(query)

retrieved = filter_top_docs(data, scores, top_n=300)
for idx, row in retrieved.iloc[0:20].iterrows():

    try:

        summary = summarize(row.abstract_processed, ratio=0.2)

        display(HTML("<p><font color='#3B57BD'>" + summary + "</font></p>"))

    except:

        continue

    display(HTML("<br>"))
def search(query):

    query = query.lower().strip()

    similarity = get_similarity(vectorizer, X, query)

    display_results(query, sentences, similarity, index_dictionary, top_n=20)



    

def get_tfidf_matrix():

    range_idx_dict = {}



    # Extract sentences from all papers

    sentences = []

    for i in range(len(retrieved)):

        doc_sents = sent_tokenize(retrieved.text_processed.iloc[i])

        

        idx = retrieved.index.values.astype(int)[i]

        full_len = len(sentences)

        new_len = len(doc_sents)

        range_idx_dict[(full_len, full_len+new_len)] = idx



        sentences.extend(doc_sents)

    

    # Match any period that's not followed by a digit, this removes periods at the end of sentences

    sentences = [re.sub('\.(?!\d)', ' ', sentence) for sentence in sentences]



    index_dict = RangeKeyDict(range_idx_dict)



    # Vectorize the sentences

    vectorizer = TfidfVectorizer(stop_words="english")

    X = vectorizer.fit_transform(sentences)

    return sentences, vectorizer, X, index_dict





def get_similarity(vectorizer, X, query):

    query = query.lower()

    query = vectorizer.transform([query])

    return cosine_similarity(X, query)





def display_results(query, sentences, similarity, index_dictionary, top_n=20):

    query = query.split(' ')

    df = pd.DataFrame({

        "sentence": sentences,

        "similarity": similarity.flatten(),

    }).sort_values('similarity', ascending=False)



    df = df[df.similarity >= 0.1]



    for idx, row in df.iloc[0:top_n].iterrows():



        title = str(retrieved.loc[index_dictionary[idx]].title)

        authors = str(retrieved.loc[index_dictionary[idx]].authors).split(',')[0] + '. et al.'

        

        prev_sentence = ""

        next_sentence = ""

        '''

        if idx != 0:

            prev_sentence = str(df.loc[idx - 1].sentence)



        if idx != len(df):

            next_sentence = str(df.loc[idx + 1].sentence)

        '''



        

        # Display title of the paper and the authors

        display(HTML("<p><b><font color='black'>" + str(title) + '. ' +

                    "</b>"+str(authors) + "</font></p>"))

        

        sentence = prev_sentence + row.sentence + next_sentence

        sentence = sentence.split(' ')

        

        paste = re.sub("(\'|\[|\]|\,|\")", "", 

                       str(['<span style="background-color:#9ae59a">' + x + "</span>" if x in query 

                            else str(x) for x in sentence]))

        

        # Display the sentence

        display(HTML("<p><font color='#3B57BD'>" + paste + "</font></p>".strip("\'")))

        display(HTML("<hr>"))





def create_search():



    #sentences, vectorizer, X, index_dictionary = get_tfidf_matrix()



    text = widgets.Text(

        placeholder='Example: incubation period',

        disabled=False,

        layout=Layout(align_items='stretch', width='90%', border='solid')

    )

    display(HTML("<h3>Please Enter a Search Query</h3>"))

    display(text)

    output = widgets.Output()



    def callback(wdgt):

        clear_output()

        display(HTML("<h3>Please Enter a Search Query</h3>"))

        display(text)

        output = widgets.Output()



        # Get the entered text

        query = wdgt.value

        query = query.lower().strip()

        similarity = get_similarity(vectorizer, X, query)

        display_results(query, sentences, similarity, index_dictionary, top_n=20)



    text.on_submit(callback)

    

    



sentences, vectorizer, X, index_dictionary = get_tfidf_matrix()
# Search using this function

search('incubation period')
# Or use this function to create the ipywidgets search box

create_search()
