import numpy as np

import pandas as pd 

from tqdm import tqdm

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer

import string

from sklearn.cluster import KMeans, MiniBatchKMeans

from gensim import corpora, models, similarities 

from itertools import chain

from nltk.corpus import stopwords

from operator import itemgetter

import re

import os

import json



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        #print(path)


# paths = ["/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset",\

#          '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv', \

#          '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset', \

#          '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license'

#         ]



# def get_files(json_folder_path):

#     all_filenames, unprocessed_files = [],[]

#     json_file_name = os.listdir(json_folder_path)

    

#     for file_name in json_file_name:

#         file = os.path.join(json_folder_path, file_name)

#         all_filenames.append(file)

     



#     for file in tqdm(all_filenames):

#         converted_file = json.load(open(file, 'rb'))

#         unprocessed_files.append(converted_file)

    

#     return unprocessed_files



# def generate_df(all_files):

#     cleaned_files = []

    

#     for file in tqdm(all_files):

#         features = [

#             file['paper_id'],

#             file['metadata']['title'],

#             file['abstract'],

#             file['body_text'],

#             file['back_matter']

#         ]



#         cleaned_files.append(features)



#     col_names = ['paper_id', 'title', 'abstract', 'text', 

#                  'back_matter']



#     clean_df = pd.DataFrame(cleaned_files, columns=col_names)

#     clean_df.head()

    

#     return clean_df



# def generate_master_df(paths):

#     temp_paths = {}

#     temp_dfs = {}

#     master_df = pd.DataFrame()

#     for i, v in enumerate(paths):

#         temp_paths['raw_files_'+ str(i)] = get_files(v)

#         temp_dfs['df_' + str(i)] = generate_df(temp_paths['raw_files_' + str(i)])

#         master_df.append(temp_dfs['df_' + str(i)])

#     return master_df



# master_df2 = generate_master_df(paths[2:4])

pd.options.display.max_colwidth = 100

pd.set_option('display.max_columns', None)

master = pd.read_csv('/kaggle/input/masterfile/master_df.csv')
cols = ['text', 'abstract', 'back_matter']

def text_clean_up(df, columns):

    for col in columns:

        df[col] = df[col].astype(str)

        df[col] = df[col].map(lambda x: x[10:] if len(x) else 'NA')

        df[col] = df[col].map(lambda x: x.replace(']}', ""))

    return df



master = text_clean_up(master, cols)

master.head()
def preprocess_text(string):

    clean_text = []

    tokens = word_tokenize(string)

    print(len(tokens))

    tokens_lower_case = [token.lower() for token in tokens]

    tokens_punc_removed = [token for token in tokens_lower_case if token.isalnum()]

    stopwords_english = set(stopwords.words('english'), 'et', 'al', 'figure', 'table')

    tokens_stops_removed = [token for token in tokens_punc_removed if token not in stopwords_english]

    for word in tokens_stops_removed:

        clean_text.append(word)

    return clean_text





text_vectors['abstract']=text_vectors['abstract'].map(preprocess_text)

text_vectors['abstract'].head()
dictionary = corpora.Dictionary(text_vectors['abstract'])

corpus = [dictionary.doc2bow(text) for text in text_vectors['abstract']]

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)

n_topics = 10

lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
#lsi.print_topics(25)

#lda.print_topics(20)
from collections import OrderedDict



topics = OrderedDict()

for i in range(n_topics):

    topic = lda.show_topic(i)

    for i in range(len(topic)):

        topics[topic[i][0]] = topic[i][1]
from operator import itemgetter    

print(sorted(topics.items(), key = itemgetter(1), reverse = True))
for i in range(0, n_topics):

    temp = lsi.show_topic(i, 10)

    terms = []

    for term in temp:

        terms.append(term[0])

    print("Top 10 terms for topic #" + str(i) + ": "+ ",".join(terms))
for i in range(0, n_topics):

    temp = lda.show_topic(i, 10)

    terms = []

    for term in temp:

        terms.append(term[0])

    print("Top 10 terms for topic #" + str(i) + ": "+ ",".join(terms))