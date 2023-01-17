import os

import json

from pprint import pprint

from copy import deepcopy

from tqdm.notebook import tqdm



from nltk.corpus import stopwords



import numpy as np

import random

import string

import re

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import pandas as pd

import collections

from nltk.util import ngrams

import nltk

import networkx as nx

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
# Some procedures from https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv

biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'

filenames = os.listdir(biorxiv_dir)



all_files = []



for filename in filenames:

    filename = biorxiv_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)



    

print(f'Total number of files: {len(all_files)}')
# Visualise the keys (columns) of the dataset

fileTest = all_files[0]

print("Dictionary keys:", fileTest.keys())
all_files = all_files[0:400] # Sample of the dataset for testing

print(f'Total number of files used: {len(all_files)}')
# Creating some functions to help us in pre-processing and ploting



# Create function to perform pre processing

def pre_treatment_ngram(text):

    # Get the text in lower case

    text = text.lower()

    # Remove everything from our dataset except letters, and spaces

    text = re.sub(r'[^A-Za-z\d ]', '', text)

    # Clean the text with regex, remove Coprights and doi.org urls 

    text = re.sub(r'[Dd]oi:.*\.|http.*\.|International license.*\.|[Cc]opyright.*\.', '', text)

    text = re.sub(' et | el | la | il | al | li | fig\. ', '', text)

    text = re.sub('[Mm]edrxiv.*\.|[Ll]icence.*\.', '', text) 

    # Tokenization

    words_tokens = nltk.word_tokenize(text)

    # Remove stopwords

    stoplist = stopwords.words('english')

    clean_word_list = [word for word in words_tokens if word not in stoplist]

    return clean_word_list





# Create function do help plot world cloud

def print_word_cloud(dictionary):

    # Convert to string and send to wordcloud

    textNgram = str(dictionary)

    # Clean words

    textNgram = re.sub('[\']', '', textNgram)

    # Send to wordcloud

    wordcloud = WordCloud(width = 800, height = 800,

                    background_color ='white',

                    min_font_size = 10).generate(textNgram)



    # plot the WordCloud image

    plt.figure(figsize = (8, 8), facecolor = None)

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.tight_layout(pad = 0)

    plt.show()



# Function to create graph of conections

# Thanks to https://www.earthdatascience.org/courses/earth-analytics-python/using-apis-natural-language-processing-twitter/calculate-tweet-word-bigrams-networks-in-python/

def create_graph_conections(importantWord):

    # Create dictionary of bigrams and their counts

    d = ngramsList_df.set_index('n-gram').T.to_dict('records')

    # Create network plot 

    G = nx.Graph()



    # Create connections between nodes

    for k, v in d[0].items():

        G.add_edge(k[0], k[1], weight=(v * 10))



    G.add_node(importantWord, weight=100)

    fig, ax = plt.subplots(figsize=(20, 15))



    pos = nx.spring_layout(G, k=1)



    # Plot networks

    nx.draw_networkx(G, pos,

                     font_size=16,

                     width=3,

                     edge_color='grey',

                     node_color='purple',

                     with_labels = False,

                     ax=ax)

    

    # Create offset labels

    for key, value in pos.items():

        x, y = value[0]+.135, value[1]+.045

        ax.text(x, y,

                s=key,

                bbox=dict(facecolor='red', alpha=0.25),

                horizontalalignment='center', fontsize=13)

    plt.show()
# Creating a body text from dataset files

# Some procedures from https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv



textFromFile = 'body_text'



body = ""

for file in all_files:

    try:

        # Reading the text body from file

        texts = [(di['section'], di['text']) for di in file[textFromFile]]

        texts_di = {di['section']: "" for di in file[textFromFile]}

        for section, text in texts:

            texts_di[section] += text

        

        for section, text in texts_di.items():

            body += section

            body += text



        # Clean the text with regex, remove Coprights and doi.org urls 

        body = re.sub(r'[Dd]oi:.*\.|http.*\.|International license.*\.|[Cc]opyright.*\.', '', body)

        body = re.sub(' et | el | la | il | al | li | fig\. ', '', body)

        body = re.sub('[Mm]edr.*\.', '', body)        

    except: 

        continue

    



print(f'Total number of characters, from {textFromFile} of the files:')

print(len(body))

# Performing pre-treatment using the created function

clean_word_list = pre_treatment_ngram(body)

print(f'Total number of filtered words, from {textFromFile} of the files:')

print(len(clean_word_list))

# print(clean_word_list)
# Define de ngram

wordChain = 4

ngramsList = ngrams(clean_word_list, wordChain)

result = collections.Counter(ngramsList)



# Convert to dataframe

ngramsList_df = pd.DataFrame(result.most_common(100),

                             columns=['n-gram', 'count'])







# ngram = ngramsList_df['n-gram']

# y_pos = np.arange(len(ngram))

# counts = ngramsList_df['count']





# plt.figure(figsize=(9,11))

# sns.barplot(x=counts,y=ngram)
# Define de ngram

wordChain = 4



# Define list of important words

importantWords = ['origin', 'COVID',

                  'coronavirus', 'evolution', 'genetics']



for word in importantWords:

    print(f'Network graphic related to the word "{word}"')

    create_graph_conections(word)    
dictNgrams = {}

# Creating a dictionary of ngram

# Thanks for https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/



for i in range(len(clean_word_list)-wordChain):

    seq = ' '.join(clean_word_list[i:i+wordChain])

    if  seq not in dictNgrams.keys():

        dictNgrams[seq] = []

        dictNgrams[seq].append(clean_word_list[i+wordChain])

        

print('Some sentences of ngrams:\n')

for x in list(dictNgrams)[0:20]:

    print (x)
print_word_cloud(dictNgrams)
# List of important words for filtering our sentences

importantWords = ['origin', 'virus', 'transmission', 'COVID',

                  'coronavirus', 'evolution', 'genetics', 'glycoprotein']

dictNgrams = {}

# Thanks for https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/

for i in range(len(clean_word_list)-wordChain):

    seq = ' '.join(clean_word_list[i:i+wordChain])

    for w in importantWords:

         if w in seq:

#             print(seq)

            if  seq not in dictNgrams.keys():

                dictNgrams[seq] = []

                dictNgrams[seq].append(clean_word_list[i+wordChain])

                



# print top keys

k = collections.Counter(dictNgrams) 

  

# Finding 600 values 

high = k.most_common(30)  



print(' Some sentences of ngrams with important words:\n')

print("Keys: Values\n") 

  

for i in high: 

    print(i[0]," :",i[1]," ") 

print('Word cloud from ngrams sentences with important words:\n')



print_word_cloud(dictNgrams)