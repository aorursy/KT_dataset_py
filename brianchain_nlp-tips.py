# LOAD DATASET

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import os
import json
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body


def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
           # format_authors(file['metadata']['authors']),
           # format_authors(file['metadata']['authors'], 
            #               with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            #format_bib(file['bib_entries']),
           # file['metadata']['authors'],
            #file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 
                 'abstract', 'text', 
                 ]

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df



# LOAD DIRECTORY, PARSE AND PRINT TARGET JSON SUB-TEXT


biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))

all_files = []

for filename in filenames:
    filename = biorxiv_dir + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
    
#print('Test', all_files)
    
file = all_files[0]
print("Dictionary keys:", file.keys())

pprint(file['abstract'])

# FORMAT AND PRINT TARGET KEY/VALUES


print("body_text type:", type(file['body_text']))
print("body_text length:", len(file['body_text']))
print("body_text keys:", file['body_text'][0].keys())

print("body_text content:")
pprint(file['body_text'][:2], depth=3)

texts = [(di['section'], di['text']) for di in file['body_text']]
texts_di = {di['section']: "" for di in file['body_text']}
for section, text in texts:
    texts_di[section] += text

pprint(list(texts_di.keys()))

body = ""

for section, text in texts_di.items():
    body += section
    body += "\n\n"
    body += text
    body += "\n\n"

#print(body)
# PRINT CUSTOMIZED JSON TARGET KEY TEXT FROM SINGLE FILE
print(format_body(file['body_text']))

# BUILD CUSTOM COLUMN LIST FROM ALL FILES 

cleaned_files = []

for file in tqdm(all_files):
    features = [
        file['paper_id'],
        file['metadata']['title'],
        #format_authors(file['metadata']['authors']),
        #format_authors(file['metadata']['authors'], 
        #              with_affiliation=True),
        format_body(file['abstract']),
        format_body(file['body_text']),
       # format_bib(file['bib_entries']),
       # file['metadata']['authors'],
       # file['bib_entries']
    ]
    
    cleaned_files.append(features)
# LOAD CUSTOM LIST INTO CSV PD DATAFRAME

col_names = [
    'paper_id', 
    'title', 
    #'authors',
    #'affiliations', 
    'abstract', 
    'text', 
    #'bibliography',
    #'raw_authors',
    #'raw_bibliography'
]

clean_df = pd.DataFrame(cleaned_files, columns=col_names)
clean_df.head()
# LOAD DIRECTORY , CLEAN, AND CREATE CSV DATAFRAME

pmc_dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'
pmc_files = load_files(pmc_dir)
pmc_df = generate_clean_df(pmc_files)
pmc_df.to_csv('clean_pmc.csv', index=False)
pmc_df.head()
# FIND MOST FREQUENT WORD WITHIN TARGET COLUMN

import nltk
from nltk import FreqDist
import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy
import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# function to plot most frequent terms
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()
    

# TARGET COLUMN
freq_words(pmc_df['abstract'])

pmc_df['abstract'] = pmc_df['abstract'].str.replace("[^a-zA-Z#]", " ")

# REMOVE STOP-WORDS 

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

# remove short words (length < 3)
pmc_df['abstract'] = pmc_df['abstract'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# remove stopwords from the text
clean_text = [remove_stopwords(r.split()) for r in pmc_df['abstract']]

# make entire text lowercase
clean_text = [r.lower() for r in clean_text]

freq_words(clean_text, 35)
# Lemmatize text

nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output


tokenized_text = pd.Series(clean_text).apply(lambda x: x.split())
print(tokenized_text[1])

lemmat_text = lemmatization(tokenized_text)
print(lemmat_text[1]) # print lemmatized review
# MOST RELAVENT FREQUENT WORDS

freq_list = []
for i in range(len(lemmat_text)):
    freq_list.append(' '.join(lemmat_text[i]))

pmc_df['abstract'] = freq_list

freq_words(pmc_df['abstract'], 35)
# Building LDA Model to find Topics 

dictionary = corpora.Dictionary(lemmat_text)

doc_term_matrix = [dictionary.doc2bow(rev) for rev in lemmat_text]

# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100,
                chunksize=1000, passes=50)
# PRINT FINAL TOPICS

lda_model.print_topics()
# Visualize the topics

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
vis