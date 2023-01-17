#importing necessary libraries 

import json
import glob
import re
import string
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from random import sample


# adding a label to each column that specifies the nature of the data in that column.
col_list = ['cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id', 'license', 'publish_time',
            'authors', 'journal', 'url']
metadata = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')

#creating another df with a more output-friendly view for exploration
metadata_view = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', usecols = col_list)

# gives few rows of dataframe
metadata_view.head()
# gives column count and other info of original metadata df
metadata.info()
print(f"{metadata['abstract'].isnull().sum()} papers with no abstracts provided")
print(metadata.isnull().sum())
# here we'll load the JSON files by gathering the path to all JSON files
# glob module allows for reading of all JSON files in the All JSON folder

all_json = glob.glob('../input/CORD-19-research-challenge/document_parses/**/*.json',
                     recursive=True)

# creating a reader class to read the files
class JSONFileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            # paper_id
            self.paper_id = content['paper_id']
            # title and initializing abstract and body_text attributes
            self.title = content['metadata'].get("title")
            self.abstract = []
            self.body_text = []
            # abstract
            for word in content['abstract']:
                self.abstract.append(word.get("text"))
            # body_text
            for word in content['body_text']:
                self.body_text.append(word.get("text"))
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
            
        # repr function gives a printable version of a given object
        def __repr__(self):
            return f'{self.paper_id}: {self.abstract[:100]}... {self.body_text[:100]}...'
# defining a data dictionary before reading through the files will help us input each attribute in each column as we iterate through the list of files
datadict = {'paper_id': [], 'title': [], 'doi': [], 'journal': [], 'authors': [], 'abstract': [], 'body_text': []}

# iterating and reading JSON files, selecting a sample for runtime purposes may be helpful
for file in sample(all_json,20000):
    try:
        reader = JSONFileReader(file)
        # you may see here that I am comparing the 'sha' column to the JSON file's paper_id using 'contains'. This is because some of the 'sha' column entries
        # include multiple paper_id entries
        sub = metadata[metadata['sha'].str.contains(reader.paper_id, na = False, flags = re.IGNORECASE, regex = False)]
        datadict['paper_id'].append(reader.paper_id)
        datadict['title'].append(reader.title)
        doilist = list(sub['doi'])
        # checking if the list is null, and setting it to N/A rather than the nan type, which is always annoying to deal with (python being a bad friend).
        if not doilist:
            doilist.append('N/A')
        journallist = list(sub['journal'])
        if not journallist:
            journallist.append('N/A')
        datadict['doi'].append(doilist[0])
        datadict['journal'].append(journallist[0])
        authorlist = list(sub['authors'])
        if not authorlist:
            authorlist.append('N/A')
        datadict['authors'].append(authorlist[0])
        if not list(reader.abstract):
            reader.abstract = 'N/A'
        datadict['abstract'].append(reader.abstract)
        if not list(reader.body_text):
            reader.body_text = 'N/A'
        datadict['body_text'].append(reader.body_text)
    # always remember to set an exception clause
    except Exception as e:
        continue

# create dataframe
data = pd.DataFrame(datadict, columns = ['paper_id', 'title','doi','journal','authors','abstract','body_text'])
# Checked for duplicates. We can safely assume there are little to no duplicates in the JSON list and it is clean
print(f" {len(data)} before")
data.drop_duplicates()
print(f" {len(data)} after")
# LANGUAGE DETECTION
!pip install langdetect
from langdetect import detect
from langdetect import DetectorFactory

# set seed
DetectorFactory.seed = 0

langlist = []

for i in range(0, len(data)):
    # split by space into list, take the first x index, separate by space
    text = data.iloc[i]['body_text'].split(" ")

    lang = "en"
    all_words = set(text)
    try:
        lang = detect(" ".join(all_words))
    except Exception as e:
        try:
            lang = detect(data.iloc[i]['abstract_summary'])
        except Exception as e:
            lang = "unknown"
            pass
    # get the language
    langlist.append(lang)

language_dict = {}
for x in langlist:
    language_dict[x] = langlist.count(x)

print(set(langlist))

# not all sources are in english, for our purposes here let's limit use to only english sources
# create and assign a column of languages, limit use to only the english papers

data['languages'] = langlist
data = data[data['languages']=='en']
# TEXT PROCESSING

from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS

# we need to remove punctuation and convert all words to their lowercase. 
# In addition, I added custom stopwords for words that appear consistently but do not provide value to the analysis. For example, 'PMC' refers to the PubMed Central archive, which does not provide value to topic modeling.

# remove punctuation
punctuations = string.punctuation
punc = list(punctuations)

custom_stop = ['``', ':', '/', 'N/A', 'etc.', 'it', 'The', 'For', ';', 'his', 'her', 'you', 'an', 'at', 'be', 'they', 'or', 'on', 'them', 'these', 'into', 'from', 'while', 'this', 'also', 'was', 'with', 'not', 'to', 'in', 'their', "''", 'are', 'by', 'per', 'as', 'is', 'that', 'the', 'and', 'of', 'a', 'for', 'doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'author', 'figure', 'table', 'http',
    'rights', 'I', 'biorxiv', 'medrxiv', '19', 'used', 'using', 'license', 'fig', 'fig.', 'al.', 'infl', 'uenza', 'Elsevier', 'PMC', 'CZI', '-PRON-', 'usually', '10']

# appending custom stopwords and punctuation
custom_stop.append(STOP_WORDS)

temp = []

for x in range(0, len(data)):
    text = data.iloc[x]['body_text']
    text_tokenized = word_tokenize(text)
    for i in range(0, len(text_tokenized)):
        text_tokenized[i] = text_tokenized[i].lower()
    text_tokenized = [word for word in text_tokenized if word not in custom_stop and word not in punc]
    temp.append(" ".join(text_tokenized))

token_final = pd.DataFrame(temp, columns=['body_text'])
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pyLDAvis
from pyLDAvis import sklearn as sklearn_lda

count_vectorizer = CountVectorizer(stop_words='english')

def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Visualising the 10 most common words

count_data = count_vectorizer.fit_transform(token_final['body_text'].values)
plot_10_most_common_words(count_data, count_vectorizer)
# counting the frequencies of each word
countvector = CountVectorizer(strip_accents='unicode', stop_words = 'english', lowercase=True,
                                  token_pattern=r'\b[a-zA-Z]{3,}\b')
dtm_tf = countvector.fit_transform(token_final['body_text'].values)
print(dtm_tf.shape)

# associated articles with 5 unique topics but fuzzy associations (multiple words may appear in multiple topic spaces)
n_topics = 5
lda_tf = LDA(n_components=n_topics, max_iter=10, learning_method='online',
             verbose=False, random_state=42)
lda_tf.fit(dtm_tf)
LDAvis = sklearn_lda.prepare(lda_tf, dtm_tf, countvector)
pyLDAvis.display(LDAvis)