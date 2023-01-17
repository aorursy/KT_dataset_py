# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from tqdm.notebook import tqdm

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.
import os

import json

from pprint import pprint

from copy import deepcopy



import numpy as np

import pandas as pd
def format_name(author):

    middle_name = " ".join(author['middle'])

    

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])





def format_affiliation(affiliation):

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))

    

    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)



def format_authors(authors, with_affiliation=False):

    name_ls = []

    

    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)

    

    return ", ".join(name_ls)



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



def format_bib(bibs):

    if type(bibs) == dict:

        bibs = list(bibs.values())

    bibs = deepcopy(bibs)

    formatted = []

    

    for bib in bibs:

        bib['authors'] = format_authors(

            bib['authors'], 

            with_affiliation=False

        )

        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]

        formatted.append(", ".join(formatted_ls))



    return "; ".join(formatted)
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

            format_authors(file['metadata']['authors']),

            format_authors(file['metadata']['authors'], 

                           with_affiliation=True),

            format_body(file['abstract']),

            format_body(file['body_text']),

            format_bib(file['bib_entries']),

            file['metadata']['authors'],

            file['bib_entries']

        ]



        cleaned_files.append(features)



    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    

    return clean_df
biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/'

filenames = os.listdir(biorxiv_dir)
all_files = []



for filename in filenames:

    filename = biorxiv_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)
cleaned_files = []



for file in tqdm(all_files):

    features = [

        file['paper_id'],

        file['metadata']['title'],

        format_authors(file['metadata']['authors']),

        format_authors(file['metadata']['authors'], 

                       with_affiliation=True),

        format_body(file['abstract']),

        format_body(file['body_text']),

        format_bib(file['bib_entries']),

        file['metadata']['authors'],

        file['bib_entries']

    ]

    

    cleaned_files.append(features)
len(cleaned_files)
col_names = [

    'paper_id', 

    'title', 

    'authors',

    'affiliations', 

    'abstract', 

    'text', 

    'bibliography',

    'raw_authors',

    'raw_bibliography'

]



clean_df = pd.DataFrame(cleaned_files, columns=col_names)
len(clean_df)
smalldf=clean_df[['title','abstract','text','authors']]
#smalldf['text'].iloc[1]

from gensim.summarization.summarizer import summarize

smalldf['text'] = smalldf['text'].apply(lambda x: summarize(x))
clean_df['text'].iloc[1]
smalldf['text'].iloc[1]
import regex as re

from nltk.stem import WordNetLemmatizer

from nltk.stem import LancasterStemmer, SnowballStemmer, RegexpStemmer, WordNetLemmatizer 

#this was part of the NLP notebook

import nltk

#import sentence tokenizer

from nltk import sent_tokenize

#import word tokenizer

from nltk import word_tokenize

#list of stopwords

from nltk.corpus import stopwords

import string

import spacy

from spacy.lemmatizer import Lemmatizer

from spacy.lookups import Lookups

from nltk.corpus import stopwords

from textblob import TextBlob

import pandas as pd

from collections import Counter
def remove_urls(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub(r'', text)

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):

    """custom function to remove the stopwords"""

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

punct =[]

punct += list(string.punctuation)

punct += '’'

punct.remove("'")



def remove_punctuations(text):

    for punctuation in punct:

        text = text.replace(punctuation, ' ')

    return text

def cleanText(col):

    smalldf[col]=smalldf[col].str.lower()

    smalldf[col]= smalldf[col].apply(lambda x: x.replace('\n', ' '))

    smalldf[col]= smalldf[col].apply(lambda text: remove_urls(text))

    smalldf[col]= smalldf[col].str.split()

    smalldf[col]= smalldf[col].apply(lambda final_df: [x for x in final_df if x.isalpha()])

    smalldf[col]= smalldf[col].str.join(' ')

    smalldf[col]= smalldf[col].apply(lambda text: remove_punctuations(text))

    smalldf[col]= smalldf[col].apply(lambda text: lemmatize_words(text))

    smalldf[col]= smalldf[col].apply(lambda text: remove_stopwords(text))

    smalldf[col]= smalldf[col].map(lambda x: re.sub(r'\d+', '', x))
cleanText('title')

cleanText('abstract')

cleanText('text')
smalldf.tail()


def extraCleaning(col):

    smalldf[col]=smalldf[col].str.lower()

    smalldf[col]=smalldf[col].str.replace('title','')

    smalldf[col]=smalldf[col].str.replace('abstract','')

    smalldf[col]=smalldf[col].str.replace('preprint','')

    smalldf[col]=smalldf[col].str.replace('biorxiv','')

    smalldf[col]=smalldf[col].str.replace('author','')

    smalldf[col]=smalldf[col].str.replace('copyright','')

    smalldf[col]=smalldf[col].str.replace('holder','')

    smalldf[col]=smalldf[col].str.replace('https','')

    smalldf[col]=smalldf[col].str.replace('license','')

    smalldf[col]=smalldf[col].str.replace('wa','')

    smalldf[col]=smalldf[col].str.replace('ha','')

    smalldf[col]=smalldf[col].str.replace('medrxiv','')

    smalldf[col]=smalldf[col].str.replace('granted','')

    smalldf[col]=smalldf[col].str.replace('rights','')

    smalldf[col]=smalldf[col].str.replace('reserved','')

    smalldf[col]=smalldf[col].str.replace('holder','')

extraCleaning('title')

extraCleaning('abstract')

extraCleaning('text')
smalldf.head()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

abstactCorpus = smalldf.abstract.values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',max_words=1000).generate(str(abstactCorpus))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.title('Abstract Corpus')

plt.show()
titleCorpus = smalldf.title.values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',max_words=1000).generate(str(titleCorpus))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.title('Title Corpus')



plt.show()


textCorpus = smalldf.text.values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',max_words=1000).generate(str(textCorpus))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.title('Text Corpus')



plt.show()
x=' '.join(str(x) for x in textCorpus) 

import collections

from nltk.util import ngrams 

tokens=x.split()

op = ngrams(tokens, 1)

op = collections.Counter(op)

op.most_common(50)
op2 = ngrams(tokens, 2)

op2 = collections.Counter(op2)

op2.most_common(50)
op2 = ngrams(tokens, 3)

op2 = collections.Counter(op2)

op2.most_common(100)
countVaccine = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('vaccine'), x))

countVaccine
countCare = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('care'), x))

countCare
countDiagnostics = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('diagnostics'), x))

countDiagnostics
def countOccurences(a, word): 

      

    x = a.split(" ") 

  

    c = 0

    for i in range(0, len(x)): 



        if (word == x[i]): 

           c = c + 1

             

    return c  
smalldf['VaccineCount'] = smalldf['text'].apply(lambda x: countOccurences(x,'vaccine'))
smalldf[smalldf['VaccineCount']>0]
smalldf['CareCount'] = smalldf['text'].apply(lambda x: countOccurences(x,'care'))

smalldf[smalldf['CareCount']>0]
smalldf['DiagnosticsCount'] = smalldf['text'].apply(lambda x: countOccurences(x,'diagnostics'))

smalldf[smalldf['DiagnosticsCount']>0]
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk.tokenize import word_tokenize

from sklearn.manifold import TSNE

!pip install TSNE
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(smalldf['text'])]
max_epochs = 100

vec_size = 40

alpha = 0.025



model = Doc2Vec(size=vec_size,alpha=alpha, min_alpha=0.00025,min_count=1,dm =1)

  

model.build_vocab(tagged_data)



for epoch in range(max_epochs):

    print('iteration {0}'.format(epoch))

    model.train(tagged_data,

                total_examples=model.corpus_count,

                epochs=model.iter)

    # decrease the learning rate

    model.alpha -= 0.0002

    # fix the learning rate, no decay

    model.min_alpha = model.alpha
model.docvecs.most_similar(0)
vec=[]

for i in range(0,len(model.docvecs)):

    vec.append(model.docvecs[i])
v=pd.DataFrame(vec)
tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=1000)

tsne_results = tsne.fit_transform(v)
import seaborn as sns

plt.figure(figsize=(16,10))



ax=sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1],palette=sns.color_palette("hls", 10),legend="full",alpha=0.3,color='red')





ax.set(xlabel='Dimension 1', ylabel='Dimension 2')



plt.title('Vector Representation of Text Corpus in 2D')

plt.show()
smalldf['OriginalTitle']=clean_df['title']

smalldf['OriginalAbstract']=clean_df['abstract']

smalldf['OriginalText']=clean_df['text']
smalldf.head(3)