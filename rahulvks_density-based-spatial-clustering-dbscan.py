# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import re

import os

# General libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time

import seaborn as sns



# Libraries for data cleaning



import re

import string

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=False)



import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

punctuations = string.punctuation



#nlp = spacy.load('en')

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

parser = English()





from geopy.geocoders import Nominatim

from folium.plugins import HeatMap

import folium



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv")

location = pd.read_csv("../input/location-fakevsreal/location.csv")



df.head(2)

## Location 

df['location'] = df['location'].astype(str)

df['Loc_1'] = df['location'].str.split(',').str[0]

df['Loc_2'] = df['location'].str.split(',').str[1]

df['Loc_3'] = df['location'].str.split(',').str[2]
## Target Distribution

fig, axes = plt.subplots(ncols=2, figsize=(17, 5), dpi=100)

plt.tight_layout()



df["fraudulent"].value_counts().plot(kind='pie', ax=axes[0], labels=['Real Post (17014)', 'Fake Post (866)'])

temp = df["fraudulent"].value_counts()

sns.barplot(temp.index, temp, ax=axes[1],color='#E1396C')



axes[0].set_ylabel(' ')

axes[1].set_ylabel(' ')

axes[1].set_xticklabels(["Real Post (17014) [0]", "Fake Post (866) [1]"])



axes[0].set_title('Dataset - Distribution', fontsize=13)

axes[1].set_title('Target Count in Dataset', fontsize=13)



plt.show()
f1, axes = plt.subplots(2, 2, figsize=(16,15))

axes = axes.flatten()

f1.subplots_adjust(hspace=0.2, wspace=0.4)



ax1 = sns.barplot(y=df['location'].value_counts()[:20].index,x=df['location'].value_counts()[:20],

            orient='h', ax=axes[0],palette='Blues_d')

ax1.set_title("Top 20 Location")



ax2 = sns.barplot(y=df['department'].value_counts()[:20].index,x=df['department'].value_counts()[:20],

            orient='h', ax=axes[1],palette='Blues_d')

ax2.set_title("Top 20 Department")





ax3 = sns.barplot(y=df['industry'].value_counts()[:20].index,x=df['industry'].value_counts()[:20],

            orient='h', ax=axes[2],palette='Blues_d')

ax3.set_title("Top 20 Industry")





ax4 = sns.barplot(y=df['function'].value_counts()[:20].index,x=df['function'].value_counts()[:20],

            orient='h', ax=axes[3],palette='Blues_d')

ax4.set_title("Top 20 Functions")
%matplotlib inline

temp = df["Loc_3"].value_counts()

sns.barplot(temp.index, temp, ax=axes[1])

def custom_preprocessor(text):

    '''

    Make text lowercase, remove text in square brackets,remove links,remove special characters

    and remove words containing numbers.

    '''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub("\\W"," ",text) # remove special chars

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    

    return text



df['description'] = df['description'].astype(str)

df['requirements'] = df['requirements'].astype(str)

df['benefits'] = df['benefits'].astype(str)

df['company_profile'] = df['company_profile'].astype(str)

df['description'] = df['description'].apply(custom_preprocessor)    

df['requirements'] = df['requirements'].apply(custom_preprocessor)    

df['benefits'] = df['benefits'].apply(custom_preprocessor)    

df['company_profile'] = df['company_profile'].apply(custom_preprocessor)    

%time

### Spacy Lemma # Own Stop words

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

def spacy_lemma_text(text):

    doc = nlp(text)

    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']

    tokens = [tok for tok in tokens if tok not in spacy_stopwords and tok not in punctuations]

    tokens = ' '.join(tokens)

    return tokens



df['description'] = df['description'].apply(spacy_lemma_text)    

df['requirements'] = df['requirements'].apply(spacy_lemma_text)    

df['benefits'] = df['benefits'].apply(spacy_lemma_text)  

df['company_profile'] = df['company_profile'].apply(spacy_lemma_text)  

df['combined_text'] = df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits']

df['combined_text'] = df['combined_text'].astype(str)

df['combined_text'] = df['combined_text'].apply(custom_preprocessor)    

df['combined_text'] = df['combined_text'].apply(spacy_lemma_text)    
df.head()
def length(text):    

    '''a function which returns the length of text'''

    return len(text)



df['Com_length'] = df['combined_text'].apply(length)

plt.rcParams['figure.figsize'] = (10.0, 6.0)

bins = 100

plt.hist(df[df['fraudulent'] == 0]['Com_length'], alpha = 0.6, bins=bins, label='Fake Job')

plt.hist(df[df['fraudulent'] == 1]['Com_length'], alpha = 0.8, bins=bins, label='Real Job')

plt.xlabel('Distribution of tokens')

plt.ylabel('numbers')

plt.legend(loc='upper right')

plt.xlim(0,150)

plt.grid()

plt.show()
## Geo location 
location.head()
!pip install pytextrank

import spacy

import pytextrank

nlp = spacy.load('en_core_web_sm')

tr = pytextrank.TextRank()

nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)
#pytext_key = []



#for text in df['combined_text']:

#    text = nlp(text)

#    t = text._.phrases

#    pytext_key.append(t)

    

#df['Pytextrank_keyword'] = pytext_key        

    
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def prepare_similarity(vectors):

    similarity=cosine_similarity(vectors)

    return similarity



def get_top_similar(sentence, sentence_list, similarity_matrix, topN):

    # find the index of sentence in list

    index = sentence_list.index(sentence)

    # get the corresponding row in similarity matrix

    similarity_row = np.array(similarity_matrix[index, :])

    # get the indices of top similar

    indices = similarity_row.argsort()[-topN:][::-1]

    return [sentence_list[i] for i in indices]



titles=df['description'].fillna("Unknown")

embed_vectors=embed(titles.values).numpy()

sentence_list=titles.values.tolist()

sentence=titles.iloc[1]

print("Find similar research papers for :")

print(sentence)



similarity_matrix=prepare_similarity(embed_vectors)

similar=get_top_similar(sentence,sentence_list,similarity_matrix,6)

for sentence in similar:

    print(sentence)

    print("\n")

def prepare_similarity(vectors):

    similarity=cosine_similarity(vectors)

    return similarity



def get_top_similar(sentence, sentence_list, similarity_matrix, topN):

    # find the index of sentence in list

    index = sentence_list.index(sentence)

    # get the corresponding row in similarity matrix

    similarity_row = np.array(similarity_matrix[index, :])

    # get the indices of top similar

    indices = similarity_row.argsort()[-topN:][::-1]

    return [sentence_list[i] for i in indices]







com=df['combined_text'].fillna("Unknown")

embed_vectors=embed(com.values).numpy()

sentence_list=com.values.tolist()

sentence=com.iloc[5]

print("Find similar requirements in the job post")

print(sentence)

similarity_matrix=prepare_similarity(embed_vectors)

similar=get_top_similar(sentence,sentence_list,similarity_matrix,10)

for sentence in similar:

    print(sentence)

    print("\n")
