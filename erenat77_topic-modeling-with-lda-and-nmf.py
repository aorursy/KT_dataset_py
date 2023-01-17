import pandas as pd

import numpy as np
# Input from csv

df = pd.read_csv('../input/voted-kaggle-dataset.csv')



# sample data

print(df['Description'][0])
df['Title'][0]
df.columns
df.head()
# shape of data frame

len(df)
# is there any NaN values

df.isnull().sum()
# nan value in Description

df.Description.isnull().sum()
df.Tags[0]
#REMOVE NaN VALUES

df['Description'].dropna(inplace=True,axis=0)



# check if there is any NaN values

df.Description.isnull().sum()
# REMOVE EMPTY STRINGS:

blanks = []  # start with an empty list



for rv in df['Description']:  # iterate over the DataFrame

    if type(rv)==str:            # avoid NaN values

        if rv.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list

print(blanks)

df['Description'].drop(blanks, inplace=True)
from nltk.tokenize import RegexpTokenizer

from stop_words import get_stop_words

from nltk.stem.wordnet import WordNetLemmatizer

import re

from nltk.corpus import stopwords



pattern = r'\b[^\d\W]+\b'

# \b is word boundry

# [^] is neget

# \d is digit and \W is not word



#tokenize from nltk

tokenizer = RegexpTokenizer(pattern)

#I created by myself

def tokenizer_man(doc,remove_stopwords=False):

    doc_rem_puct = re.sub(r'[^a-zA-Z]',' ',doc)

    words = doc_rem_puct.lower().split()    

    if remove_stopwords:

        stops = set(stopwords.words("english"))     

        words = [w for w in words if not w in stops]

    return words



en_stop = get_stop_words('en')

lemmatizer = WordNetLemmatizer()
#NTLK stopwords



#check how many stopwords you have

stops1=set(stopwords.words('english'))

print(stops1)

#lenght of stopwords

len(stopwords.words('english'))
#adding new element to the set

stops1.add('newWords') #newWord added into the stopwords

print(len(stops1))
raw = str(df['Description'][0]).lower()

tokens = tokenizer.tokenize(raw)

" ".join(tokens)

len(tokens)
#test manual 

string=df['Description'][0]

vocab = tokenizer_man(string)

" ".join(vocab)

len(vocab)
remove_words = ['data','dataset','datasets','content','context','acknowledgement','inspiration']
# list for tokenized documents in loop

texts = []



# loop through document list

for i in df['Description'].iteritems():

    # clean and tokenize document string

    raw = str(i[1]).lower()

    tokens = tokenizer.tokenize(raw)



    # remove stop words from tokens

    stopped_tokens = [raw for raw in tokens if not raw in en_stop]

    

    # remove stop words from tokens

    stopped_tokens_new = [raw for raw in stopped_tokens if not raw in remove_words]

    

    # lemmatize tokens

    lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens_new]

    

    # remove word containing only single char

    new_lemma_tokens = [raw for raw in lemma_tokens if not len(raw) == 1]

    

    # add tokens to list

    texts.append(new_lemma_tokens)



# sample data

print(texts[0])
len(texts)
df['desc_preprocessed'] = ""

for i in range(len(texts)):

    df['desc_preprocessed'][i] = ' '.join(map(str, texts[i]))
print(df['desc_preprocessed'][0])
df.shape
df.columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.9,min_df=2,stop_words='english')
dtm = tfidf.fit_transform(df['desc_preprocessed'])



dtm
from sklearn.decomposition import NMF,LatentDirichletAllocation
nmf_model = NMF(n_components=7,random_state=42)

nmf_model.fit(dtm)
LDA = LatentDirichletAllocation(n_components=7,random_state=42)

LDA.fit(dtm)
len(tfidf.get_feature_names())
# words for NMF modeling

for index,topic in enumerate(nmf_model.components_):

    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')

    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])

    print('\n')
# words for LDA modeling

for index,topic in enumerate(LDA.components_):

    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')

    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])

    print('\n')
topic_results = nmf_model.transform(dtm)

df['NMF_Topic'] = topic_results.argmax(axis=1)
LDA_topic_results = LDA.transform(dtm)

df['LDA_Topic'] = LDA_topic_results.argmax(axis=1)
mytopic_dict = {0:'public',

                1:'sports',

                2:'machine_learning',

                3:'neuron_network',

                4:'politic',

                5:'economy',

                6:'text analysis'

               }



df['topic_label_NMF']=df['NMF_Topic'].map(mytopic_dict)
df.head(-5)
df['LDA_Topic'].unique()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



first_topic = nmf_model.components_[0]

first_topic_words = [tfidf.get_feature_names()[i] for i in first_topic.argsort()[:-15 - 1 :-1]]



firstcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=4000,

                          height=2500

                         ).generate(" ".join(first_topic_words))

plt.imshow(firstcloud)

plt.axis('off')

plt.show()