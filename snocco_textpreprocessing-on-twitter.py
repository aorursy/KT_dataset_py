import sys

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as sk

import nltk

#nltk.download('stopwords')

#nltk.download('wordnet')



print('*'*50)

print('Numpy Version     : ', np.__version__)

print('Pandas Version    : ', pd.__version__)

print('Matplotlib Version: ', mpl.__version__)

print('Seaborn Version   : ', sns.__version__)

print('SKLearn Version   : ', sk.__version__)

print('NLTK Version      : ', nltk.__version__)

print('*'*50)
#seaborn options

sns.set_style('white')



#pandas options

pd.options.display.max_rows = 100

pd.options.display.max_columns = 100



#Reproducibility!

seed = 42

num_folds = 10

v_size = 0.2

metric = 'accuracy'



#Random seeds

np.random.seed(seed)
raw_twcs = pd.read_csv('../input/twcs/twcs.csv',encoding='utf-8')
raw_twcs.sample(10)
raw_twcs.shape
raw_twcs.info()
def missingData(data):

    '''

    @author steno

    Check the missing data in the features.

    This function returns  dataframe with the missing values.

    '''

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    md = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    md = md[md["Percent"] > 0]

    sns.set(style = 'darkgrid')

    plt.figure(figsize = (8, 4))

    plt.xticks(rotation='90')

    sns.barplot(md.index, md["Percent"],color="g",alpha=0.8)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)

    return md
missingData(raw_twcs)
#first inbound = Richiesta iniziale di un cliente

first_inbound = raw_twcs[pd.isnull(raw_twcs.in_response_to_tweet_id) & raw_twcs.inbound]



inbOutb = pd.merge(first_inbound, raw_twcs, left_on='tweet_id', 

                                  right_on='in_response_to_tweet_id').sample(frac=1)



# Filter to only outbound replies (from companies)

inbOutb = inbOutb[inbOutb.inbound_y ^ True]
inbOutb.shape
missingData(inbOutb)
inbOutb.columns
toDrop = ['tweet_id_x', 'inbound_x','response_tweet_id_x', 'in_response_to_tweet_id_x', 

          'tweet_id_y', 'inbound_y','response_tweet_id_y', 'in_response_to_tweet_id_y']
inbOutb.drop(toDrop, axis=1, inplace=True)

print('inbOutb shape: ', inbOutb.shape)
inbOutb.sample(5)
plt.figure(figsize=(20, 10))

sns.set_style('white')

sns.countplot(x='author_id_y', data=inbOutb )

plt.xticks(rotation = 90)

plt.title("Distributions of Number of Companies' Replies ", fontsize = 20)

plt.show()
inbOutb.info()
inbOutb.shape
def remove_uppercase(text):

    text_lowercase = ' '.join(x.lower() for x in text.split())# It will discard all uppercases

    return text_lowercase
inbOutb['text_x_clean'] = inbOutb['text_x'].apply(lambda x: remove_uppercase(x))

inbOutb['text_y_clean'] = inbOutb['text_y'].apply(lambda x: remove_uppercase(x))

#in modo da poter rimuovere i nomi delle compagnie

inbOutb['author_id_y'] = inbOutb['author_id_y'].apply(lambda x: remove_uppercase(x)) 
inbOutb.head(3)
import string

string.punctuation
#Function to remove Punctuation

def remove_punct(text):

    text_nopunct = "".join([char for char in text if char not in string.punctuation])# It will discard all punctuations

    return text_nopunct
inbOutb['text_x_clean'] = inbOutb['text_x_clean'].apply(lambda x: remove_punct(x))

inbOutb['text_y_clean'] = inbOutb['text_y_clean'].apply(lambda x: remove_punct(x))
inbOutb.head(3)
#usernames = inbOutb['author_id_x'].unique()

companies = inbOutb['author_id_y'].unique()
# gli username dei clienti sono numeri

inbOutb['text_x_clean'] = inbOutb['text_x_clean'].str.replace('\d+', '')

inbOutb['text_y_clean'] = inbOutb['text_y_clean'].str.replace('\d+', '')
inbOutb['text_x_clean'] = inbOutb['text_x_clean'].str.replace('|'.join(companies), '')

inbOutb['text_y_clean'] = inbOutb['text_y_clean'].str.replace('|'.join(companies), '')
inbOutb.head(3)
inbOutb.shape
#common worlds



freqX = pd.Series(' '.join(inbOutb['text_x_clean']).split()).value_counts()[:10]

freqY = pd.Series(' '.join(inbOutb['text_y_clean']).split()).value_counts()[:10]

print('FREQ X: \n',freqX,'\nFREQ Y: \n', freqY)
#removing them

freqX = list(freqX.index)

freqY = list(freqY.index)

inbOutb['text_x_clean'] = inbOutb['text_x_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in freqX))

inbOutb['text_y_clean'] = inbOutb['text_y_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in freqY))
rareX = pd.Series(' '.join(inbOutb['text_x_clean']).split()).value_counts()[-100:]

rareY = pd.Series(' '.join(inbOutb['text_y_clean']).split()).value_counts()[-100:]

print('RARE X: \n',rareX,'\nRARE Y: \n', rareY)
#removing them

rareX = list(rareX.index)

rareY = list(rareY.index)

inbOutb['text_x_clean'] = inbOutb['text_x_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in rareX))

inbOutb['text_y_clean'] = inbOutb['text_y_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in rareY))
import re



# Function to Tokenize words

def tokenize(text):

    tokens = re.split('\W+', text) #W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.

    return tokens
inbOutb['text_x_tokenized'] = inbOutb['text_x_clean'].apply(lambda x: tokenize(x.lower())) 

inbOutb['text_y_tokenized'] = inbOutb['text_y_clean'].apply(lambda x: tokenize(x.lower()))

#We convert to lower as Python is case-sensitive. 
inbOutb.head(3)
import nltk



stopword = nltk.corpus.stopwords.words('english') # All English Stopwords



# Function to remove Stopwords

def remove_stopwords(tokenized_list):

    text = [word for word in tokenized_list if word not in stopword]# To remove all stopwords

    return text



inbOutb['text_x_tokenized'] = inbOutb['text_x_tokenized'].apply(lambda x: remove_stopwords(x))

inbOutb['text_y_tokenized'] = inbOutb['text_y_tokenized'].apply(lambda x: remove_stopwords(x))
inbOutb.head(3)
#ps = nltk.PorterStemmer()



#def stemming(tokenized_text):

#    text = [ps.stem(word) for word in tokenized_text]

#    return text
#inbOutb['text_x_stemmed'] = inbOutb['text_x_nostop'].apply(lambda x: stemming(x))

#inbOutb['text_y_stemmed'] = inbOutb['text_y_nostop'].apply(lambda x: stemming(x))
wn = nltk.WordNetLemmatizer()



def lemmatizing(tokenized_text):

    text = [wn.lemmatize(word) for word in tokenized_text]

    return text
inbOutb['text_x_lemmatized'] = inbOutb['text_x_tokenized'].apply(lambda x: lemmatizing(x))

inbOutb['text_y_lemmatized'] = inbOutb['text_y_tokenized'].apply(lambda x: lemmatizing(x))
inbOutb.head(3)
#inbOutb.to_csv('inbOutb.csv')
## Load the library with the CountVectorizer method

from sklearn.feature_extraction.text import CountVectorizer
questions = inbOutb['text_x_clean'].dropna()

q = np.array(questions)
# Initialise the count vectorizer with the English stop words

countV = CountVectorizer(stop_words='english',

                         max_features=10000)



# Fit and transform the processed titles

bagQuestions = countV.fit_transform(q)



print('BOW Questions: ',bagQuestions.shape)
words = countV.get_feature_names()

total_counts = np.zeros(len(words))

for t in bagQuestions:

    total_counts += t.toarray()[0]

    

count_dict = (zip(words, total_counts))

count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)

d = dict(count_dict)
from wordcloud import WordCloud



sns.set_style('white')

plt.figure(figsize=(15,15))

wc = WordCloud(background_color="white",width=1000,height=1000, max_words=50,

               relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(d)

plt.title('WordCloud')

plt.imshow(wc)

plt.show()