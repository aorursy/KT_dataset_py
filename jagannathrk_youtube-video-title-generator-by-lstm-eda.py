import numpy as np 

import pandas as pd 

import os

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

import seaborn as sns

%matplotlib notebook



import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline





from lime.lime_text import LimeTextExplainer

from tqdm import tqdm

import string

import random

import operator

import seaborn as sns

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from statistics import *

import concurrent.futures

import time

import pyLDAvis.sklearn

from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig

import warnings

import nltk





# spaCy based imports

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English



# keras module for building LSTM 

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping

from keras.models import Sequential

import keras.utils as ku 



# set seeds for reproducability

from tensorflow import set_random_seed

from numpy.random import seed

set_random_seed(2)

seed(1)



import warnings

warnings.filterwarnings("ignore")
#list of data that we have in the workspace



print(os.listdir("../input"))
# countries that use English as an official language

british_youtube = pd.read_csv("../input/GBvideos.csv")

canadian_youtube = pd.read_csv("../input/CAvideos.csv")

us_youtube = pd.read_csv("../input/USvideos.csv")

canadian_youtube.head()
british_youtube.head()
us_youtube.head()
#combine tables

three_countries=pd.concat([canadian_youtube, british_youtube,us_youtube])

three_countries.shape

#Check duplicate. It is always good to check whether there are some duplicates in dataset!

three_countries.video_id.value_counts()[:10]
#remove duplicate

three_countries= three_countries.drop_duplicates(['video_id'], keep='first')

three_countries.video_id.value_counts()[:10]
#need to be decoded 

three_countries.category_id.head()
import json



def category_name(path):

    with open(path) as json_file:  

        data = json.load(json_file)

    category_info_list=[]

    for row in data['items']:

        id_info=row['id']

        category_name=row['snippet']['title']

        categoty_info=(id_info ,category_name)

        category_info_list.append(categoty_info)

    return(dict(category_info_list))

        

    
category_name("../input/CA_category_id.json")
category_list=category_name("../input/CA_category_id.json")

category_names=[]

for i in three_countries.category_id:

    category_name=category_list.get(str(i))

    category_names.append(category_name)



three_countries['category_names']=category_names
#now, we have category name :)

three_countries['category_names'].head()
three_countries.info()
#give you a report of distribution of data/correlation

import pandas_profiling as pp



pp.ProfileReport(three_countries[['views','likes','dislikes','comment_count']])


Q1 = three_countries.views.quantile(0.25)

Q3 = three_countries.views.quantile(0.75)

IQR = Q3 - Q1

print(IQR)



popular_videos=three_countries.loc[three_countries.views > (Q3 + 1.5 * IQR)]



three_countries['popular']=0

three_countries.loc[three_countries.views > (Q3 + 1.5 * IQR),'popular']=1



three_countries['popular'].value_counts()
#make a variable that tells ratio of like and dislike

three_countries['like_percentage']=(three_countries['likes']/(three_countries['likes']+three_countries['dislikes'])*100)

#date column as datatime datatype

three_countries["publish_time"] = pd.to_datetime(three_countries["publish_time"])
#top 20 channels by the mean of views



fig = plt.figure(figsize=(15, 10))

three_countries.groupby('channel_title').mean().sort_values(by=['views'], ascending=False)[:20].views.sort_values(ascending=True).plot(kind='barh',colormap='winter',fontsize=20)

#among popular videos, here are top 10 like percentage video. All of them are k-pop stars MV lol 



three_countries.loc[three_countries.popular==1].sort_values(by=['like_percentage'], ascending=False)[:10]
#among popular videos, here are top 10 dislike percentage video. 



three_countries.loc[three_countries.popular==1].sort_values(by=['like_percentage'], ascending=True)[:10]
#simple example 



nlp = spacy.load("en_core_web_sm")

doc = nlp(u"An apple is not a banana")

for token in doc:

    print(token.text)
punctuations = string.punctuation

stopwords = list(STOP_WORDS)

parser = English()



def spacy_tokenizer(sentence):

    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]

    mytokens = " ".join([i for i in mytokens])

    return mytokens





tqdm.pandas()



normal = three_countries["title"][three_countries["popular"] == 0].progress_apply(spacy_tokenizer)

popular = three_countries["title"][three_countries["popular"] == 1].progress_apply(spacy_tokenizer)
#tokenize words by popularity 



def word_generator(text):

    word = list(text.split())

    return word

def bigram_generator(text):

    bgram = list(nltk.bigrams(text.split()))

    bgram = [' '.join((a, b)) for (a, b) in bgram]

    return bgram

def trigram_generator(text):

    tgram = list(nltk.trigrams(text.split()))

    tgram = [' '.join((a, b, c)) for (a, b, c) in tgram]

    return tgram





normal_words = normal.progress_apply(word_generator)

popular_words = popular.progress_apply(word_generator)

normal_bigrams = normal.progress_apply(bigram_generator)

popular_bigrams = popular.progress_apply(bigram_generator)

normal_trigrams = normal.progress_apply(trigram_generator)

popular_trigrams = popular.progress_apply(trigram_generator)
#function that makes a pretty word frequency plot



def word_plot(words,my_color):

    slist =[]

    for x in words:

        slist.extend(x)

    fig = plt.figure(figsize=(15, 10))

    pd.Series(slist).value_counts()[:20].sort_values(ascending=True).plot(kind='barh',fontsize=20, color=my_color)

    plt.show()

word_plot(popular_words,'blue')

word_plot(popular_bigrams,'orange')

word_plot(popular_trigrams,'red')
txt1 = ['I like banana', 'An apple is not a banana', 'banana banana oh banana']

tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')

txt_fitted = tf.fit(txt1)

txt_transformed = txt_fitted.transform(txt1)

print ("The text: ", txt1)
tf.vocabulary_
idf = tf.idf_

print(dict(zip(txt_fitted.get_feature_names(), idf)))

print("\nThe token 'banana' appears 5 times but it is also in all documents, so its idf is the lowest")
word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 1),

    max_features=10000)

word_vectorizer.fit(three_countries.title)

word_features = word_vectorizer.transform(three_countries.title)



classifier_popular = LogisticRegression(C=0.1, solver='sag')

classifier_popular.fit(word_features ,three_countries.popular)

names=['normal','popular']
c_tf = make_pipeline( word_vectorizer,classifier_popular)

explainer_tf = LimeTextExplainer(class_names=names)



exp = explainer_tf.explain_instance(three_countries.title.iloc[10], c_tf.predict_proba, num_features=4, top_labels=1)

exp.show_in_notebook(text=three_countries.title.iloc[10])

exp = explainer_tf.explain_instance(three_countries.title.iloc[4], c_tf.predict_proba, num_features=5, top_labels=1)

exp.show_in_notebook(text=three_countries.title.iloc[4])
exp = explainer_tf.explain_instance(three_countries.title.iloc[10035], c_tf.predict_proba, num_features=5, top_labels=1)

exp.show_in_notebook(text=three_countries.title.iloc[10035])
import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.tools as tls





labels = list(three_countries.category_names.value_counts().index.values)

values = list(three_countries.category_names.value_counts().values)



trace = go.Pie(labels=labels, values=values)



iplot([trace], filename='basic_pie_chart')
three_countries.groupby('category_names')['views'].describe()
entertainment_title= three_countries["title"][(three_countries['category_names'] == 'Entertainment')] 

news_politics_title= three_countries["title"][(three_countries['category_names'] == 'News & Politics')] 

people_title= three_countries["title"][(three_countries['category_names'] == 'People & Blogs')] 

music_title= three_countries["title"][(three_countries['category_names'] == 'Music')] 

sports_title= three_countries["title"][(three_countries['category_names'] == 'Sports')] 

comedy_title= three_countries["title"][(three_countries['category_names'] == 'Comedy')] 
vectorizer_entertainment_title = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

entertainment_title_vectorized = vectorizer_entertainment_title.fit_transform(entertainment_title)

lda_popular_entertainment_title_vectorized = LatentDirichletAllocation(n_components=7, max_iter=5, learning_method='online',verbose=True)

entertainment_title_vectorized_lda = lda_popular_entertainment_title_vectorized.fit_transform(entertainment_title_vectorized )



pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda_popular_entertainment_title_vectorized,entertainment_title_vectorized, vectorizer_entertainment_title, mds='tsne')

dash
vectorizer_news_politics_title = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

news_politics_title_vectorized = vectorizer_news_politics_title.fit_transform(news_politics_title)

lda_news_politics_title_vectorized= LatentDirichletAllocation(n_components=7, max_iter=5, learning_method='online',verbose=True)

news_politics_title_vectorized_lda = lda_news_politics_title_vectorized.fit_transform(news_politics_title_vectorized )



pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda_news_politics_title_vectorized,news_politics_title_vectorized, vectorizer_news_politics_title , mds='tsne')

dash
vectorizer_people_title = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

people_title_vectorized = vectorizer_people_title.fit_transform(people_title)

lda_people_title_vectorized= LatentDirichletAllocation(n_components=7, max_iter=5, learning_method='online',verbose=True)

people_title_vectorized_lda = lda_people_title_vectorized.fit_transform(people_title_vectorized )



pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda_people_title_vectorized,people_title_vectorized, vectorizer_people_title , mds='tsne')

dash
vectorizer_music_title = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

music_title_vectorized = vectorizer_music_title.fit_transform(music_title)

lda_music_title_vectorized= LatentDirichletAllocation(n_components=7, max_iter=5, learning_method='online',verbose=True)

music_title_vectorized_lda = lda_music_title_vectorized.fit_transform(music_title_vectorized )



pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda_music_title_vectorized,music_title_vectorized, vectorizer_music_title , mds='tsne')

dash
vectorizer_sports_title = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

sports_title_vectorized = vectorizer_sports_title.fit_transform(sports_title)

lda_sports_title_vectorized= LatentDirichletAllocation(n_components=7, max_iter=5, learning_method='online',verbose=True)

sports_title_vectorized_lda = lda_sports_title_vectorized.fit_transform(sports_title_vectorized )



pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda_sports_title_vectorized,sports_title_vectorized, vectorizer_sports_title , mds='tsne')

dash
vectorizer_comedy_title = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

comedy_title_vectorized = vectorizer_comedy_title.fit_transform(comedy_title)

lda_comedy_title_vectorized= LatentDirichletAllocation(n_components=7, max_iter=5, learning_method='online',verbose=True)

comedy_title_vectorized_lda = lda_comedy_title_vectorized.fit_transform(comedy_title_vectorized )



pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda_comedy_title_vectorized,comedy_title_vectorized, vectorizer_comedy_title , mds='tsne')

dash
tokenizer = Tokenizer()



def get_sequence_of_tokens(corpus):

    ## tokenization

    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    

    ## convert data to sequence of tokens 

    input_sequences = []

    for line in corpus:

        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):

            n_gram_sequence = token_list[:i+1]

            input_sequences.append(n_gram_sequence)

    return input_sequences, total_words



inp_sequences, total_words = get_sequence_of_tokens(popular)

inp_sequences[:10]
def generate_padded_sequences(input_sequences):

    max_sequence_len = max([len(x) for x in input_sequences])

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len



predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
def create_model(max_sequence_len, total_words):

    input_len = max_sequence_len - 1

    model = Sequential()

    

    # Add Input Embedding Layer

    model.add(Embedding(total_words, 10, input_length=input_len))

    

    # Add Hidden Layer 1 - LSTM Layer

    model.add(LSTM(100))

    model.add(Dropout(0.1))

    

    # Add Output Layer

    model.add(Dense(total_words, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='adam')

    

    return model



model = create_model(max_sequence_len, total_words)

model.summary()
model.fit(predictors, label, epochs=50, verbose=5)

def generate_text(seed_text, next_words, model, max_sequence_len):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted = model.predict_classes(token_list, verbose=0)

        

        output_word = ""

        for word,index in tokenizer.word_index.items():

            if index == predicted:

                output_word = word

                break

        seed_text += " "+output_word

    return seed_text.title()
print (generate_text("Drake", 5, model, max_sequence_len))

print (generate_text("united states", 5, model, max_sequence_len))

print (generate_text("Bangtan", 4, model, max_sequence_len))

print (generate_text("Fergie", 4, model, max_sequence_len))

print (generate_text("korea", 4, model, max_sequence_len))

print (generate_text("Minnesota", 4, model, max_sequence_len))