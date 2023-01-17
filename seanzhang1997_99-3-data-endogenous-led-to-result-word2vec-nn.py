#Import common packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import string
# import packages for text

import nltk

import re

from wordcloud import WordCloud



#Set ignore warning for futureWarning

import warnings

warnings.filterwarnings("ignore")



#import deep learning packages for text classification

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

fake_news = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

true_news = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
fake_news.head(3)
# See how many fake news are there from different subjects

fake_news_by_subject = fake_news.groupby(by="subject").count()["title"]

print(fake_news_by_subject)

plt.figure(figsize=(10,3))

sns.countplot("subject", data=fake_news, palette="Blues")
fake_text_data = "".join(str(x) for x in fake_news["text"])

stop_words = set(nltk.corpus.stopwords.words("english"))

word_cloud_fake = WordCloud(stopwords=stop_words, width=2000, height=1000,\

                            max_font_size=160, min_font_size=30).generate(fake_text_data)

plt.figure(figsize=(12,6), facecolor="k")

plt.imshow(word_cloud_fake)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
#Lets do the wordcloud for true news

true_text_data = "".join(str(x) for x in true_news["text"])

word_cloud_fake = WordCloud(stopwords=stop_words, width=2000, height=1000,\

                            max_font_size=160, min_font_size=30).generate(true_text_data)

plt.figure(figsize=(12,6), facecolor="k")

plt.imshow(word_cloud_fake)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
del fake_text_data, true_text_data
#Label the true_or_not for concat

true_news["true"]=1

fake_news["true"]=0

df = pd.concat([fake_news, true_news])

# df.shape[0] # 44898 rows

df.head()
df["text"] = df["text"]+" "+df["title"]

#I am not sure whether giving different weights to these two vars show difference. 

df = df.filter(["text","true"], axis=1)

df.head()
import unicodedata

def remove_punct(text):

    text  = "".join([char for char in text if char not in string.punctuation])

    text = re.sub('[0-9]+', '', text)

    return text



def remove_stopwords(text):

    filtered_text = []

    for i in text.split():

        i = i.strip()

        if i.lower() not in stop_words:

            filtered_text.append(i)

    filtered_text = ' '.join(filtered_text)    

    return filtered_text



def normalize_accented_characters(text):

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf8')

    return text



def normalize_text(text):

    text = remove_punct(text)

    text = remove_stopwords(text)

    text = normalize_accented_characters(text)

    return text    
df['text']=df['text'].apply(normalize_text)
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):

    lemmas = []

    for word in text.split():

        lemmas.append(lemmatizer.lemmatize(word))

    return " ".join(lemmas)

df['text']=df['text'].apply(lemmatize_text)
x, y = df["text"].values, df["true"].values

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

X=[]

for article in x:

    sentence_list = []

    article = nltk.sent_tokenize(article)

    for sentence in article:

        sentence = sentence.lower()

        #The seqence here is very important since they are different data types.

        #Generally speaking, Word2Vec needs a "list of list" for input. 

        tokens = tokenizer.tokenize(sentence)

        sentence_list.extend([x.strip() for x in tokens])

    X.append(sentence_list)
import gensim

emb_dim = 100 #vector dimension

word2vec_model = gensim.models.Word2Vec(sentences=X, size=emb_dim, window=10, min_count=1)

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)

X = pad_sequences(X, maxlen=1000)
word_index = tokenizer.word_index #A dictionary with index and words

vocab_size = len(word_index) + 1

#Get the weight matrix for embedding layer

def get_weight(model,word_index):

    weight_matrix = np.zeros((vocab_size,emb_dim))

    for word, index in word_index.items():

        weight_matrix[index]=model[word]

    return weight_matrix

emb_vec = get_weight(word2vec_model,word_index)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
import tensorflow as tf

nn_model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, output_dim=emb_dim, weights = [emb_vec], input_length=1000,trainable=False),

    #REMEMBER TO PUT THE EMBEDDING VECTORS TO A LIST

    tf.keras.layers.LSTM(128, return_sequences=True),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(32,activation="relu"),

    tf.keras.layers.Dense(1,activation="sigmoid")

])

nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

nn_model.summary
history = nn_model.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test),batch_size=128)

classification_result = nn_model.evaluate(x_test,y_test)

classification_result