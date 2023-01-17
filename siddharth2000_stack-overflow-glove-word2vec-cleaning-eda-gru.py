# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the libraries

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as pyoff
import plotly.graph_objs as go
import nltk
from collections import Counter

from plotly import graph_objs as go
from sklearn import preprocessing 
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import nltk
import gensim

import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout, Bidirectional, Conv2D
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import transformers
from tokenizers import BertWordPieceTokenizer
from keras.layers import LSTM,Dense,Bidirectional,Input
from keras.models import Model
import torch
import transformers
df = pd.read_csv('/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv')
df.head()
df['text'] = df['Title'] + df['Body']

df.drop(['Id', 'Title', 'Body', 'CreationDate', 'Tags'], axis=1, inplace=True)
df.head()
sns.countplot(df['Y'])
df.info()
# Data Cleaning
stop = set(stopwords.words('english'))

def cleaner(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", 'can not', phrase)
  
  # general
    phrase = re.sub(r"n\'t"," not", phrase)
    phrase = re.sub(r"\'re'"," are", phrase)
    phrase = re.sub(r"\'s"," is", phrase)
    phrase = re.sub(r"\'ll"," will", phrase)
    phrase = re.sub(r"\'d"," would", phrase)
    phrase = re.sub(r"\'t"," not", phrase)
    phrase = re.sub(r"\'ve"," have", phrase)
    phrase = re.sub(r"\'m"," am", phrase)
    
    return phrase

cleaned_title = []

for sentance in tqdm(df['text'].values):
    sentance = str(sentance)
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = cleaner(sentance)
    sentance = re.sub(r'[?|!|\'|"|#|+]', r'', sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stop)
    cleaned_title.append(sentance.strip())
    
df['text'] = cleaned_title
df.head()
df.head()
# Creating some basic EDA plots
# WordCloud for HighQuality Posts

plt.figure(figsize = (20,20)) # Text that is Not Sarcastic
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.Y == 'HQ'].text))
plt.imshow(wc , interpolation = 'bilinear')
# WordCloud for LowQuality Posts Closed

plt.figure(figsize = (20,20)) # Text that is Not Sarcastic
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.Y == 'LQ_CLOSE'].text))
plt.imshow(wc , interpolation = 'bilinear')
# WordCloud for LowQuality Posts Open

plt.figure(figsize = (20,20)) # Text that is Not Sarcastic
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.Y == 'LQ_EDIT'].text))
plt.imshow(wc , interpolation = 'bilinear')
# Continuing with some n-gram analysis

def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]
# Bi-grams for HQ posts

HQ_words = basic_clean(''.join(str(df[df.Y == 'HQ']['text'].tolist())))
bigram_HQ=(pd.Series(nltk.ngrams(HQ_words, 2)).value_counts())[:20]
bigram_HQ=pd.DataFrame(bigram_HQ)
bigram_HQ['idx']=bigram_HQ.index
bigram_HQ['idx'] = bigram_HQ.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+')',axis=1)
plot_data = [
    go.Bar(
        x=bigram_HQ['idx'],
        y=bigram_HQ[0],
        marker = dict(
            color = 'Blue'
        )
    )
]
plot_layout = go.Layout(
        title='Top 20 bi-grams from High Quality Posts',
        yaxis_title='Count',
        xaxis_title='bi-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
# Bi-grams for LQ-CLOSED posts

LQC_words = basic_clean(''.join(str(df[df.Y == 'LQ_CLOSE']['text'].tolist())))
bigram_LQC=(pd.Series(nltk.ngrams(LQC_words, 2)).value_counts())[:20]
bigram_LQC=pd.DataFrame(bigram_LQC)
bigram_LQC['idx']=bigram_LQC.index
bigram_LQC['idx'] = bigram_LQC.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+')',axis=1)
plot_data = [
    go.Bar(
        x=bigram_LQC['idx'],
        y=bigram_LQC[0],
        marker = dict(
            color = 'Green'
        )
    )
]
plot_layout = go.Layout(
        title='Top 20 bi-grams from Low Quality Posts Closed',
        yaxis_title='Count',
        xaxis_title='bi-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
# Bi-grams for LQ-OPEN posts

LQE_words = basic_clean(''.join(str(df[df.Y == 'LQ_EDIT']['text'].tolist())))
bigram_LQE=(pd.Series(nltk.ngrams(LQE_words, 2)).value_counts())[:20]
bigram_LQE=pd.DataFrame(bigram_LQE)
bigram_LQE['idx']=bigram_LQE.index
bigram_LQE['idx'] = bigram_LQE.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+')',axis=1)
plot_data = [
    go.Bar(
        x=bigram_LQE['idx'],
        y=bigram_LQE[0],
        marker = dict(
            color = 'Red'
        )
    )
]
plot_layout = go.Layout(
        title='Top 20 bi-grams from Low Quality Posts Open',
        yaxis_title='Count',
        xaxis_title='bi-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
# Word2Vec
# Model Building
# Step 1 - Tokenization
X = []
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in df['text'].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)
print ('Tokenization done...')   
# Model Building and Training
w2v_model = gensim.models.Word2Vec(sentences=X, size=150, window=5, min_count=2)
print ('Word2Vec model created')
# Making some naive observations

w2v_model.wv.most_similar(positive = 'python')
w2v_model.wv.most_similar(positive = 'java')
w2v_model.wv.most_similar(positive = 'bug')
w2v_model.wv.most_similar(positive = 'stack')
w2v_model.wv.similarity('java', 'kotlin')
w2v_model.wv.similarity('java', 'python')
w2v_model.wv.doesnt_match(['java', 'python', 'scala', 'kotlin'])
w2v_model.wv.doesnt_match(['java', 'python', 'pandas', 'numpy'])
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'class'. 
df['Y']= label_encoder.fit_transform(df['Y']) 
X = df['text']
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tokenizer = text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

tokenized_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(tokenized_train, maxlen=300)

tokenized_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(tokenized_test, maxlen=300)
print(len(tokenizer.word_index))
vocab_size = 10000 + 1
EMBEDDING_FILE = '../input/glovetwitter27b100dtxt/glove.twitter.27B.200d.txt'
embeddings_index = dict()
f = open(EMBEDDING_FILE)
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
embedding_matrix = zeros((vocab_size, 200))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
embedding_matrix.shape
# Training the Model. We will use a GRU model.

batch_size = 256
epochs = 10
embed_size = 200
maxlen = 300
max_features = 10001

#Defining Neural Network
model = Sequential()
#Non-trainable embeddidng layer
model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=True))
#LSTM
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.4 , dropout = 0.4))
#GRU
model.add(GRU(units=256 , return_sequences = False, dropout = 0.4))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(lr = 0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
history = model.fit(X_train, y_train, batch_size = batch_size , 
                    validation_data = (X_test, y_test) , epochs = 5, 
                    callbacks = [learning_rate_reduction])
print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")
epochs = [i for i in range(5)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')
ax[1].set_title('Training & Testing Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()
# We can see that the accuracy rises steadily for training while the growth is damped in case of testing
# The loss values for both training and testing are decreasing steadily.
# If we train for 15-20 epochs we can have a good convergent model
# We can also use different layers like :
# 1. Stacked GRU's
# 2. Bidirectional LSTM
# 3. Stacked LSTM's
# 4. Stacked Bidirectional LSTM's
