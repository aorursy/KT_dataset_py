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
from __future__ import unicode_literals

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

import re

import string

import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer,WordNetLemmatizer

from string import punctuation

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix

import tensorflow as tf

from tensorflow import keras

from keras.callbacks import EarlyStopping,ReduceLROnPlateau

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense,Embedding,Bidirectional,Dropout,SpatialDropout1D,GlobalMaxPool1D,LSTM

from keras.models import Sequential

from keras.optimizers import Adam

from keras import regularizers
data = pd.read_csv("../input/nlp-getting-started/train.csv")

data = data.drop(['id','location'],axis=1)

data.head()
#Stopwords

stop = (stopwords.words('english'))

punctuation = list(string.punctuation)

for i in punctuation:

    stop.append(i)
#Cleaning Data



stemmer = SnowballStemmer('english',ignore_stopwords=True)

lemmatizer = WordNetLemmatizer()



def remove_stopwords(text):

    sentences = []

    for word in text.split():

        if word.lower().strip() not in stop and len(word)>3:

            word = lemmatizer.lemmatize(word)

            sentences.append(word.lower().strip())

    return " ".join(sentences)



def remove_punctuations(text):

    punc = re.compile(r'[%s]'%string.punctuation)

    return punc.sub(r'',text)

                      

def remove_urls(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_squarebrackets(text):

    square = re.compile(r'\[.*?\]')

    return square.sub(r'',text)



def remove_tags(text):

    tags = re.compile(r'<.*?>')

    return tags.sub(r'',text)

    

def remove_numbers(text):

    num = re.compile(r'\w*\d\w*')

    return num.sub(r'',text)

    

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'',text)



def clean(text):

    text = remove_punctuations(text)

    text = remove_urls(text)

    text = remove_tags(text)

    text = remove_squarebrackets(text)

    text = remove_emoji(text)

    text = remove_numbers(text)

    text = remove_stopwords(text)

    return text

data['text'] = data['text'].apply(lambda x:clean(x))
#Count of tweets

sns.countplot(data['target'],palette='RdBu_r')

plt.title("Non-Disaster vs Disaster Tweets")

print("No of Non-Disaster Tweets: " ,data['target'].value_counts()[0])

print("No of Disaster Tweets: " ,data['target'].value_counts()[1])
#Wordcloud

fig,ax = plt.subplots(figsize=(12,16))

plt.axis('off')



plt.subplot(2,1,1)

text = " ".join(data[data['target']==0]['text'])

wordcloud = WordCloud(max_font_size = None,background_color='white',width=1200,height=1000).generate(text)

plt.title("WordCloud for Non-Disaster Tweet")

plt.axis('off')

plt.imshow(wordcloud)



plt.subplot(2,1,2)

text = " ".join(data[data['target']==1]['text'])

wordcloud = WordCloud(max_font_size = None,background_color='white',width=1200,height=1000).generate(text)

plt.title("WordCloud for Disaster Tweet")

plt.axis('off')

plt.imshow(wordcloud)
#No Of Characters In A Tweet

fig,ax = plt.subplots(figsize=(12,6))

fig.suptitle("NO OF CHARACTERS IN A TWEET")



plt.subplot(1,2,1)

plt.title("Non-Disaster Tweets")

words = data[data['target']==0]['text'].str.len()

sns.distplot(words,kde=True)



plt.subplot(1,2,2)

plt.title("Disaster Tweets")

words = data[data['target']==1]['text'].str.len()

sns.distplot(words,kde=True)
#Average Word Length In A Tweet

fig,ax = plt.subplots(figsize=(12,6))

fig.suptitle("AVERAGE WORD LENGTH IN A Tweet")



plt.subplot(1,2,1)

plt.title("Non-Disaster Tweets")

word_length = data[data['target']==0]['text'].str.split().apply(lambda x:[len(i) for i in x])

sns.distplot(word_length.map(lambda x:np.mean(x)),kde=True)



plt.subplot(1,2,2)

plt.title("Disaster Tweets")

word_length = data[data['target']==1]['text'].str.split().apply(lambda x:[len(i) for i in x])

sns.distplot(word_length.map(lambda x:np.mean(x)),kde=True)
#Keywords

plt.figure(figsize=(14,6))

sns.barplot(x=data['keyword'].value_counts()[:20],y=data['keyword'].value_counts()[:20].index,palette='RdBu_r')

plt.xlabel("Count")

plt.ylabel("Keyword")
#Split the data

x_train,x_test,y_train,y_test = train_test_split(data['text'],data['target'],test_size=0.2,random_state=0)
#Tokenizer

vocab_size=10000

embedding_dim=200

max_length=100

trunc_type="post"

pad_type="post"

oov_tok="<OOV>"



tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(list(x_train)+list(x_test))

word_index = tokenizer.word_index



train_sequences = tokenizer.texts_to_sequences(x_train)

train_padded = pad_sequences(train_sequences,maxlen=max_length,truncating=trunc_type,padding=pad_type)



test_sequences = tokenizer.texts_to_sequences(x_test)

test_padded = pad_sequences(test_sequences,maxlen=max_length,truncating=trunc_type,padding=pad_type)
print("words =", len(word_index))

print("train =",len(train_padded))

print("test =",len(test_padded))
#GloVe Embeddings

embeddings_index={}

with open("../input/glove6b/glove.6B.200d.txt",'r',encoding='utf-8') as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

        

embeddings_matrix = np.zeros((len(word_index)+1, embedding_dim))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embeddings_matrix[i] = embedding_vector
#Neural Network

model = Sequential()



model.add(Embedding(len(word_index)+1,embedding_dim,input_length=max_length,weights=[embeddings_matrix]))

model.add(SpatialDropout1D(0.5))

model.add(Bidirectional(LSTM(128,recurrent_dropout=0.5,dropout=0.5,return_sequences=True)))

model.add(GlobalMaxPool1D())

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(32,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(16,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

model.summary()
#Callbacks

earlystop = EarlyStopping(monitor='val_loss',patience=3,verbose=1)

learning_reduce = ReduceLROnPlateau(patience=2,monitor="val_acc",verbose=1,min_lr=0.00001,factor=0.5,cooldown=1)

callbacks = [earlystop,learning_reduce]
epoch=10

history = model.fit(train_padded,y_train,epochs=epoch,validation_data=(test_padded,y_test),callbacks=callbacks)
#Plot

def plot_graphs(history, string):

    plt.plot(history.history[string])

    plt.plot(history.history["val_"+string])

    plt.xlabel("Epochs")

    plt.ylabel(string)

    plt.legend([string,"val_"+string])

    plt.show()

plot_graphs(history,'acc')

plot_graphs(history,'loss')

y_pred = model.predict_classes(test_padded)

print("Accuracy: ",accuracy_score(y_test,y_pred).round(3))

print("Precision: ",precision_score(y_test,y_pred).round(3))

print("Recall: ",recall_score(y_test,y_pred).round(3))

print("F1-Score: ",f1_score(y_test,y_pred).round(3))
#Confusion Matrix

cm = confusion_matrix(y_test,y_pred)

cm = pd.DataFrame(cm , index = ['Non-Disaster','Disaster'] , columns = ['Non-Disaster','Disaster'])

sns.heatmap(cm,cmap= "Blues",annot=True,fmt='')

plt.title("Confusion Matrix")
#Test Data

test = pd.read_csv("../input/nlp-getting-started/test.csv")

submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test['text'] = test['text'].apply(lambda x:clean(x))



testing_sequences = tokenizer.texts_to_sequences(test['text'])

testing_padded = pad_sequences(testing_sequences,maxlen=max_length,truncating=trunc_type,padding=pad_type)



predictions = model.predict(testing_padded)



submission['target'] = (predictions>0.5).astype(int)
submission.head()
submission.to_csv("submission.csv", index=False, header=True)