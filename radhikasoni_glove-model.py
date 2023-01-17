# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

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

import keras

from keras.models import Sequential

from keras.layers import Dense,Embedding,LSTM,Dropout

from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
true = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")

false = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
true.head()
false.head()
true['category'] = 1

false['category'] = 0
News = pd.concat([true,false]) 
sns.set_style("darkgrid")

sns.countplot(df.category)
News.head()
News.isna().sum() # Checking for nan Values
News.title.count()
News.subject.value_counts()
plt.figure(figsize = (12,8))

sns.set(style = "whitegrid",font_scale = 1.2)

chart = sns.countplot(x = "subject", hue = "category" , data = df)

chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
def strip_html(text):

    soup = BeautifulSoup(text, "html.parser")

    return soup.get_text()





def square_brackets(text):

    return re.sub('\[[^]]*\]', '', text)



def url(text):

    return re.sub(r'http\S+', '', text)



def stopwords(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            final_text.append(i.strip())

    return " ".join(final_text)



def preprocess(text):

    text = strip_html(text)

    text = square_brackets(text)

    text = url(text)

    text = stopwords(text)

    return text



News['text']=News['text'].apply(preprocess)
plt.figure(figsize = (20,20)) 

wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(News[News.category == 1].text))

plt.imshow(wc , interpolation = 'bilinear')

plt.axis('off')
plt.figure(figsize = (20,20)) # Text that is Fake

wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(News[News.category == 0].text))

plt.imshow(wc , interpolation = 'bilinear')

plt.axis('off')
def get_corpus(text):

    words = []

    for i in text:

        for j in i.split():

            words.append(j.strip())

    return words

corpus = get_corpus(News.text)

corpus[:5]
from collections import Counter

counter = Counter(corpus)

most_common = counter.most_common(10)

most_common = dict(most_common)

most_common
x_train,x_test,y_train,y_test = train_test_split(News.text,News.category,random_state = 0)
max_features = 10000

maxlen = 300
tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(x_train)

tokenized_train = tokenizer.texts_to_sequences(x_train)

x_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
tokenized_test = tokenizer.texts_to_sequences(x_test)

X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
EMBEDDING_FILE = '../input/glove-twitter/glove.twitter.27B.100d.txt'
def get_coefs(word, *arr): 

    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))



embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
batch_size = 256

epochs = 10

embed_size = 100
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
model = Sequential()



model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False))

 

model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))

model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))

model.add(Dense(units = 32 , activation = 'relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, batch_size = batch_size , validation_data = (X_test,y_test) , epochs = epochs , callbacks = [learning_rate_reduction])
print("Accuracy of the model on Training Data is - " , model.evaluate(x_train,y_train)[1]*100 , "%")

print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")
epochs = [i for i in range(10)]

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
pred = model.predict_classes(X_test)

pred[:5]
print(classification_report(y_test, pred, target_names = ['Fake','Not Fake']))
cm = confusion_matrix(y_test,pred)

cm
cm = pd.DataFrame(cm , index = ['Fake','Original'] , columns = ['Fake','Original'])
plt.figure(figsize = (10,10))

sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Fake','Original'] , yticklabels = ['Fake','Original'])

plt.xlabel("Predicted")

plt.ylabel("Actual")