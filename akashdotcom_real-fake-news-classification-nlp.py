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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from wordcloud import WordCloud, STOPWORDS

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize, sent_tokenize #(word tokenize, sentence tokenize)

from bs4 import BeautifulSoup

import re, string, unicodedata

from keras.preprocessing import text, sequence

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import train_test_split

from string import punctuation

from nltk import pos_tag

from nltk.corpus import wordnet

import keras

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, Dropout

from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
df = pd.read_csv('../input/fake-news/news.csv')
df.head()
#replacing FAKE = 0, REAL = 1

df['label'] = df['label'].replace('FAKE', 0)

df['label'] = df['label'].replace('REAL', 1)
#Removing unwanted columns

df.drop('Unnamed: 0', axis =1 , inplace = True)
df
#distribution of fake and real dataset

sns.set_style("darkgrid")

sns.countplot(df['label'])
#checking na values

df.isna().sum()
stop_words = set(stopwords.words('english')) #set of all stopwords

punctuation = list(string.punctuation) #all punctuation

#adding everything into one set

stop_words.update(punctuation)
#Data Cleaning - Part-1



def strip_html(text):

    soup = BeautifulSoup(text, 'html.parser')

    return soup.get_text()



def square_brackets(text):

    return re.sub('\[[^]]*\]', '', text)



def url_extract(text):

    return re.sub(r'http\S+', '', text)

#Data Cleaning - Part-2

def stopwords(text):

    final_text = []

    for i in text.split():

        #checking in stopwords and also lowering the text

        if i.strip().lower() not in stop_words:

            final_text.append(i.strip())

    return " ".join(final_text)



#finally getting all outputs in preprocessing the text using above functions

def preprocess(text):

    text = strip_html(text)

    text = square_brackets(text)

    text = url_extract(text)

    text = stopwords(text)

    return text
df['text'] = df['text'].apply(preprocess)
#creating wordclouds - Real News texts

plt.figure(figsize = (20,20)) 



wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , 

               stopwords = STOPWORDS).generate(" ".join(df[df.label == 1].text))

plt.imshow(wc , interpolation = 'bilinear')

plt.axis('off')
#Fake news text: Wordcloud

plt.figure(figsize = (20,20)) 



wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , 

               stopwords = STOPWORDS).generate(" ".join(df[df.label == 0].text))

plt.imshow(wc , interpolation = 'bilinear')

plt.axis('off')
#crating vocab for the news

def get_corpus(text):

    words = []

    for i in text:

        for j in i.split():

            words.append(j.strip())

    return words



corpus = get_corpus(df.text)
corpus[:5]
#getting count for each word now using Counter

from collections import Counter

counter = Counter(corpus)

most_common_words = counter.most_common(10) #prining most common 10 words

most_common_words = dict(most_common_words)

most_common_words
#train_test split

X_train, X_test, y_train, y_test= train_test_split(df.text, df.label, random_state = 420)
X_test.shape[0], X_train.shape[0] #test train data rows
maxfeatures = 10000

maxlength = 400
#tokenize

tokenizer = text.Tokenizer(num_words=maxfeatures)

tokenizer.fit_on_texts(X_train)

tokenized_train = tokenizer.texts_to_sequences(X_train)

X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlength)
X_train
tokenized_test = tokenizer.texts_to_sequences(X_test)

X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlength)
#calling GLOVE model

EMBEDDING_FILE = '../input/glove-twitter/glove.twitter.27B.200d.txt'
def get_coeff(word, *arr):

    return word, np.asarray(arr, dtype = 'float32')



embeddings_index = dict(get_coeff(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
all_embedd = np.stack(embeddings_index.values())

embedd_mean, embedd_std = all_embedd.mean(), all_embedd.std()

embedd_size = all_embedd.shape[1]



word_index = tokenizer.word_index

nb_words = min(maxfeatures, len(word_index))



#creating a matrix

embedding_matrix = np.random.normal(embedd_mean, embedd_std, (nb_words, embedd_size))
for word, i in word_index.items():

    if i>= maxfeatures:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
batch_size = 256

epochs = 10

embedd_size = 200
learning_rate = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 2,

                                 verbose = 1, factor = 0.5, min_lr=0.0001)
#creating a model

model = Sequential()



model.add(Embedding(maxfeatures, output_dim = embedd_size,

                    weights = [embedding_matrix], input_length = maxlength, trainable = False))



model.add(LSTM(units = 128, return_sequences = True, 

               recurrent_dropout = 0.25, dropout = 0.25))



model.add(LSTM(units = 64, recurrent_dropout = 0.1, dropout = 0.1))

model.add(Dense(32, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(optimizer= keras.optimizers.Adam(lr = 0.01), 

             loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
history = model.fit(X_train, y_train, batch_size=batch_size, 

                    validation_data=(X_test, y_test), 

                    epochs=epochs, callbacks= [learning_rate])
#accuracy on train and test data

print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,y_train)[1]*100 , "%")



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
#making predictions now

predictions = model.predict_classes(X_test)

predictions[:10]
#classification report

print(classification_report(y_test, predictions, target_names = ['FAKE','REAL']))
#confusion matrix

plt.figure(figsize = (10,10))

cm = confusion_matrix(y_test,predictions)

sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['FAKE','REAL'] , yticklabels = ['FAKE','REAL'])

plt.xlabel("Predicted")

plt.ylabel("Actual")