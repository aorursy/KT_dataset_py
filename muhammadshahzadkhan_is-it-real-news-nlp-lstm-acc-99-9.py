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
import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

import nltk

import re

from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize, sent_tokenize

import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

#Now keras libraries

from tensorflow.keras.preprocessing.text import one_hot, Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional

from tensorflow.keras.models import Model
fake_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

true_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake_df
true_df
# Check null values True

true_df.isnull().sum()
# Check null values of False

fake_df.isnull().sum()
# If we also want to know the memory usage

fake_df.info()
true_df.info()
# Now lets add an target column for indicating real and fake news

true_df['isfake'] = 0

fake_df['isfake'] = 1
true_df.head()
fake_df.head()
#Now lets concatenate both of them

df = pd.concat([true_df, fake_df]).reset_index(drop = True)

df
# Drop the date column

df.drop(columns= ['date'], axis =1, inplace= True)

df
# Combine "title" and "text" in single column

df['original'] = df['title'] + ' ' + df['text']

df.head()
df['original'][0]
# Add some additional stopwors in nltk stopwords package

stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
stop_words
# To remove stopwords and words with length less than 3

def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) >3 and token not in stop_words:

            result.append(token)

    return result
# Now lets apply the defined function of "original" text in our datadrame

df['clean'] = df['original'].apply(preprocess)
df
# original news

df['original'][0]
# clean news after removing stopwords

print(df['clean'][0])
# Total number of words in dataset

list_words = []

for i in df.clean:

    for j in i:

        list_words.append(j)

print ('Total number of words are: {}'.format(len(list_words)))
#list_words
#total unique words

total_unique_words = len(list(set(list_words)))

total_unique_words
# Now lets convert words in "clean" column to a string

df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

df
df['clean_joined'][1]
print (df['clean'][1])
print (df['original'][1])
# First, lets plot the num of samples in subject 

plt.figure(figsize=(8,8))

sns.countplot(y = 'subject', data = df)
# fake vs true news

plt.figure(figsize=(8,8))

sns.countplot(y = 'isfake', data = df)
# Now lets plot word cloud for Real news. This will show the most common words when the news is Fake.

plt.figure(figsize=(20,20))

wc = WordCloud(max_words= 2000, width= 1600, height= 800, stopwords= stop_words).generate(" ".join(df[df.isfake == 1].clean_joined))

plt.imshow(wc, interpolation= 'bilinear')
# Now lets plot word cloud for Real news. This will show the most common words when the news is Fake.

# As most new are Political, so words are clearly visible in these plots.

plt.figure(figsize=(20,20))

wc = WordCloud(max_words= 2000, width= 1600, height= 800, stopwords= stop_words).generate(" ".join(df[df.isfake == 1].clean_joined))

plt.imshow(wc, interpolation= 'bilinear')
# Now maximum length is required to create word embeddings

maxlen = -1

for news in df.clean_joined:

    tokens = nltk.word_tokenize(news) #converts text to tokens (words)

    if (maxlen < len(tokens)):

        maxlen = len(tokens)

print ("The maximum number of words in any news is =", maxlen)
#Now lets visualize the distribution of number of words in a text, using a very interactive tool "Plotly"

import plotly.express as px

fig = px.histogram(x = [len(nltk.word_tokenize(x)) for x in df.clean_joined], nbins = 100)

fig.show()
# Split data into Train and Test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2, random_state = 42)
from nltk import word_tokenize
# Lets create tokenizer to tokenize the words and conert them into a sequence

tokenizer = Tokenizer(num_words= total_unique_words)

tokenizer.fit_on_texts(x_train) #It creates vocabulary index ("word_index") based on word frequency

train_sequences = tokenizer.texts_to_sequences(x_train) # Replace each word in text with corresponding integer value from "word_index"

test_sequences = tokenizer.texts_to_sequences(x_test)
len(train_sequences)
len(test_sequences)
print ("The encoding for news\n", df.clean_joined[0], "\n is\n :", train_sequences[0])
pad_train = pad_sequences(train_sequences, maxlen = 4405, padding = 'post', truncating= 'post')

pad_test = pad_sequences(test_sequences, maxlen=4405, padding = 'post', truncating= 'post')
# Lets visualize the padding sequence for 1st two samples

for i, news in enumerate(pad_train[:2]):

    print("The padded encoding for news", i+1, " is : ", news)
# Lets visualize the padding sequence for 1st two samples

for i, news in enumerate(pad_test[:2]):

    print("The padded encoding for news", i+1, " is : ", news)
# Now lets build the model

model = Sequential() 



model.add(Embedding(total_unique_words, output_dim = 128)) #Embedding Layer



model.add(Bidirectional(LSTM(128))) #Bi-directional LSTM



#Dense layer

model.add(Dense(128, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid')) # binary classification (0\1)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['acc'])

model.summary()
y_train = np.asarray(y_train)
# train the model

model.fit(pad_train, y_train, batch_size= 64, validation_split = 0.1, epochs= 2)
pred = model.predict(pad_test) #prediction
# Lets set the threshold 0.5, i.e if pred >0.5, it implies the news is fake and vice versa.

prediction = []

for i in range (len(pred)):

    if pred[i].item() > 0.5:

        prediction.append(1)

    else:

        prediction.append(0)
#accuracy

from sklearn.metrics import accuracy_score



accuracy = accuracy_score(list(y_test), prediction)



print ("The model accuracy is :", accuracy)
#confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(list(y_test), prediction)

plt.figure(figsize=(10,10))

sns.heatmap(cm, annot = True)