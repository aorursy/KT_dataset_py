

import numpy as np 
import pandas as pd

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import nltk
from nltk.tokenize import word_tokenize
import string
import pandas as pd
from bs4 import BeautifulSoup
nltk.download('all')
dataset = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
# First five rows of the dataset

dataset.head()
print("Shape of the dataset = ", dataset.shape)
print("# of rows = ", dataset.shape[0])
print("# of columns = ", dataset.shape[1])
# Let's check the no. of positive and negative reviews in the dataset

dataset.sentiment.value_counts()
# Now we are going to label the sentiment column 
# For positive - 1
# For negetive - 0

dataset['sentiment'] = pd.get_dummies(dataset['sentiment'],drop_first = True)
dataset.head()
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import wordnet

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
wn = nltk.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

def text_preprocess(text):
    soup = BeautifulSoup(text, 'html.parser').text
    no_punctuation = "".join([c for c in soup if c not in string.punctuation]).lower()
    tokens = pos_tag(word_tokenize(no_punctuation))
    clean_text = [word for word in tokens if word not in stopwords]
    lemma = [wn.lemmatize(word[0],get_wordnet_pos(word[1])) for word in clean_text]
    lemma = ' '.join(lemma)
    return lemma
dataset['clean_review'] = dataset['review'].apply(lambda x: text_preprocess(x))
dataset.head()
from collections import Counter

def counter_word(text):
  count = Counter()
  for row in text.values:
    for word in row.split():
      count[word] += 1
  return count

text = dataset.clean_review
counter = counter_word(text)
print(f"There are {len(counter)} unique words in the clean_review column")
num_words = len(counter)
max_length = 100
lst = []
for i in range(len(dataset)):
    lst.append(len(dataset.clean_review[i]))
        
counter
len(dataset.clean_review[1])
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(dataset.clean_review)
word_index = tokenizer.word_index
dataset_sequences = tokenizer.texts_to_sequences(dataset.clean_review)

from keras.preprocessing.sequence import pad_sequences

dataset_padded = pad_sequences(dataset_sequences, maxlen=max_length,padding='post',truncating='post')
dataset_padded.shape
y = dataset['sentiment']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset_padded,y,test_size=0.25,random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from keras.models import Sequential
from keras.layers import Embedding,Dropout,Dense,LSTM
from keras.initializers import Constant
from keras.optimizers import Adam

model = Sequential()

model.add(Embedding(num_words,50, input_length = max_length))
model.add(LSTM(64,dropout=0.5))
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

optimizer = Adam(learning_rate=3e-4)

model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train,batch_size=128,epochs=10,validation_data=(X_test,y_test))
score = model.evaluate(X_test,y_test)
score
dataset[dataset['sentiment']==0]
text=dataset.review[49996]
testing_text = text
clean_testing_text = text_preprocess(testing_text)
sentence = np.array([clean_testing_text])
sentence_sequence = tokenizer.texts_to_sequences(sentence)
sentence_padded =  pad_sequences(sentence_sequence,maxlen=max_length)

pred = model.predict(sentence_padded)
if pred >= 0.5:
    print('Positive')
else:
    print('Negative')
