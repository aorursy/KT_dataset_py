# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pyplot

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel('/kaggle/input/william/William.xlsx')

df.head(10)
reviews_df = df[['sentence','Overall Sentiment\n(Positive/Negative/Neutral)', 'Services & Staff\n(0 = not mentioned, \n1 = mentioned)', 'Services & Staff Sentiment', 'Amenities \n(0 = not mentioned, \n1 =  mentioned)', 'Amenities Sentiment', 'Hotel Condition \n(0 = not mentioned, \n1 =  mentioned)', 'Hotel Condition Sentiment', 'Cleanliness\n(0 = not mentioned, \n1 =  mentioned)', 'Cleanliness Sentiment']]

reviews_df.head(10)
reviews_df['Overall Sentiment\n(Positive/Negative/Neutral)'].value_counts()
from matplotlib import pylab

from pylab import *

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

label = ['Pos', 'Neg', 'Neutral']

ax.pie(reviews_df['Overall Sentiment\n(Positive/Negative/Neutral)'].value_counts(), labels = label,autopct='%1.2f%%')

plt.title('Sentiment Distr')

plt.show
aspect_df = reviews_df[['Services & Staff\n(0 = not mentioned, \n1 = mentioned)',

 'Amenities \n(0 = not mentioned, \n1 =  mentioned)',

 'Hotel Condition \n(0 = not mentioned, \n1 =  mentioned)',

 'Cleanliness\n(0 = not mentioned, \n1 =  mentioned)']]



aspect_dist = reviews_df.apply(pd.Series.value_counts)

aspect_dist
columns = ['Services & Staff\n(0 = not mentioned, \n1 = mentioned)',

 'Amenities \n(0 = not mentioned, \n1 =  mentioned)',

 'Hotel Condition \n(0 = not mentioned, \n1 =  mentioned)',

 'Cleanliness\n(0 = not mentioned, \n1 =  mentioned)']



mentioned_list = []

for name in columns:

    mentioned_list.append(aspect_df.apply(pd.Series.value_counts)[name].tolist()[1])



print("Service & Staff: ",mentioned_list[0],"\nAmenities: ",mentioned_list[1],"\nHotel Condition: ",

      mentioned_list[2],"\nCleanliness: ",mentioned_list[3])
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

label = columns

ax.pie(mentioned_list, labels = label,autopct='%1.2f%%')

plt.title('Aspect Distribution')

fig, (ax1,ax2, ax3, ax4) = plt.subplots(1,4,figsize=(20,30))

label = ['Pos', 'Neg', 'Neutral']

ax1.pie(reviews_df['Services & Staff Sentiment'].value_counts(), labels = label,autopct='%1.2f%%')

ax1.set_title('Services & Staff Sentiment')





ax2.pie(reviews_df['Amenities Sentiment'].value_counts(), labels = label,autopct='%1.2f%%')

ax2.set_title('Amenities Sentiment')





ax3.pie(reviews_df['Hotel Condition Sentiment'].value_counts(), labels = label,autopct='%1.2f%%')

ax3.set_title('Hotel Condition Sentiment')





ax4.pie(reviews_df['Cleanliness Sentiment'].value_counts(), labels = label,autopct='%1.2f%%')

ax4.set_title('Cleanliness Sentiment')



plt.show
import re

reviews_df['sentence'] = reviews_df['sentence'].str.replace('[^\w\s]','')

reviews_df['sentence'] = reviews_df['sentence'].str.replace('\d+', '')

reviews_df['sentence'] = reviews_df['sentence'].str.lower()
drop = reviews_df[pd.isnull(reviews_df['Overall Sentiment\n(Positive/Negative/Neutral)'])].index

reviews_df.drop(drop , inplace=True)

reviews_df = reviews_df.reset_index(drop = True) 

reviews_df.head(10)
reviews_df['sentence'].replace('', np.nan, inplace=True)

drop = reviews_df[pd.isnull(reviews_df['sentence'])].index

reviews_df.drop(drop , inplace=True)

reviews_df = reviews_df.reset_index(drop = True) 

reviews_df.head(10)
import nltk

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
reviews_df['no_sw'] = reviews_df['sentence'][:].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
reviews_df
import nltk

from nltk import pos_tag, word_tokenize
# reviews_df['cleaned'] = [' '.join([Speller(i) for i in x.split()]) for x in reviews_df['no_sw']]
# wordfreq = {}

# for sentence in reviews_df['no_sw']:

#     tokens = word_tokenize(sentence)

#     sent_vec = []

#     for token in tokens:

#         if token not in wordfreq.keys():

#             wordfreq[token] = 1

#         else:

#             wordfreq[token] += 1
# import heapq

# num_features = 2500

# most_freq = heapq.nlargest(num_features, wordfreq, key=wordfreq.get)
# sentence_vectors = []

# for sentence in reviews_df['no_sw']:

#     sentence_tokens = nltk.word_tokenize(sentence)

#     sent_vec = []

#     for token in most_freq:

#         if token in sentence_tokens:

#             sent_vec.append(1)

#         else:

#             sent_vec.append(0)

#     sentence_vectors.append(sent_vec)
# sentence_vectors = np.asarray(sentence_vectors)

# sentence_vectors[0]
labels = []

for i in reviews_df['Overall Sentiment\n(Positive/Negative/Neutral)']:

    if i == "Positive":

        labels.append(0)

    elif i == "Neutral":

        labels.append(1)

    else:

        labels.append(2)
from sklearn.model_selection import train_test_split



sentences = reviews_df['no_sw'].values



sent_train, sent_test, y_train, y_test = train_test_split(sentences, labels, stratify=labels, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

vectorizer.fit(sent_train)



X_train = vectorizer.transform(sent_train)

X_test  = vectorizer.transform(sent_test)

X_train
from sklearn.ensemble import RandomForestClassifier



text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)

text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

print(accuracy_score(y_test, predictions))
import keras

from keras.models import Sequential

from keras import layers

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping
y_train = keras.utils.to_categorical(y_train, 3)

y_test = keras.utils.to_categorical(y_test, 3)
input_dim = X_train.shape[1] # Number of features



model = Sequential()

model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))

model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=10,

                    verbose=True,

                    validation_data=(X_test, y_test),

                    batch_size=5)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))
y_pred = model.predict(X_test, batch_size=5, verbose=1)

y_pred_bool = np.argmax(y_pred, axis=1)

y_labels = np.argmax(y_test, axis=1)



print(classification_report(y_labels, y_pred_bool))
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(sent_train)



X_train = tokenizer.texts_to_sequences(sent_train)

X_test = tokenizer.texts_to_sequences(sent_test)



vocab_size = len(tokenizer.word_index) + 1  
from keras.preprocessing.sequence import pad_sequences



maxlen = 100



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



print(X_train[0, :])
embedding_dim = 50



model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(10, activation='relu'))

model.add(Dense(3, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=5,

                    verbose=True,

                    validation_data=(X_test, y_test),

                    batch_size=32,

                    workers=4,

                    use_multiprocessing=True,

                    max_queue_size=100)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))



y_pred = model.predict(X_test, batch_size=5, verbose=1)

y_pred_bool = np.argmax(y_pred, axis=1)

y_labels = np.argmax(y_test, axis=1)



print(classification_report(y_labels, y_pred_bool))
embedding_dim = 128



model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))

model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=4))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(3, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=20,

                    verbose=True,

                    validation_data=(X_test, y_test),

                    batch_size=32,

                    workers=4,

                    use_multiprocessing=True,

                    max_queue_size=100)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))



y_pred = model.predict(X_test, batch_size=5, verbose=1)

y_pred_bool = np.argmax(y_pred, axis=1)

y_labels = np.argmax(y_test, axis=1)



print(classification_report(y_labels, y_pred_bool))