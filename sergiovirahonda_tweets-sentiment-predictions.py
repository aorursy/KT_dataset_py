import re

import matplotlib.pyplot as plt

import string

from nltk.corpus import stopwords

import nltk

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

from nltk.tokenize.treebank import TreebankWordDetokenizer

from collections import Counter

from wordcloud import WordCloud

from nltk.corpus import stopwords

import nltk

from gensim.utils import simple_preprocess

from nltk.corpus import stopwords

import gensim

from sklearn.model_selection import train_test_split

import spacy

import pickle

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt 

import tensorflow as tf

import keras

import numpy as np

import pandas as pd

print('Done')
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
train.head(15)
#Let's get the dataset lenght

len(train)
#Is there any other different value than neutral, negative and positive?

train['sentiment'].unique()
#How's distributed the dataset? Is it biased?

train.groupby('sentiment').nunique()
#Let's keep only the columns that we're going to use

train = train[['selected_text','sentiment']]

train.head()
#Is there any null value?

train["selected_text"].isnull().sum()
#Let's fill the only null value.

train["selected_text"].fillna("No content", inplace = True)
def depure_data(data):

    

    #Removing URLs with a regular expression

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    data = url_pattern.sub(r'', data)



    # Remove Emails

    data = re.sub('\S*@\S*\s?', '', data)



    # Remove new line characters

    data = re.sub('\s+', ' ', data)



    # Remove distracting single quotes

    data = re.sub("\'", "", data)

        

    return data
temp = []

#Splitting pd.Series to list

data_to_list = train['selected_text'].values.tolist()

for i in range(len(data_to_list)):

    temp.append(depure_data(data_to_list[i]))

list(temp[:5])
def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

        



data_words = list(sent_to_words(temp))



print(data_words[:10])
len(data_words)
def detokenize(text):

    return TreebankWordDetokenizer().detokenize(text)
data = []

for i in range(len(data_words)):

    data.append(detokenize(data_words[i]))

print(data[:5])
data = np.array(data)
labels = np.array(train['sentiment'])

y = []

for i in range(len(labels)):

    if labels[i] == 'neutral':

        y.append(0)

    if labels[i] == 'negative':

        y.append(1)

    if labels[i] == 'positive':

        y.append(2)

y = np.array(y)

labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")

del y
len(labels)
from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras import regularizers

from keras import backend as K

from keras.callbacks import ModelCheckpoint

max_words = 5000

max_len = 200



tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(data)

sequences = tokenizer.texts_to_sequences(data)

tweets = pad_sequences(sequences, maxlen=max_len)

print(tweets)
print(labels)
#Splitting the data

X_train, X_test, y_train, y_test = train_test_split(tweets,labels, random_state=0)

print (len(X_train),len(X_test),len(y_train),len(y_test))
#model0 = Sequential()

#model0.add(layers.Embedding(max_words, 15))

#model0.add(layers.SimpleRNN(15))

#model0.add(layers.Dense(3,activation='softmax'))





#model0.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

#Implementing model checkpoins to save the best metric and do not lose it on training.

#checkpoint0 = ModelCheckpoint("best_model0.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)

#history = model0.fit(X_train, y_train, epochs=5,validation_data=(X_test, y_test),callbacks=[checkpoint0])
model1 = Sequential()

model1.add(layers.Embedding(max_words, 20))

model1.add(layers.LSTM(15,dropout=0.5))

model1.add(layers.Dense(3,activation='softmax'))





model1.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

#Implementing model checkpoins to save the best metric and do not lose it on training.

checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)

history = model1.fit(X_train, y_train, epochs=70,validation_data=(X_test, y_test),callbacks=[checkpoint1])
model2 = Sequential()

model2.add(layers.Embedding(max_words, 40, input_length=max_len))

model2.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))

model2.add(layers.Dense(3,activation='softmax'))

model2.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

#Implementing model checkpoins to save the best metric and do not lose it on training.

checkpoint2 = ModelCheckpoint("best_model2.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)

history = model2.fit(X_train, y_train, epochs=70,validation_data=(X_test, y_test),callbacks=[checkpoint2])
from keras import regularizers

model3 = Sequential()

model3.add(layers.Embedding(max_words, 40, input_length=max_len))

model3.add(layers.Conv1D(20, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),bias_regularizer=regularizers.l2(2e-3)))

model3.add(layers.MaxPooling1D(5))

model3.add(layers.Conv1D(20, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),bias_regularizer=regularizers.l2(2e-3)))

model3.add(layers.GlobalMaxPooling1D())

model3.add(layers.Dense(3,activation='softmax'))

model3.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

checkpoint3 = ModelCheckpoint("best_model3.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)

history = model3.fit(X_train, y_train, epochs=70,validation_data=(X_test, y_test),callbacks=[checkpoint3])
#Let's load the best model obtained during training

best_model = keras.models.load_model("best_model2.hdf5")
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)

print('Model accuracy: ',test_acc)
predictions = best_model.predict(X_test)
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test.argmax(axis=1), np.around(predictions, decimals=0).argmax(axis=1))
import seaborn as sns

conf_matrix = pd.DataFrame(matrix, index = ['Neutral','Negative','Positive'],columns = ['Neutral','Negative','Positive'])

#Normalizing

conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (15,15))

sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})
sentiment = ['Neutral','Negative','Positive']
sequence = tokenizer.texts_to_sequences(['this experience has been the worst , want my money back'])

test = pad_sequences(sequence, maxlen=max_len)

sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]
sequence = tokenizer.texts_to_sequences(['this data science article is the best ever'])

test = pad_sequences(sequence, maxlen=max_len)

sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]
sequence = tokenizer.texts_to_sequences(['i hate youtube ads, they are annoying'])

test = pad_sequences(sequence, maxlen=max_len)

sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]
sequence = tokenizer.texts_to_sequences(['i really loved how the technician helped me with the issue that i had'])

test = pad_sequences(sequence, maxlen=max_len)

sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]
#Saving weights and tokenizer so we can reduce training time on SageMaker



# serialize model to JSON

model_json = best_model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

best_model.save_weights("model-weights.h5")

print("Model saved")



# saving tokenizer

with open('tokenizer.pickle', 'wb') as handle:

    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Tokenizer saved')