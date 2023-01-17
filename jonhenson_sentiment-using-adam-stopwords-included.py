
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from wordcloud import WordCloud

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout, SpatialDropout1D

import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

df = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',encoding = 'latin',header=None)
df.columns = ['sentiment', 'x1', 'x2', 'x3', 'x4', 'text']
df = df.drop(['x1', 'x2', 'x3', 'x4'], axis=1)

print(df)
#function to change labels to Positive or Negative
Convert_Sentiment = {0: "Negative", 4: "Positive"}
def Sentiment_change(lcol):
  return Convert_Sentiment[lcol]
df.sentiment = df.sentiment.apply(lambda x: Sentiment_change(x))
#lets look at our distribution
val_count = df.sentiment.value_counts()
print(val_count)
## Clean data. Typically stop words would be used. This study includes all words, excluding stop words in a 280 char tweet is too restrictive.
## Remove words with no meaning, mentions and urls. 
cleaning = "amp\S+|quot\S+|@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]"
def preprocess(text):
  text = re.sub(cleaning, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    tokens.append(token)
  return " ".join(tokens)
df.text = df.text.apply(lambda x: preprocess(x))
plt.figure(figsize=(20, 20))
cloud = WordCloud(max_words=1000, width=1000, height=800)
cloud.generate(" ".join(df[df.sentiment == 'Positive'].text))
plt.imshow(cloud, interpolation='bilinear')
plt.show()
plt.figure(figsize=(20, 20))
cloud = WordCloud(max_words=1000, width=1000, height=800)
cloud.generate(" ".join(df[df.sentiment == 'Negative'].text))
plt.imshow(cloud, interpolation='bilinear')
plt.show()
#setup for training and testing splits. 
Training_percent = 0.8
MAX_SEQUENCE_LENGTH = 30
train_data, test_data = train_test_split(df, test_size=1-Training_percent, random_state=7) 
tokens = Tokenizer()
tokens.fit_on_texts(train_data.text)
word_index = tokens.word_index
#pesky +1 to account for 0 starting.
vocab_size = len(tokens.word_index) + 1
print("# of words :", vocab_size)
x_train = pad_sequences(tokens.texts_to_sequences(train_data.text),maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(tokens.texts_to_sequences(test_data.text), maxlen=MAX_SEQUENCE_LENGTH)

labels = train_data.sentiment.unique().tolist()

encoder = LabelEncoder()
encoder.fit(train_data.sentiment.to_list())
#Create y-train
y_train = encoder.transform(train_data.sentiment.to_list())
y_test = encoder.transform(test_data.sentiment.to_list())
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print("Y Train shape: ",y_train.shape)
print("Y Test shape: ",y_test.shape)
GLOVE_EMB = '../input/glove6b300dtxt/glove.6B.300d.txt'
EMBEDDING_DIM = 300
LR = .005
BATCH_SIZE = 2048
EPOCHS = 15
#start embedding, open glove and resolve total vectors
embed_index = {}
f = open(GLOVE_EMB, encoding="utf-8")
for line in f:
  values = line.split()
  word = values[0]
  vectors = np.asarray(values[1:], dtype='float32')
  embed_index[word] = vectors
f.close()
print('Word vectors: ',len(embed_index))
#create the matrix and fill with 0s, size of total vocab words and the embeded dimentions from the Glove file used. 
emb_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

#get each word, get corresponding glove value, write if value exists to the embedding matrix
for word, i in word_index.items():
  embedding_vector = embed_index.get(word)
  if embedding_vector is not None:
    emb_matrix[i] = embedding_vector
#create the embedding layer with given dimensions, vectors. Trainable false is used, if using trainable = true the model will over fit significantly
embedding_layer = tf.keras.layers.Embedding(vocab_size,EMBEDDING_DIM,weights=[emb_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False)
#sequence creation. 
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)
x = SpatialDropout1D(0.2)(embedding_sequences)
x = Conv1D(64, 5, activation='relu')(x)
x = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.1))(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(sequence_input, outputs)
model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())
#Here we go
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))
s, (at, al) = plt.subplots(2, 1)
at.plot(history.history['accuracy'], c='b')
at.plot(history.history['val_accuracy'], c='r')
at.set_title('model accuracy')
at.set_ylabel('accuracy')
at.set_xlabel('epoch')
at.legend(['LSTM_train', 'LSTM_val'], loc='upper left')

al.plot(history.history['loss'], c='m')
al.plot(history.history['val_loss'], c='c')
al.set_title('model loss')
al.set_ylabel('loss')
al.set_xlabel('epoch')
al.legend(['train', 'val'], loc='upper left')
plt.show()
def decode_sentiment(score):
    return "Positive" if score > 0.5 else "Negative"


scores = model.predict(x_test, verbose=1, batch_size=10000)
y_pred_1d = [decode_sentiment(score) for score in scores]
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=13)
    plt.yticks(tick_marks, classes, fontsize=13)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    

cnf_matrix = confusion_matrix(test_data.sentiment.to_list(), y_pred_1d)
plt.figure(figsize=(6, 6))
plot_confusion_matrix(cnf_matrix, classes=test_data.sentiment.unique(), title="Confusion matrix")
plt.show()

print(classification_report(list(test_data.sentiment), y_pred_1d))

model.save_weights("model.h5")