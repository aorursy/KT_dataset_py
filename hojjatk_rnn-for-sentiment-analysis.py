#

# Import packages

#

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Scikit-learn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer



# Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM

from keras import utils

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



# nltk

import nltk

from nltk.corpus import stopwords

from  nltk.stem import SnowballStemmer



# Word2vec

import gensim



# Utility

import re

import numpy as np

import os

from collections import Counter

import logging

import time

import pickle

import itertools



# Set log

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print(os.listdir("../input"))
#

# Sownload stop words

#



nltk.download('stopwords')
#

# Setting variables

#



# DATASET

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

DATASET_ENCODING = "ISO-8859-1"

TRAIN_SIZE = 0.8



# TEXT CLENAING

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"



# WORD2VEC 

W2V_SIZE = 300

W2V_WINDOW = 7

W2V_EPOCH = 32

W2V_MIN_COUNT = 10



# KERAS

SEQUENCE_LENGTH = 300

EPOCHS = 8

BATCH_SIZE = 1024



# SENTIMENT

POSITIVE = "POSITIVE"

NEGATIVE = "NEGATIVE"

NEUTRAL = "NEUTRAL"

SENTIMENT_THRESHOLDS = (0.4, 0.7)



# EXPORT

KERAS_MODEL = "model.h5"

WORD2VEC_MODEL = "model.w2v"

TOKENIZER_MODEL = "tokenizer.pkl"

ENCODER_MODEL = "encoder.pkl"
#

# Reading data

#



"""

Dataset details

    target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

    ids: The id of the tweet ( 2087)

    date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

    flag: The query (lyx). If there is no query, then this value is NO_QUERY.

    user: the user that tweeted (robotickilldozr)

    text: the text of the tweet (Lyx is cool)

"""



dataset_filename = os.listdir("../input")[0]

dataset_path = os.path.join("..","input",dataset_filename)

print("Open file:", dataset_path)

df = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)

print("Dataset size:", len(df))
#

# Show first 5 rows

df.head(5)
%%time

#

# Map target to string

# 



decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

def decode_sentiment(label):

    return decode_map[int(label)]



df.target = df.target.apply(lambda x: decode_sentiment(x))
#

# Split train and test

#



df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)

print("TRAIN size:", len(df_train))

print("TEST size:", len(df_test))
#

# Extract words

#

documents = [_text.split() for _text in df_train.text] 

print('training tweets count', len(documents))
%%time

#

# Word2Vec Build the model

#

w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=8)
%%time

#

# Word2Vec Build the vocab

#

w2v_model.build_vocab(documents)
#

# Get Words

#



words = w2v_model.wv.vocab.keys()

vocab_size = len(words)

print("Vocab size", vocab_size)
%%time

#

# Word2Vec train the model

#

print('train w2v ....')

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

print('done')
#

# Test the trained model

#

w2v_model.most_similar("love")
#

# Create tokenizer

#

%%time

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df_train.text)



vocab_size = len(tokenizer.word_index) + 1

print("Total words", vocab_size)
#

# Convert Text to Sequences

#

%%time

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)

x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)
#

# Label Encoding

#



labels = df_train.target.unique().tolist()

labels.append(NEUTRAL)



encoder = LabelEncoder()

encoder.fit(df_train.target.tolist())



y_train = encoder.transform(df_train.target.tolist())

y_test = encoder.transform(df_test.target.tolist())



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("x_train", x_train.shape)

print("y_train", y_train.shape)

print()

print("x_test", x_test.shape)

print("y_test", y_test.shape)
#

# Embedding layer

#

embedding_matrix = np.zeros((vocab_size, W2V_SIZE))

for word, i in tokenizer.word_index.items():

    if word in w2v_model.wv:

        embedding_matrix[i] = w2v_model.wv[word]

print(embedding_matrix.shape)



embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)

#

# Build Model

#

model = Sequential()

model.add(embedding_layer)

model.add(Dropout(0.5))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))



model.summary()



model.compile(loss='binary_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])
%%time

#

# Train the Model

#



callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]



history = model.fit(x_train, y_train,

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS,

                    validation_split=0.1,

                    verbose=1,

                    callbacks=callbacks)
%%time

#

# Evaluate trained model

#



score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

print()

print("ACCURACY:",score[1])

print("LOSS:",score[0])
#

# Evaluate model (draw charts)

#



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

 

epochs = range(len(acc))

 

plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

 

plt.show()
#

# Predicts

#

def decode_sentiment(score, include_neutral=True):

    if include_neutral:        

        label = NEUTRAL

        if score <= SENTIMENT_THRESHOLDS[0]:

            label = NEGATIVE

        elif score >= SENTIMENT_THRESHOLDS[1]:

            label = POSITIVE



        return label

    else:

        return NEGATIVE if score < 0.5 else POSITIVE



def predict(text, include_neutral=True):

    start_at = time.time()

    # Tokenize text

    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)

    # Predict

    score = model.predict([x_test])[0]

    # Decode sentiment

    label = decode_sentiment(score, include_neutral=include_neutral)



    return {"label": label, "score": float(score),

       "elapsed_time": time.time()-start_at}  

predict("I love the music")
predict("I hate the rain")
predict("i don't know what i'm doing")
#

# Confusion Matrix

#

%%time

y_pred_1d = []

y_test_1d = list(df_test.target)

scores = model.predict(x_test, verbose=1, batch_size=8000)

y_pred_1d = [decode_sentiment(score, include_neutral=False) for score in scores]



def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """



    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=30)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)

    plt.yticks(tick_marks, classes, fontsize=22)



    fmt = '.2f'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label', fontsize=25)

    plt.xlabel('Predicted label', fontsize=25)

    

cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)

plt.figure(figsize=(12,12))

plot_confusion_matrix(cnf_matrix, classes=df_train.target.unique(), title="Confusion matrix")

plt.show()
#

# Classification Report

#

print(classification_report(y_test_1d, y_pred_1d))
#

# Accuracy Score

#

accuracy_score(y_test_1d, y_pred_1d)
#

# Save the model

#

model.save(KERAS_MODEL)

w2v_model.save(WORD2VEC_MODEL)

pickle.dump(tokenizer, open(TOKENIZER_MODEL, "wb"), protocol=0)

pickle.dump(encoder, open(ENCODER_MODEL, "wb"), protocol=0)