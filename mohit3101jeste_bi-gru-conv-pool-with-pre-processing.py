import sys, os, re, csv, codecs, numpy as np, pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

import nltk

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

import re

import sys

import warnings

import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from tensorflow.keras.layers import Bidirectional, GRU, Conv1D, GlobalAveragePooling1D

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import Callback

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
train.head()
print("Check for missing values in Train dataset")

null_check=train.isnull().sum()

print(null_check)

print("Check for missing values in Test dataset")

null_check=test.isnull().sum()

print(null_check)

print("filling NA with \"unknown\"")

train["comment_text"].fillna("unknown", inplace=True)

test["comment_text"].fillna("unknown", inplace=True)

data = train

if not sys.warnoptions:

    warnings.simplefilter("ignore")

def cleanHtml(sentence):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', str(sentence))

    return cleantext

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    cleaned = cleaned.strip()

    cleaned = cleaned.replace("\n"," ")

    return cleaned

def keepAlpha(sentence):

    alpha_sent = ""

    for word in sentence.split():

        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)

        alpha_sent += alpha_word

        alpha_sent += " "

    alpha_sent = alpha_sent.strip()

    return alpha_sent

data['comment_text'] = data['comment_text'].str.lower()

data['comment_text'] = data['comment_text'].apply(cleanHtml)

data['comment_text'] = data['comment_text'].apply(cleanPunc)

data['comment_text'] = data['comment_text'].apply(keepAlpha)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])

re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def removeStopWords(sentence):

    global re_stop_words

    return re_stop_words.sub(" ", sentence)

data['comment_text'] = data['comment_text'].apply(removeStopWords)
train = data

print(train.shape)

train.head()
test.head()
data = test



data['comment_text'] = data['comment_text'].str.lower()

data['comment_text'] = data['comment_text'].apply(cleanHtml)

data['comment_text'] = data['comment_text'].apply(cleanPunc)

data['comment_text'] = data['comment_text'].apply(keepAlpha)

data['comment_text'] = data['comment_text'].apply(removeStopWords)



test = data

print(test.shape)
test.head()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_train = train["comment_text"]

list_sentences_test = test["comment_text"]
max_features = 20000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
maxlen = 200

X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

print(X_train.shape)

print(y.shape)

print(X_test.shape)
X_train, X_val, y_train, y_val = train_test_split(X_train, y, train_size=0.8, random_state=233)
class RocAucEvaluation(Callback):

    def __init__(self, validation_data=(), interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.X_val, self.y_val = validation_data



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict(self.X_val, verbose=0)

            score = roc_auc_score(self.y_val, y_pred)

            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
def model():

    inputs = Input(shape=(maxlen,), name="input")

    layer = Embedding(max_features, 128, name="embedding")(inputs)

    layer = Bidirectional(GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name="bi_gru_0"))(layer)

    layer = Conv1D(64, kernel_size = 3, padding = "valid", activation='relu', name="conv1d_0")(layer)

    layer = GlobalAveragePooling1D(name="avg_pool_0")(layer)

    layer = Dense(32,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5,name="fc1_dropout")(layer)

    layer = Dense(6,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model = model()

model.summary()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),

                  optimizer='adam',

                  metrics=['accuracy'])
checkpoint_path = os.path.join("../input/output/","lstm-custom-embeddings-v4.hdf5")

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,

                                                 verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')





ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)
batch_size = 64

epochs = 3

history = model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),shuffle=True, callbacks=[cp_callback,ra_val])
y_pred = model.predict(X_test,batch_size=1024,verbose=1)
'''

submission = pd.read_csv(os.path.join("../input/jigsaw-toxic-comment-classification-challenge/","sample_submission.csv.zip"))

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred

submission.to_csv(os.path.join("../input/toxic-comment-challenge-submission/",'submission.csv'), index=False)

'''
test_labels = pd.read_csv(os.path.join("../input/jigsaw-toxic-comment-classification-challenge","test_labels.csv.zip"))
test_set = test.join(test_labels.set_index("id"),on="id")
test_set.head()
test_set = test_set[test_set.obscene!=-1]
test_set.head()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_test = test_set[list_classes].values

list_sentences_test = test_set["comment_text"]
max_features = 20000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_test))

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
maxlen = 200

X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

print(y_test.shape)

print(X_test.shape)
model.evaluate(X_test,y_test)