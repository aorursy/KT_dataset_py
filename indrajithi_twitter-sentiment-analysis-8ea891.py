!pip install gensim --upgrade

!pip install keras --upgrade

!pip install pandas --upgrade

# DataFrame

import pandas as pd



# Matplot

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
nltk.download('stopwords')
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
dataset_filename = os.listdir("../input")[0]

dataset_path = os.path.join("..","input",dataset_filename)

print("Open file:", dataset_path)

df = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
print("Dataset size:", len(df))
df.head(5)
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

def decode_sentiment(label):

    return decode_map[int(label)]
%%time

df.target = df.target.apply(lambda x: decode_sentiment(x))
target_cnt = Counter(df.target)



plt.figure(figsize=(16,8))

plt.bar(target_cnt.keys(), target_cnt.values())

plt.title("Dataset labels distribuition")
stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")
def preprocess(text, stem=False):

    # Remove link,user and special characters

    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

    tokens = []

    for token in text.split():

        if token not in stop_words:

            if stem:

                tokens.append(stemmer.stem(token))

            else:

                tokens.append(token)

    return " ".join(tokens)
%%time

df.text = df.text.apply(lambda x: preprocess(x))
df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)

print("TRAIN size:", len(df_train))

print("TEST size:", len(df_test))
%%time

documents = [_text.split() for _text in df_train.text] 
w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=8)
w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()

vocab_size = len(words)

print("Vocab size", vocab_size)
%%time

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
w2v_model.most_similar("love")
%%time

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df_train.text)



vocab_size = len(tokenizer.word_index) + 1

print("Total words", vocab_size)
%%time

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)

x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)
labels = df_train.target.unique().tolist()

labels.append(NEUTRAL)

labels
encoder = LabelEncoder()

encoder.fit(df_train.target.tolist())



y_train = encoder.transform(df_train.target.tolist())

y_test = encoder.transform(df_test.target.tolist())



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("y_train",y_train.shape)

print("y_test",y_test.shape)
print("x_train", x_train.shape)

print("y_train", y_train.shape)

print()

print("x_test", x_test.shape)

print("y_test", y_test.shape)
y_train[:10]
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))

for word, i in tokenizer.word_index.items():

  if word in w2v_model.wv:

    embedding_matrix[i] = w2v_model.wv[word]

print(embedding_matrix.shape)
embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)
model = Sequential()

model.add(embedding_layer)

model.add(Dropout(0.5))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))



model.summary()
model.compile(loss='binary_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
%%time

history = model.fit(x_train, y_train,

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS,

                    validation_split=0.1,

                    verbose=1,

                    callbacks=callbacks)
%%time

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

print()

print("ACCURACY:",score[1])

print("LOSS:",score[0])
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
%%time



cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)

plt.figure(figsize=(12,12))

plot_confusion_matrix(cnf_matrix, classes=df_train.target.unique(), title="Confusion matrix")

plt.show()
print(classification_report(y_test_1d, y_pred_1d))
accuracy_score(y_test_1d, y_pred_1d)
model.save(KERAS_MODEL)

w2v_model.save(WORD2VEC_MODEL)

pickle.dump(tokenizer, open(TOKENIZER_MODEL, "wb"), protocol=0)

pickle.dump(encoder, open(ENCODER_MODEL, "wb"), protocol=0)