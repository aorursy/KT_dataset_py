import matplotlib.pyplot as plt

%matplotlib inline

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

from nltk import sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer

import string

from sklearn.model_selection import train_test_split

import tensorflow as tf



import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")

test = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")



import warnings

warnings.filterwarnings('ignore')
maxlen = 200

max_features = 20000



def remove_punctuation(text):

    no_punc = "".join([c for c in text if c not in string.punctuation])

    return no_punc



regtok = RegexpTokenizer(r'\w+')



def remove_stop_words(text):

    output = [c for c in text if c not in stopwords.words('english')]

    return output



lemmatizer = WordNetLemmatizer()



def word_lemmatizer(text):

    lem_text = " ".join([lemmatizer.lemmatize(i) for i in text])

    return lem_text



stemmer = PorterStemmer()



def word_stemmer(text):

    stem_text = " ".join([stemmer.stem(i) for i in text])

    return stem_text

    

def preprocess_text(data):

    data = data.apply(lambda x: remove_punctuation(x))

    data = data.apply(lambda x: regtok.tokenize(x.lower()))

    data = data.apply(lambda x: remove_stop_words(x))

    data = data.apply(lambda x: word_lemmatizer(x))

    

    return data



def prepare_data_for_training(train, test, tok):

    train = tok.texts_to_sequences(train)

    test = tok.texts_to_sequences(test)

    word_index = tok.word_index

    print('Found %s unique tokens.' % len(word_index))

    

    train = pad_sequences(train, maxlen=maxlen)

    test = pad_sequences(test, maxlen=maxlen)



    return train, test



def estimator(X_t, y):

    inp = Input(shape=(maxlen, ))

    embed_size = 128

    x = Embedding(max_features, embed_size)(inp)

    x = LSTM(60, return_sequences=True,name='lstm_layer')(x)

    x = GlobalMaxPool1D()(x)

    x = Dropout(0.1)(x)

    x = Dense(50, activation="relu")(x)

    x = Dropout(0.1)(x)

    x = Dense(6, activation="sigmoid")(x)

    

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',

                      optimizer='adam',

                      metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])

    batch_size = 32

    epochs = 5

    history = model.fit(X_t,y, batch_size=batch_size, epochs=epochs)

    

    return model, history



def evaluate_model(model, features, predictions):

    results = model.evaluate(features, predictions)

    

    for i in range(len(model.metrics_names)):

        print(model.metrics_names[i], results[i])
# Without Preprocessing

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_train = train["comment_text"]

list_sentences_test = test["comment_text"]



# # SHORT DATASET FOR DEBUGGING

# list_sentences_train = list_sentences_train[0:500]

# y = y[0:500]



# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(list_sentences_train, y, test_size=0.2, random_state=42)



tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(X_train))



sentences_train, sentences_test = prepare_data_for_training(X_train, X_test, tokenizer)

model, hist = estimator(sentences_train, y_train)
print('\n# Evaluate on test data')

evaluate_model(model, sentences_test, y_test)
# With preprocessing.

X_train_pp = preprocess_text(X_train)

X_test_pp = preprocess_text(X_test)



tokenizer_pp = Tokenizer(num_words=max_features)

tokenizer_pp.fit_on_texts(list(X_train_pp))



sentences_train_pp, sentences_test_pp = prepare_data_for_training(X_train_pp, X_test_pp, tokenizer_pp)

model_pp, hist_pp = estimator(sentences_train_pp, y_train)
print('\n# Evaluate on test data')

evaluate_model(model_pp, sentences_test_pp, y_test)
def loadEmbeddingMatrix_wv(word_index):

    embed_size = 100

    embeddings_index = dict()

    

    for word in wv_model.wv.vocab:

        embeddings_index[word] = wv_model.wv[word]

    print('Loaded %s word vectors.' % len(embeddings_index))

    

    gc.collect()

    all_embs = np.stack(list(embeddings_index.values()))

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    

    nb_words = len(word_index) + 1

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    gc.collect()

  

    embeddedCount = 0

    for word, i in word_index.items():

        i-=1

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: 

            embedding_matrix[i] = embedding_vector

            embeddedCount+=1

    print('total embedded:',embeddedCount,'common words')



    del(embeddings_index)

    gc.collect()



    return embedding_matrix



def estimator_embedding(X_t, y, embedding_matrix, total_words):

    inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier

    x = Embedding(total_words, embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)

    x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)

    x = GlobalMaxPool1D()(x)

    x = Dropout(0.1)(x)

    x = Dense(50, activation="relu")(x)

    x = Dropout(0.1)(x)

    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',

                      optimizer='adam',

                      metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])



    batch_size = 32

    epochs = 5

    history = model.fit(X_t,y, batch_size=batch_size, epochs=epochs)

    

    return model, history
# With word embeddings

import gc

from gensim.models import Word2Vec

from nltk.tokenize import RegexpTokenizer



# Training custom word2vec model

list_sentences_tok = X_train.apply(lambda x: regtok.tokenize(x))

wv_model = Word2Vec(list_sentences_tok, min_count=1)

embedding_matrix = loadEmbeddingMatrix_wv(tokenizer.word_index)



model_w2v, hist_w2v = estimator_embedding(sentences_train, y_train, embedding_matrix, len(tokenizer.word_index) + 1)
print('\n# Evaluate on test data')

evaluate_model(model_w2v, sentences_test, y_test)
# Training custom word2vec model - Preprocessed

list_sentences_tok_pp = X_train_pp.apply(lambda x: regtok.tokenize(x))

wv_model = Word2Vec(list_sentences_tok_pp, min_count=1)

embedding_matrix = loadEmbeddingMatrix_wv(tokenizer_pp.word_index)



model_w2v, hist_w2v = estimator_embedding(sentences_train_pp, y_train, embedding_matrix, len(tokenizer_pp.word_index) + 1)
print('\n# Evaluate on test data')

evaluate_model(model_w2v, sentences_test_pp, y_test)