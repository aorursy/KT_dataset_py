from sklearn.metrics import accuracy_score

import numpy as np

import pandas as pd

import spacy

from sklearn.ensemble import RandomForestClassifier

import time

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import cross_validate

import time

from scipy import sparse

import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerLine2D

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC

# import xgboost as xgb

from sklearn.metrics import classification_report

from nltk.stem import WordNetLemmatizer

from sklearn.naive_bayes import BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from tqdm import tqdm

import time

from keras.preprocessing.sequence import pad_sequences

import os

from glob import glob

from zipfile import ZipFile

import zipfile

from numpy import asarray

from numpy import zeros

from nltk.stem import PorterStemmer

ps = PorterStemmer()

from nltk.stem.lancaster import LancasterStemmer

lc = LancasterStemmer()

from nltk.stem import SnowballStemmer

sb = SnowballStemmer("english")

import gc

import gensim

import io



def read_LSTM(reset_data):

    from keras.preprocessing.text import Tokenizer

    from keras.preprocessing import sequence

    import os

    max_length = 50

    # load data

    if(reset_data or not os.path.exists('lstm_data_vector.npy') or not os.path.exists("lstm_labels_vector.npy") or not os.path.exists("lstm_qid_vector.npy")):

        start_time = time.time()

        train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv').fillna('')

        test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv').fillna('')

        qid = test.drop('question_text', axis=1)

        training_labels = train['target'].to_numpy()    

        train_text = train['question_text']

        test_text = test['question_text']

        text_list = pd.concat([train_text, test_text])

        y = train['target'].values

        num_train_data = y.shape[0]

        start_time = time.time()

        nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])

        nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

        word_dict = {}

        word_index = 1

        lemma_dict = {}

        docs = nlp.pipe(text_list, n_threads = 2)

        word_sequences = []

        for doc in tqdm(docs):

            word_seq = []

            for token in doc:

                if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):

                    word_dict[token.text] = word_index

                    word_index += 1

                    lemma_dict[token.text] = token.lemma_

                if token.pos_ is not "PUNCT":

                    word_seq.append(word_dict[token.text])

            word_sequences.append(word_seq)

        del docs

        train_word_sequences = word_sequences[:num_train_data]

        test_word_sequences = word_sequences[num_train_data:]

        train_word_sequences = pad_sequences(train_word_sequences, maxlen=max_length, padding='post')

        test_word_sequences = pad_sequences(test_word_sequences, maxlen=max_length, padding='post')

        # print(padded_train)



        # Save Data

        #np.save("lstm_data_vector.npy",train_word_sequences)

        #np.save("lstm_labels_vector.npy", training_labels)

        #np.save("lstm_qid_vector.npy", qid)

        #np.save("lstm_test_vector.npy", test_word_sequences)

        

    else:

        padded_train = np.load("lstm_data_vector.npy")

        training_labels = np.load("lstm_data_vector.npy")

        qid = np.load("lstm_qid_vector.npy")

        padded_test = np.load("lstm_test_vector.npy")







    x_train, x_test, y_train, y_test = train_test_split(train_word_sequences, training_labels, test_size=0.01, random_state=27)

    print('x_train.shape: ', x_train.shape)

    print('x_test.shape: ', x_test.shape)

    print('y_train.shape: ', y_train.shape)

    print('y_test.shape: ', y_test.shape)

    max_question_length = 50

    vocab_size = len(word_dict)

    print(vocab_size)

    return word_dict, lemma_dict, test_word_sequences, qid, x_train, x_test, y_train, y_test, vocab_size, max_question_length



def do_LSTM(word_dict, lemma_dict, actual_test, x_train, x_test, y_train, y_test, vocab_size, max_question_length):

    from keras.models import Sequential

    from keras.layers import Dense, LSTM

    from keras.layers.embeddings import Embedding

    max_length = 50

    dim=300

    embeddings_index={}

    with zipfile.ZipFile("../input/quora-insincere-questions-classification/embeddings.zip") as zf:

        with io.TextIOWrapper(zf.open("glove.840B.300d/glove.840B.300d.txt"), encoding="utf-8") as f:

            for line in tqdm(f):

                values=line.split(' ')

                word=values[0]

                vectors=np.asarray(values[1:],'float32')

                embeddings_index[word]=vectors

    embeddings_index = dict(embeddings_index)

    print('embedding stuff')

    embed_size = 300

    embedding_matrix = np.zeros((len(word_dict)+1, embed_size), dtype=np.float32)

    empty = np.zeros((embed_size,), dtype=np.float32) - 1.

    for key in tqdm(word_dict):

        word = key

        embedding_vector1 = embeddings_index.get(word)

        word = key.lower()

        embedding_vector2 = embeddings_index.get(word)

        word = key.upper()

        embedding_vector3 = embeddings_index.get(word)

        word = key.capitalize()

        embedding_vector4 = embeddings_index.get(word)

        word = ps.stem(key)

        embedding_vector5 = embeddings_index.get(word)

        word = lemma_dict[key]

        embedding_vector6 = embeddings_index.get(word)

        if embedding_vector1 is not None:

            embedding_matrix[word_dict[key]] = embedding_vector1

            continue

        if embedding_vector2 is not None:

            embedding_matrix[word_dict[key]] = embedding_vector2

            continue

        if embedding_vector3 is not None:

            embedding_matrix[word_dict[key]] = embedding_vector3

            continue

        if embedding_vector4 is not None:

            embedding_matrix[word_dict[key]] = embedding_vector4

            continue

        if embedding_vector5 is not None:

            embedding_matrix[word_dict[key]] = embedding_vector5

            continue

        if embedding_vector6 is not None:

            embedding_matrix[word_dict[key]] = embedding_vector6

            continue

        embedding_matrix[word_dict[key]] = empty        

    

    embedding_vecor_length = 50 # num features for each sentence?

    model = Sequential()

    model.add(Embedding(vocab_size+1, 300, weights=[embedding_matrix], input_length=max_length, trainable=False))

    model.add(LSTM(100))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    model.fit(x_train, y_train, epochs=3, batch_size=512)

    

    print("starting predict")

    start = time.time()

    predictions = model.predict_classes(x_test)

    print('Time to predict: ', time.time() - start)    

    

    print("starting Real predict")

    start = time.time()

    actual_pred = model.predict_classes(actual_test)

    print('Time to real predict: ', time.time() - start)    

    

    return actual_pred, predictions





def main():

    max_length = 55

    lstm = True

    reset_data = True

    if lstm:

        word_dict, lemma_dict, actual_test, qid, x_train, x_test, y_train, y_test, vocab_size, max_question_length = read_LSTM(reset_data)

    else:

        return

    print("starting training")

    start = time.time()

    # use actual_test instead of x_test to do real predictions

    # tes = do_LSTM(x_train, x_test, y_train, y_test, vocab_size, max_question_length)

    actual_pred, tes = do_LSTM(word_dict, lemma_dict, actual_test, x_train, x_test, y_train, y_test, vocab_size, max_question_length)

    



    print('Time to train & predict: ', time.time() - start)

    print("confusion matrix:\n", confusion_matrix(y_test, tes))

    print("accuracy: ", accuracy_score(y_test, tes))

    print("f1: ", f1_score(y_test, tes, average='macro'))

    print("classification report:\n", classification_report(y_test, tes))

    print("precision: ", precision_score(y_test, tes))

    print("recall: ", recall_score(y_test, tes))

    print(actual_pred)

    qid['prediction'] = actual_pred

    qid.to_csv('submission.csv', index=False)



if __name__ == "__main__":

    main()