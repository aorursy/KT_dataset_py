import numpy as np

import pandas as pd

import nltk

import re

import os

import string

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer

from tensorflow import keras

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.models import Model, load_model

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, Flatten, Dense, Conv1D, MaxPool1D, Concatenate, TimeDistributed, LSTM

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.model_selection import KFold
PATH_DATASET = "../input/foodreview/foodreview.csv"

PATH_GLOVE = "../input/glove6b100dtxt/glove.6B.100d.txt"
def get_word2vec(resource_path):

    embeddings_index = {}

    f = open(resource_path, encoding="utf8")

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index



FOODREVIEW_CSV = pd.read_csv(PATH_DATASET, encoding = "ISO-8859-1")

GLOVE_W2V = get_word2vec(PATH_GLOVE)
#preprocessing functions



def lower(input_string):

    return input_string.lower()



def remove_html(input_string):

    html_compile = re.compile('<.*?>')

    cleantext = re.sub(html_compile, '', str(input_string))

    return cleantext



def remove_number(input_string):

    return re.sub(r'\d+', '', str(input_string))



def remove_punc(input_string):

    return re.sub(r'[^\w\s]','', str(input_string))



def remove_whitespace(input_string):

    return input_string.strip()



def tokenize(input_string):

    return word_tokenize(input_string)

    

def remove_stopwords(input_tokens):

    stop_words = set(stopwords.words('english'))

    return [i for i in input_tokens if not i in stop_words]



def stemming(input_tokens):

    ps = PorterStemmer()

    return [ps.stem(i) for i in input_tokens]



def preprocess_one_text(input_sentence):

    result = remove_html(input_sentence)

    result = remove_number(result)

    result = remove_punc(result)

    result = lower(result)

    result = remove_whitespace(result)

    result = tokenize(result)

    result = stemming(result)

    result = remove_stopwords(result)

    return result



def preprocess_texts(dataset):

    return [preprocess_one_text(datum) for datum in dataset]



def preprocess_train_dataset(dataset_df):

    dropped_na_df = dataset_df.dropna()

    dropped_na_df.reset_index()

    merged_texts = [str(sum_text) + " " + str(main_text) for sum_text, main_text in zip(dataset_df['Summary'], 

                                                                                        dataset_df['Text'])]

    return preprocess_texts(dataset_df), dataset_df['Score']

                            

def preprocess_test_dataset(dataset_df):

    merged_texts = [str(sum_text) + " " + str(main_text) for sum_text, main_text in zip(dataset_df['Summary'], 

                                                                                        dataset_df['Text'])]

    return preprocess_texts(dataset_df)
foodreview_sample = FOODREVIEW_CSV

merged_texts = [str(sum_text) + " " + str(main_text) for sum_text, main_text in zip(foodreview_sample['Summary'], foodreview_sample['Text'])]

prepocessed_data = preprocess_texts(merged_texts)

labels = foodreview_sample['Score']



def keras_tokenizer(dataset, num_words=500):

    tokenizer  = Tokenizer(num_words)

    tokenizer.fit_on_texts(dataset)

    sequences =  tokenizer.texts_to_sequences(dataset)



    word_index = tokenizer.word_index

    print("unique words : {}".format(len(word_index)))



    return tokenizer, word_index, pad_sequences(sequences, maxlen=num_words)



# ini 2 input yang dimasukin ke CNN

tokenizer_keras, word_idx, keras_input = keras_tokenizer(prepocessed_data)
def generate_embedded_matrix(embedded_index, word_index, embedding_dim):

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():

        embedding_vector = embedded_index.get(word)

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

    

    return embedding_matrix



def create_embedded_layer(word_index, embedding_dim, max_seq_length, w2v_model=None):

    word2vec_index = w2v_model if w2v_model else get_word2vec(PATH_GLOVE)

    embedding_matrix = generate_embedded_matrix(word2vec_index, word_index, embedding_dim)

    return Embedding(len(word_index) + 1,

                            embedding_dim,

                            weights=[embedding_matrix],

                            input_length=max_seq_length,

                            trainable=False)
class ProposedMultiChannelCNNLSTM:

    def __init__(self, word_index, epochs=20):

        

        self.model = None

        self.word_idx = word_index

        self.max_seq_len = 500

        self.embedding_dim = 100

        self.epochs = epochs

        

    def _build_model(self, label_count):

        sequence_input = Input(shape=(self.max_seq_len,), dtype='int32')

        

        # pembuatan embedded layer

        embedding_layer = create_embedded_layer(self.word_idx, 

                                                self.embedding_dim, 

                                                self.max_seq_len)

        embedded_sequences = embedding_layer(sequence_input)

        

        # convultion layers. untuk feature extraction

        conv_1 = Conv1D(200, 3, activation='relu')(embedded_sequences)

        pool_1 = MaxPool1D(5)(conv_1)

        

        conv_2 = Conv1D(200, 4, activation='relu')(embedded_sequences)

        pool_2 = MaxPool1D(5)(conv_2)

        

        conv_3 = Conv1D(200, 5, activation='relu')(embedded_sequences)

        pool_3 = MaxPool1D(5)(conv_3)

        

        concat_layer = Concatenate(axis=1)([pool_1, pool_2, pool_3])

        

        x = Conv1D(200, 5, activation='relu')(concat_layer)

        x = MaxPool1D(293)(x)  # global max pooling

        x = TimeDistributed(Flatten())(x)

        x = LSTM(512)(x)

        x = Dense(128, activation='relu')(x)

        preds = Dense(label_count, activation='softmax')(x)



        self.model = Model(sequence_input, preds)

        self.model.compile(loss='categorical_crossentropy',

                      optimizer='rmsprop',

                      metrics=['acc'])

        

    def fit(self, X, y):

        print("Fitting...")

        self._build_model(len(set(y)))

        y_categ = to_categorical(y)

        self.model.fit(X, y_categ[:,1:], epochs=self.epochs, batch_size=1024)

        print("Fitting completed")

        print()

        

    def predict(self, X):

        print("Predicting...")

        return [np.argmax(x_pred)+1 for x_pred in self.model.predict(X)]

    

    def summarize(self):

        print("Summarizing...")

        if not self.model:

            print("Model hasn't been trained")

        else:

            self.model.summary()

            print("Summarizing completed")

        print()

    

    def save(self, filename):

        self.model.save(filename)
kf = KFold(n_splits=5)

proposed_cnn = ProposedMultiChannelCNNLSTM(word_idx, 20)

X = keras_input

y = labels

acc_list = []

f1_list = []

confuse_list = []



for train_index, test_index in kf.split(X):

    print(train_index)

    print(test_index)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    proposed_cnn.fit(X_train, y_train)

    proposed_preds = proposed_cnn.predict(X_test)

    acc_list.append(accuracy_score(y_test, proposed_preds))

    f1_list.append(f1_score(y_test, proposed_preds, average='macro'))

    confuse_list.append(confusion_matrix(y_test, proposed_preds))
print("Cross Validation Results")

k = 1

for acc, f1_sc, conf_mtx in zip(acc_list, f1_list, confuse_list):

    print("Fold ", k)

    print("Accuracy: ", acc)

    print("F1 Score: ", f1_sc)

    print("Confusion Matrix")

    print(conf_mtx)

    print()

    k+=1
df_metrics = {"Fold": ["Fold " + str(k+1) for k in range(5)], "Accuracy": acc_list, "F1 Score":f1_list}

print("Cross Validation Evaluation")

pd.DataFrame(df_metrics)
proposed_cnn.summarize()
# masukkan test set anda pada variabel ini. Diasumsikan test_set berupa csv

test_set = pd.read_csv("[INSERT CSV DATASET HERE]")
# preprocessing test test

preprocessed_test_set = preprocess_test_dataset(test_set)

test_set_keras_input = pad_sequences(tokenizer_keras.text_to_sequences(preprocessed_test_set), maxlen=num_words)
model_to_test = ProposedMultiChannelCNNLSTM(word_idx, 20)

model_to_test.fit(keras_input, labels)

model_to_test.save("multi_channel_cnn_lstm.h5")
# model_to_test = load_model("multi_channel_cnn_lstm.h5")
# predicting

preds = model_to_test.predict(test_set_keras_input)
# jika test label berada dalam file terpisah, silahkan definisikan sendiri (dengan read_csv atau semacamnya)

test_labels = test_set["Score"]



# Penghitungan Metrik

print("Test Accuracy: ", accuracy_score(test_set["Score"], preds))

print("Test F1 Score: ", f1_score(test_set["Score"], preds))

print("Confusion Matrix")

print(confusion_matrix(test_set["Score"], preds))