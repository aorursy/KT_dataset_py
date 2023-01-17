import pandas as pd

from nltk import word_tokenize, sent_tokenize

from random import shuffle

from tqdm import tqdm as tqdm

from keras import backend as K

import numpy

from sklearn.metrics import f1_score



from keras.preprocessing.text import Tokenizer

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Input, Embedding, Activation, Flatten, Dense

from keras.layers import Conv1D, MaxPooling1D, Dropout, Conv2D, MaxPooling2D

from keras.models import Model

from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.preprocessing import sequence

from keras.engine.topology import Layer



import tensorflow as tf
df = pd.read_csv("/kaggle/input/ukara-enhanced/dataset.csv")
df.head()
groups = {}



kelompok = [1, 3, 4, 7, 8, 9, 10, 'A', 'B']



for I in kelompok:

    groups[(I,0)] = df.query("kelompok == '%s' and label == 0 " % (str(I)))['teks'].values 



for I in kelompok:

    groups[(I,1)] = df.query("kelompok == '%s' and label == 1 " % (str(I)))['teks'].values



for T in groups:

    length = len(groups[T])

    groups[T] = [list(groups[T][length*(J)//5 : length*(J+1)//5]) for J in range(5)]



def generate_fold_data(test_index, sentence_preprocess):

    train = []

    test  = {}

    

    for T in groups:

        if T[0] not in test:

            test[T[0]] = []

        for index in range(5):

            if test_index == index:

                for M in groups[T][index]:

                    test[T[0]].append((sentence_preprocess(T, M), T[1]))

            else:

                for M in groups[T][index]:

                    train.append((sentence_preprocess(T, M), T[1]))

    

    shuffle(train)

    return train, test

def remove_stopwords(words):

    words = word_tokenize(words.lower())

    words = [I for I in words if I not in stopwords]

    return " ".join(words)
def append_id_short(group_id):

    dictionary = {1: "aaaa",

                 3: "bbbb",

                 4: "cccc",

                 7: "ffff",

                 8: "gggg",

                 9: "hhhh",

                 10: "iiii",

                 "A": "jjjj",

                 "B": "kkkk"}

    return dictionary[group_id]
file = open("/kaggle/input/ukara-enhanced/stopword_list.txt")

    

stopwords = [I.strip() for I in file.readlines()]



file.close()
def preprocess_stop_short(group_id, sentence):

    sentence = remove_stopwords(sentence)

    tokens = split_letters(sentence)

    tokens.insert(0, append_id_short(group_id[0]))

    return " ".join(tokens)



def preprocess_short(group_id, sentence):

    sentence = append_id_short(group_id[0], sentence)

    return sentence



def split_letters(sentence):

    tokens = [I for I in sentence]

    return tokens
def construct_char_dict():

    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ,.!?'"



    label = ["aaaa", "bbbb", "cccc", "ffff", "gggg", "hhhh", "iiii", "jjjj", "kkkk"]



    char_dict = {}

    for i, char in enumerate(label):

        char_dict[char] = i + 1



    for i, char in enumerate(alphabet):

        char_dict[char] = i + len(label) + 1

        

    return char_dict
len(construct_char_dict())
max([len(I) for I in df['teks'].values])
MAX_LENGTH = 2000

EMBEDDING_VECTOR_LENGTH = 40

EPOCH = 5

BATCH_SIZE = 64
def evaluate_model(model_function, preprocessing_function):

    scores = {}

    for R in tqdm(range(5)):

        train, test = generate_fold_data(R, preprocessing_function)

        train_X = [I[0] for I in train]

        train_y = [I[1] for I in train]

        

        tk = Tokenizer(num_words=None, oov_token='UNK')

        tk.fit_on_texts(train_X)

        

        tk.word_index = construct_char_dict()



        tk.word_index[tk.oov_token] = max(construct_char_dict().values()) + 1

        

        vector_train = tk.texts_to_sequences(train_X)

        

        train_Z = sequence.pad_sequences(vector_train, maxlen=MAX_LENGTH)

        model = model_function()

        model.fit(train_Z, train_y,  epochs= EPOCH, batch_size= BATCH_SIZE, verbose=0)

        

        for J in test:

            test_X = [T[0] for T in test[J]]

            test_y = [T[1] for T in test[J]]



            

            vector_test = tk.texts_to_sequences(test_X)



            test_Z = sequence.pad_sequences(vector_test, maxlen=MAX_LENGTH)         

            prediction = model.predict(test_Z)

            accuracy = f1_score(test_y, [I[0] > 0.5 for I in prediction])

            if J not in scores:

                scores[J] = 0

            scores[J] += accuracy / 5

    return scores
def model_function():

    model = Sequential()

    model.add(Embedding(53, EMBEDDING_VECTOR_LENGTH, input_length=MAX_LENGTH))

    model.add(Conv1D(32, 3))

    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(25, activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(1, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
def format_accuracy(dictionary_result):

    total = 0

    for I in dictionary_result:

        total += dictionary_result[I]

        print(I, dictionary_result[I])

    print("Macro All", total / len(dictionary_result))    
scores = evaluate_model(model_function, preprocess_stop_short)

format_accuracy(scores)
def model_function_2():

    model = Sequential()

    model.add(Embedding(53, EMBEDDING_VECTOR_LENGTH, input_length=MAX_LENGTH))

    model.add(Conv1D(32, 3))

    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(32, 3))

    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(25, activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(1, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
scores = evaluate_model(model_function_2, preprocess_stop_short)

format_accuracy(scores)