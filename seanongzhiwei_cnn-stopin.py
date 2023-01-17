# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import time

import pandas as pd

import numpy as np

import json

from gensim.models import Word2Vec

start = time.time()

test = pd.read_csv('../input/ndsc-beginner/test.csv')

train = pd.read_csv('../input/ndsc-beginner/train.csv')

categories = pd.read_json('../input/ndsc-beginner/categories.json')



# with open('../input/ndsc-beginner/categories.json') as f:

#     cats = json.load(f)





# English stop words

english_stopwords = open('../input/stopword-lists-for-19-languages/englishST.txt').read()

english_stopwords = english_stopwords.split(sep = '\n')



# Bahasa Indonesia stop words

bi_stopwords = pd.read_csv('../input/indonesian-stoplist/stopwordbahasa.csv')



def prep_bi_stopwords():

    temp_list = []

    for word in bi_stopwords['ada']:

        temp_list.append(word)

    

    return temp_list



bi_stopwords = prep_bi_stopwords()



# junk alphabets and fullstop

lone_alphabets = 'a b c d e f g h i j k l m n o p q r s t u v w x y z .'

lone_alphabets = lone_alphabets.split()



# stop_words_master_list = english_stopwords + bi_stopwords + lone_alphabets

stop_words_master_list = english_stopwords + bi_stopwords





def remove_num_from_string(string_sentence):

    string_sentence =  list(filter(lambda x: x not in '0123456789', string_sentence))

    string_sentence = ''.join(string_sentence)

    return string_sentence



def df_column_to_list(train_or_test):

    

    train_or_test = train_or_test['title']

    train_or_test = list(map(lambda x: remove_num_from_string(x), train_or_test))

    train_or_test = list(map(lambda y: y.strip().lower().split(), train_or_test))

    

    return train_or_test



train_feature_list = df_column_to_list(train)

test_feature_list = df_column_to_list(test)



corpus = train_feature_list + test_feature_list







def remove_words(target_list, remove_list):

    done = list(filter(lambda x: x not in remove_list,target_list))

    return done



train_feature_list = list(map(lambda x: remove_words(x,stop_words_master_list), train_feature_list))

test_feature_list = list(map(lambda x: remove_words(x,stop_words_master_list), test_feature_list))





corpus = train_feature_list + test_feature_list

time.time()-start
def replace_empty_with_other_2d(target_list):

    

    output_list = list(map(lambda x: 'product' if len(x)==0 else x, target_list))

    

    return output_list





train_feature_list = replace_empty_with_other_2d(train_feature_list)

test_feature_list = replace_empty_with_other_2d(test_feature_list)

start = time.time()

word_model = Word2Vec(corpus,size = 100, min_count = 1)

time.time()-start
start = time.time()





def prep_feature_arr_list(target_list):

    output_list = word_model.wv[target_list]

    output_list_final = output_list.copy()

    output_list_final.resize(19,100)

    return output_list_final





train_feature_arr = np.array(list(map(lambda x: prep_feature_arr_list(x), train_feature_list)))



test_feature_arr = np.array(list(map(lambda x: prep_feature_arr_list(x), test_feature_list)))



time.time()-start
train_feature_arr.shape
import keras

from keras.models import Sequential

from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation



# start = time.time()



# model = Sequential()

# # model.add(embedding_layer)

# # model.add(Dropout(0.2))



# # model.add(Conv1D(128, 3, padding='valid',activation='relu',strides=1))

# model.add(Conv1D(64, 3, padding='valid',activation='relu',strides=1))

# model.add(Conv1D(32, 3, padding='valid',activation='relu',strides=1))

# model.add(Flatten())

# model.add(Dropout(0.2))

# model.add(Dense(64,activation='relu'))

# model.add(Dropout(0.2))

# model.add(Dense(58,activation='softmax'))

# # opt = keras.optimizers.Adam(lr=1, epsilon=0.1)

# model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['acc'])

# time.time()-start
from keras.models import Sequential

from keras.layers import LSTM, Dense

import numpy as np



# data_dim = 16

# timesteps = 8

# num_classes = 58



train_label = keras.utils.to_categorical(train['Category'].values, num_classes=58, dtype='float32')



# expected input data shape: (batch_size, timesteps, data_dim)

model = Sequential()

model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(19, 100)))  # returns a sequence of vectors of dimension 32

model.add(LSTM(32, return_sequences=True, activation='relu'))  # returns a sequence of vectors of dimension 32

model.add(Dropout(0.2))

model.add(LSTM(32, activation='relu'))  # return a single vector of dimension 32

model.add(Dropout(0.2))

model.add(Dense(58, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])







history = model.fit(train_feature_arr, train_label,validation_split = 0.2, epochs=5, batch_size=64, verbose = 1)
train_feature_arr.shape
# import keras

# start = time.time()



# train_label = keras.utils.to_categorical(train['Category'].values, num_classes=58, dtype='float32')





# history = model.fit(train_feature_arr, train_label, validation_split = 0.2, epochs=3, batch_size=64, verbose = 1)



# time.time()-start
# plt.plot(model.history['acc'])
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
start = time.time()

predicted = model.predict(test_feature_arr)

predicted = predicted.argmax(axis=1)

print(predicted)

time.time()-start
ans = pd.DataFrame(predicted)

submission = pd.DataFrame(test.itemid)

submission = submission.join(ans)

submission = submission.rename(columns = {0:'Category'})

submission.head()
submission.to_csv('Predominant_submission_LSTM_3epoch.csv', index = False)