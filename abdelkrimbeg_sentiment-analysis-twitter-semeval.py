# get data 

import pandas as pd



dataTrain = pd.read_table('../input/semevalll/SemEval2017-train.txt' , usecols=[1,2], encoding='utf-8', names=['sentiment', 'tweet'])

dataTest = pd.read_table('../input/semevalll/SemEval2017-test.txt', usecols=[1,2], encoding='utf-8', names=['sentiment', 'tweet'])                   

combine = [dataTrain,dataTest]



# preprocessing of data





import os

from pandas import DataFrame

import pandas as pd

import re

from nltk.corpus import stopwords

import string

from nltk.tokenize import word_tokenize









def clean_twt(text):

   

    # remove tashkeel

    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')

    text = re.sub(p_tashkeel, "", text)



    # remove longation

    p_longation = re.compile(r'(.)\1+')

    subst = r"\1\1"

    text = re.sub(p_longation, subst, text)



    text = text.replace('وو', 'و')

    text = text.replace('يي', 'ي')

    text = text.replace('اا', 'ا')

   

    text = text.replace('آ', 'ا')

    text = text.replace('إ', 'ا')

    text = text.replace('أ', 'ا')

    text = text.replace('ة', 'ه')





    return text





def normalize(text):

    noise = re.compile("""

                                ~   | # Tashdid

                                 ّ    | # Tashdid

                                 َ    | # Fatha

                                 ً    | # Tanwin Fath

                                 ُ    | # Damma

                                 ٌ    | # Tanwin Damm

                                 ِ    | # Kasra

                                 ٍ    | # Tanwin Kasr

                                 ْ    | # Sukun

                                 _     # tawila

                                """, re.VERBOSE)

    text=re.sub(noise,'',text)

    text = re.sub("[إأآا]", "ا", text)

    text = re.sub("ى", "ي", text)

    text = re.sub("ؤ", "ء", text)

    text = re.sub("ئ", "ء", text)

    text = re.sub("ة", "ه", text)

    text = re.sub("گ", "ك", text)

    text = re.sub(r'[a-zA-Z?]', '', text).strip()

    return text







def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)





def remove_nbr(text):

    string_no_numbers = re.sub("\d+", " ", text)

    return string_no_numbers





def stop_words(text):

    

    stop_word = open("../input/stopwords/stpo_words.txt","r").read().split()

    table = []

    wrd = word_tokenize(text)

    for i in wrd:

        if i not in stop_word:

            table.append(i)

    filtre=" ".join(table)

    return filtre



def remove_punctuation(text):

    my_punctuations = string.punctuation + "،" + "؛" + "؟" + "«" + "»"

    translator = str.maketrans('', '', my_punctuations)

    sup_ponctuation = text.translate(translator)

    return sup_ponctuation





def remove_links(text):

    sup_liens=re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text, flags=re.MULTILINE)

    return sup_liens





def remove_repeating_char(text):

    sup_cara_repete= re.sub(r'(.)\1+', r'\1\1', text)

    return sup_cara_repete









def remove_Arobaze(text):

    sentence = re.sub(r'@[A-Za-z0-9+\u0627-\u064a\_\-]+', '', text)

    return sentence





def delete_Hashtag(text):

    sentence1 = re.sub(r'#[A-Za-z0-9+\u0627-\u064a\_\-\u0623\u0625\u0624\u0626]+', '', text)

    return sentence1











def preparation_Data(review):



    sentance=[]

    for index, r in review.iterrows():

        txt = remove_links(r['tweet'])

        txt = remove_Arobaze(txt)

        txt = delete_Hashtag(txt)

        txt = remove_punctuation(txt)

        txt = remove_repeating_char(txt)

        txt = remove_emoji(txt)

        txt = stop_words(txt)

        txt = normalize(txt)

        txt = remove_nbr(txt)

        txt = stop_words(txt)

        txt = clean_twt(txt)





        if r['sentiment'] == 'positive':

            sentance.append(['1', txt])

        elif r['sentiment'] == 'negative':

            sentance.append(['-1', txt])

        elif r['sentiment'] == 'neutral':

            sentance.append(['0', txt])



    sentence_df=DataFrame(sentance,columns=['sentiment', 'tweet'])

    return sentence_df





dataTrain = preparation_Data(dataTrain)

dataTest = preparation_Data(dataTest)

dataTest.info()

print("terminer")


import collections

import matplotlib.pyplot as plt

import gensim

import pandas as pd

import keras

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from keras.layers.core import Dropout, Activation, ActivityRegularization

from keras.layers.normalization import BatchNormalization

from keras.layers.recurrent import LSTM 

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.layers import  Flatten

from keras.layers.wrappers import Bidirectional

from keras.regularizers import l2, l1

from pandas import DataFrame

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.sequence import pad_sequences

from keras import models

from keras import layers

import numpy as np





# train model 



my_df_train=dataTrain

y_train = my_df_train.sentiment

x_train = my_df_train.tweet





my_df_test=dataTest

y_test = my_df_test.sentiment

x_test = my_df_test.tweet



print("x train",x_train.shape)

print("y train",y_train.shape)

print("x test",x_test.shape)

print("y test",y_test.shape)



print('-----------------------------------')

print(x_train.describe())

print('-----------------------------------')

print(y_train.describe())



tk = Tokenizer(num_words=900000,

               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',

               lower=True,

               split=" ")



tk.fit_on_texts(x_train)



print('-----------------------------------')

print('Top 10 most common words are:', collections.Counter(tk.word_counts).most_common(100))



word_index = tk.word_index



print('-----------------------------------')

print('Found %s unique tokens.' % len(word_index))





X_train_seq = tk.texts_to_sequences(x_train)

X_test_seq = tk.texts_to_sequences(x_test)









print('{} -- is converted to -- {}'.format(x_train[15], X_train_seq[15]))



seq_lengths = x_train.apply(lambda x: len(x.split(' ')))

print(seq_lengths.describe())



length = []

for x in x_train:

    length.append(len(x.split()))

max_len=max(length)

print('-----------------------------------')

print("max length",max(length))



MAX_LEN = max_len

X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN)

X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN)



print('-----------------------------------')

print('{} -- is converted to -- {}'.format(X_train_seq[15], X_train_seq_trunc[15]))









le = LabelEncoder()

y_train_le = le.fit_transform(y_train)

y_test_le = le.transform(y_test)

y_train_oh = to_categorical(y_train_le)

y_test_oh = to_categorical(y_test_le)



X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.2, random_state=3)



print('-----------------------------------')

print('Shape of train set:',X_train_emb.shape)

print('Shape of validation set:',X_valid_emb.shape)



model = gensim.models.Word2Vec.load('../input/twt-sg-100/tweets_sg_100')







embeddings_index = {}

for w in model.wv.vocab.keys():

    embeddings_index[w] = model.wv[w]









embedding_matrix = np.zeros((len(word_index) + 1, 100))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector



# NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary

# EPOCHS = 150

# BATCH_SIZE = 64



NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary

EPOCHS = 300

BATCH_SIZE = 50





model = models.Sequential()

model.add(layers.Embedding(len(word_index) + 1, 100, input_length=MAX_LEN, weights=[embedding_matrix], trainable=False))

model.add(Dropout(0.5))



model.add(Bidirectional(LSTM(32,return_sequences=True, input_shape=(100, 100), dropout=0.5, recurrent_dropout=0.5)))



model.add(LSTM(64, dropout=0.5 ,return_sequences=True, recurrent_dropout=0.5))

model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5)))



# model.add(Dropout(0.4))

model.add(BatchNormalization())



# ActivityRegularization(l1=0.01, l2=0.001)

model.add(Activation('sigmoid'))

model.add(layers.Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



filepath="weights.best1.hdf5"



checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="val_accuracy", mode="max", patience=300)





callbacks_list = [checkpoint, early] #early



history=model.fit(X_train_emb, y_train_emb, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, validation_data=(X_valid_emb, y_valid_emb), callbacks=callbacks_list)







# evaluation of model

test_model = model.evaluate(X_test_seq_trunc, y_test_oh)

print("Accuracy: %.2f%%" % (test_model[1]*100))

model.save_weights('model111.h5')





print(history.history.keys())

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()