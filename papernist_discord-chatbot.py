!pip install sklearn_crfsuite
!pip install https://github.com/PyThaiNLP/pythainlp/archive/dev.zip
!pip install fastai==1.0.46
!pip install emoji
!pip install oauth2client gspread
import tensorflow as tf
import keras

import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from pythainlp.util import normalize

import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import jaccard_score

from scipy import stats

import seaborn as sns

import skimage
from skimage.transform import rotate

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, UpSampling2D, GlobalMaxPool2D, GlobalAveragePooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Input, Embedding, LSTM, RNN
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import NASNetMobile, Xception, DenseNet121, MobileNetV2, InceptionV3, InceptionResNetV2, vgg16, resnet50, inception_v3, xception, DenseNet201
from tensorflow.keras.applications.vgg16 import VGG16


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datetime import datetime

import pickle
import numpy as np
import os
import cv2
import pandas as pd
# import imutils
import random
from PIL import Image
import matplotlib.pyplot as plt
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pprint
import requests
url = 'https://docs.google.com/spreadsheets/d/13VUK5UKiwqcs8b2cB7Avk7V2ZD3ppB-0KF4cBzyniCQ/edit?usp=sharing'
def get_data():
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('../input/qadataset/credentials.json', scope)
    client = gspread.authorize(credentials)

    sheet = client.open_by_url(url)

    worksheet = sheet.get_worksheet(0)

    X = []
    Y = []
    for i in worksheet.get_all_values()[1:]:
        X.append(i[1])
        Y.append(i[2])
    
    return X, Y

get_data()
word2index = {}
index2word = {}

max_len = 50
vocab_size = 30
word_embedding_size = 128
sentence_embedding_size = 128

word2index['<EOS>'] = 0
index2word[0] = '<EOS>'
word2index[' '] = 1
index2word[1] = ' '
def get_vocab_size():
#     return vocab_size
    return len(word2index)

def add_word(word):
    if not(word in word2index.keys()):
        current_index = len(word2index)
        word2index[word] = current_index
        index2word[current_index] = word
# from pythainlp import word_vector
# model = word_vector.get_model()

# thai2dict = {}
# for word in model.index2word:
#     print(word, model[word])
#     thai2dict[word] = model[word]
# thai2vec = pd.DataFrame.from_dict(thai2dict,orient='index')
with tf.device('/device:GPU:0'):
    def get_model():
        inputs = Input(shape=(max_len,))
        inputs_ = Input(shape=(max_len,))

        x = Embedding(output_dim=word_embedding_size, input_dim=get_vocab_size(), input_length=max_len)(inputs)
        y = Embedding(output_dim=word_embedding_size, input_dim=get_vocab_size(), input_length=max_len)(inputs_)
        
        x = LSTM(sentence_embedding_size)(x)
        x = BatchNormalization()(x)
        
        y = LSTM(sentence_embedding_size)(y)
        y = BatchNormalization()(y)
        
        x = concatenate([x,y])
        x = Dense(sentence_embedding_size * 2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(get_vocab_size())(x)
        x = BatchNormalization()(x)
        outputs = Activation('softmax')(x)

        model = Model(inputs=[inputs, inputs_], outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         model.summary()

        return model

get_model().summary()
plot_model(get_model(), show_shapes=True)
def encoder(input_sentences):
    input_sentences = input_sentences.copy()
    list_sentence = []
    for sentence in input_sentences:
        inputs = []
        for i in sentence:
            inputs.append(word2index[i])

        inputs = np.array(inputs)
        
        list_sentence.append(inputs)

    return list_sentence
def decoder(input_index):
    input_index = input_index.copy()
    list_sentence = []
    for index in input_index:
        index = np.delete(index, np.argwhere(index == 0))
        inputs = []
        for i in index:
            inputs.append(index2word[i])
            
        inputs = np.array(inputs)
        
        list_sentence.append(inputs)
        
    return list_sentence
X = ['สวัสดี','เป็นไงบ้าง', 'ทำอะไรอยู่','ชอบแมวไหม','หมามีกี่ขา','แมวเหมือนหมาไหม']
Y = ['สวัสดีครับ','สบายดีครับ','เรื่องของผม','ไม่ชอบ','สี่ขา','ไม่เหมือน']
X_, Y_ = get_data()
X = X + X_
Y = Y + Y_
X[:10], Y[:10]
def preprocessing(inputs):
    inputs = inputs.copy()
    for i, words in enumerate(inputs):
        word = normalize(words)
        word = word_tokenize(word)
        for j, w in enumerate(word):
            word[j] = correct(w)
            add_word(word[j])
            
        inputs[i] = word

#     inputs = np.array(pad_sequences(encoder(inputs), maxlen=max_len))
    
    return encoder(inputs)
def postprocessing(inputs):
    sentence = decoder(inputs.copy())
    list_string = []
    for words in sentence:
        string = ''
        for word in words:
            string += word
        list_string.append(string)
    
    return list_string
def prepare_data(X, Y):
    X = preprocessing(X.copy())
    Y = preprocessing(Y.copy())
    x_train = []
    x1_train = []
    y_train = []
    for count in range(len(X)):
        x = X[count]
        y = Y[count]
        x_train.append(x)
        x1_train.append([0])
        y_train.append(to_categorical(y[0], num_classes=get_vocab_size()))
        for i in range(len(y)-1):
            x_train.append(x)
            x1_train.append(y[:i+1])
            y_train.append(to_categorical(y[i+1], num_classes=get_vocab_size()))
        
        x_train.append(x)
        x1_train.append(y[:])
        y_train.append(to_categorical(0, num_classes=get_vocab_size()))
        
    x_train = np.array(pad_sequences(x_train, maxlen=max_len))
    x1_train = np.array(pad_sequences(x1_train, maxlen=max_len))
    y_train = np.array(y_train)
    
    return x_train, x1_train, y_train
x_data, a_data, y_data = prepare_data(X,Y)
x_data.shape, a_data.shape, y_data.shape
# X.shape, Y.shape
x_train, a_train, y_train = x_data, a_data, y_data
model = get_model()
batch_size = 20
epoch = 10
for i in range(50):
    model.fit([x_train, a_train], y_train, batch_size=batch_size, epochs=epoch, verbose=0)
    model.evaluate([x_train, a_train], y_train)
model.save('model.h5')
pickle.dump(word2index, open('word2index.dict', 'wb'))
pickle.dump(index2word, open('index2word.dict', 'wb'))
pickle.dump(max_len, open('max_len.txt', 'wb'))
pred = model.predict([x_train, a_train])

for i,j in zip(pred[:10], y_train[:10]):
    print(np.argmax(i), np.argmax(j))
    
model.evaluate([x_train, a_train], y_train)
# def is_understand(inputs):
def messange_to_bot(sentences, model_chatbot):
    sentence = ''
    word = preprocessing([sentences])
    continued = True
    answer_index = [[]]
    while continued:
        x = np.array(pad_sequences(word, maxlen=max_len))
        a = np.array(pad_sequences(answer_index, maxlen=max_len))
        predict = np.argmax(model_chatbot.predict([x, a]), axis=1)[0]
        
        if predict == 0:
            continued = False
            break
        
        answer_index[0].append(predict)
        
        sentence += index2word[predict]
        
    return sentence

def talk_with_bot():
    while True:
        text = input('ข้อความของคุณ : ')
        if text == 'quit':
            break
        print('Bot :', messange_to_bot(text, model))
talk_with_bot()
import pythainlp
pythainlp.corpus.thai_words()