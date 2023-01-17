import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('../input/spam.csv',encoding='latin-1')



data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

data = data.rename(columns={"v1":'label', "v2":'text'})

print(data.head())

tags = data["label"]

texts = data["text"]
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from sklearn.preprocessing import LabelEncoder

from keras import metrics



num_max = 1000

le = LabelEncoder()

tags = le.fit_transform(tags)

tok = Tokenizer(num_words=num_max)

tok.fit_on_texts(texts)

mat_texts = tok.texts_to_matrix(texts,mode='count')



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(texts,tags, test_size = 0.3)

mat_texts_tr = tok.texts_to_matrix(x_train,mode='count')

mat_texts_tst = tok.texts_to_matrix(x_test,mode='count')



max_len = 100

x_train = tok.texts_to_sequences(x_train)

x_test = tok.texts_to_sequences(x_test)

cnn_texts_mat = sequence.pad_sequences(x_train,maxlen=max_len)

max_len = 100

cnn_texts_mat_tst = sequence.pad_sequences(x_test,maxlen=max_len)
def get_simple_model():

    model = Sequential()

    model.add(Dense(512, activation='relu', input_shape=(num_max,)))

    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc',metrics.binary_accuracy])

    return model
def check_model(model,xtr,ytr,xts,yts):

    model.fit(xtr,ytr,batch_size=32,epochs=10,verbose=1,validation_split=0.3)

    print(' ')

    model.evaluate(xts,yts)
m = get_simple_model()

check_model(m,mat_texts_tr,y_train,mat_texts_tst,y_test)
def get_cnn_model_v1():   

    model = Sequential()

    model.add(Embedding(1000,20,input_length=max_len))

    model.add(Dropout(0.2))

    model.add(Conv1D(64,3,padding='valid',activation='relu',strides=1))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(256))

    model.add(Dropout(0.2))

    model.add(Activation('relu'))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc',metrics.binary_accuracy])

    return model
m = get_cnn_model_v1()

check_model(m,cnn_texts_mat,y_train,cnn_texts_mat_tst,y_test)
def get_cnn_model_v2():   

    model = Sequential()

    model.add(Embedding(1000,50,input_length=max_len))

    model.add(Dropout(0.2))

    model.add(Conv1D(64,3,padding='valid',activation='relu',strides=1))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(256))

    model.add(Dropout(0.2))

    model.add(Activation('relu'))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc',metrics.binary_accuracy])

    return model
m = get_cnn_model_v2()

check_model(m,cnn_texts_mat,y_train,cnn_texts_mat_tst ,y_test)
def get_cnn_model_v3():    

    model = Sequential()

    model.add(Embedding(1000,20,input_length=max_len))

    model.add(Dropout(0.2))

    model.add(Conv1D(256,3,padding='valid',activation='relu',strides=1))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(256))

    model.add(Dropout(0.2))

    model.add(Activation('relu'))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc',metrics.binary_accuracy])

    return model


m = get_cnn_model_v3()

check_model(m,cnn_texts_mat,y_train,cnn_texts_mat_tst ,y_test)
from keras.layers import LSTM



max_features = 5000

maxlen = 400

batch_size = 32

embedding_dims = 50

filters = 250

kernel_size = 3

hidden_dims = 250

epochs = 2
def get_cnn_model_v4():    

    model = Sequential()

    model.add(Embedding(max_features, 128))

    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc',metrics.binary_accuracy])



    print('Train...')

    return model
m = get_cnn_model_v4()

check_model(m,cnn_texts_mat,y_train,cnn_texts_mat_tst ,y_test)