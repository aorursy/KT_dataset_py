from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout,GRU

from sklearn.model_selection import train_test_split

import re#regular expression

import numpy as np 

import pandas as pd

from nltk.corpus import stopwords

from nltk import word_tokenize

from tensorflow.keras import backend

import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.utils import shuffle

import keras

import tensorflow as tf
config = tf.compat.v1.ConfigProto() 

config.gpu_options.allow_growth=True

sess = tf.compat.v1.Session(config=config)



#keras.backend.set_session(sess)



tf.compat.v1.keras.backend.set_session( sess)



def _get_available_devices():

    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()

    return [x.name for x in local_device_protos]

print( _get_available_devices())
df = pd.read_csv("../input/train.csv")

df_predict=pd.read_csv("../input/test.csv")



#text, label

print("\nDATAFRAME INFORMATION:\n")

print(df.info())



fields= ['text','label'] 



df=pd.read_csv("../input/train.csv", usecols=fields)

df_predict=pd.read_csv("../input/test.csv", usecols=['text'])
print("\nPRODUCT VALUE COUNT:\n")

print(df["label"].value_counts())

print("\nDATAFRAME HEAD:\n")

print(df.head())
STOPWORDS=stopwords.words('english')

stopwords_extra=['bank', 'america', 'x/xx/xxxx', '00']

STOPWORDS.extend(stopwords_extra)



remove_caracteres = re.compile('[^0-9a-z #+_]')

replace_espaco = re.compile('[/(){}\[\]\|@,;]')

df = df.reset_index(drop=True)

Y = pd.get_dummies(df["label"]).values

print("shape Y", Y.shape)

class_name = [x for x in range(11)]

#for i in range(100):

#    for j in range(10):

#        if(Y[i,j]==1):

#            class_name[j]=df.iloc[i]['label']

np.save('class_name',class_name)  

print(class_name)
X, Y = shuffle(df['text'] , Y)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state = 42)







def pre_processamento(text):

    text = text.lower()

    text = remove_caracteres.sub('', text)

    text = replace_espaco.sub(' ', text)

    text = text.replace('x', '')

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)

    return text



X_train = X_train.apply(pre_processamento)

X_test = X_test.apply(pre_processamento)



#print(type(X_train))

#print(type(df_predict))



X_predict=df_predict['text'].apply(pre_processamento)
n_max_palavras = 5000

tamanho_maximo_sent = 250



tokenizer = Tokenizer(num_words=n_max_palavras, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(df['text'].values)

word_index = tokenizer.word_index

print(' %s tokens unicos.' % len(word_index))





X_train = tokenizer.texts_to_sequences(X_train.values)

X_train = pad_sequences(X_train, maxlen=tamanho_maximo_sent)

print("shape X_train", X_train.shape)



X_test = tokenizer.texts_to_sequences(X_test.values)

X_test = pad_sequences(X_test, maxlen=tamanho_maximo_sent)

print("shape X_test", X_test.shape)



X_predict = tokenizer.texts_to_sequences(X_predict.values)

X_predict = pad_sequences(X_predict, maxlen=tamanho_maximo_sent)

print("shape X_predict", X_predict.shape)







np.save('X_test', X_test)

np.save('Y_test', Y_test)

np.save('X_predict', X_predict)

#model = Sequential()

#model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))

#model.add(SpatialDropout1D(0.2))

#model.add(GRU(units=100, return_sequences=True))

#model.add(Dropout(0.2))

#model.add(GRU(100))

#model.add(Dropout(0.2))

#model.add(Dense(10, activation="softmax"))

#model.add(Dense(11, activation="softmax"))

#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



#print(model.summary())

#history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05,shuffle=True)

embedding_dimensions = 100

epochs = 10

#batch_size = 512

batch_size = 128



model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(GRU(units=100, return_sequences=True))

model.add(Dropout(0.1))

model.add(GRU(100))

model.add(Dropout(0.1))

#model.add(Dense(10, activation="softmax"))

model.add(Dense(11, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())

history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05,shuffle=True)



model.save('my_user_model.h5')





fig1 = plt.figure()

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves :RNN - GRU',fontsize=16)

plt.show()



scores = model.evaluate(X_test, Y_test, verbose=0)

print("\nAccuracy: %.2f%%" % (scores[1]*100))
embedding_dimensions = 100

epochs = 10

#batch_size = 512

batch_size = 128



model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(GRU(units=100, return_sequences=True))

model.add(Dropout(0.2))

model.add(GRU(100))

model.add(Dropout(0.2))

#model.add(Dense(10, activation="softmax"))

model.add(Dense(11, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())

history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05,shuffle=True)



model.save('my_user_model.h5')





fig1 = plt.figure()

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves :RNN - GRU',fontsize=16)

plt.show()



scores = model.evaluate(X_test, Y_test, verbose=0)

print("\nAccuracy: %.2f%%" % (scores[1]*100))
embedding_dimensions = 200

epochs = 10

#batch_size = 512

batch_size = 128



model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(GRU(units=100, return_sequences=True))

model.add(Dropout(0.1))

model.add(GRU(100))

model.add(Dropout(0.1))

#model.add(Dense(10, activation="softmax"))

model.add(Dense(11, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())

history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,shuffle=True)



model.save('my_user_model.h5')





fig1 = plt.figure()

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves :RNN - GRU',fontsize=16)

plt.show()



scores = model.evaluate(X_test, Y_test, verbose=0)

print("\nAccuracy: %.2f%%" % (scores[1]*100))
embedding_dimensions = 300

epochs = 10

#batch_size = 512

batch_size = 128



model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(GRU(units=100, return_sequences=True))

model.add(Dropout(0.1))

model.add(GRU(100))

model.add(Dropout(0.1))

#model.add(Dense(10, activation="softmax"))

model.add(Dense(11, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())

history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05,shuffle=True)



model.save('my_user_model.h5')





fig1 = plt.figure()

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves :RNN - GRU',fontsize=16)

plt.show()



scores = model.evaluate(X_test, Y_test, verbose=0)

print("\nAccuracy: %.2f%%" % (scores[1]*100))
embedding_dimensions = 500

epochs = 6

#batch_size = 512

batch_size = 128



model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(GRU(units=100, return_sequences=True))

model.add(Dropout(0.1))

model.add(GRU(100))

model.add(Dropout(0.1))

#model.add(Dense(10, activation="softmax"))

model.add(Dense(11, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())

history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05,shuffle=True)



model.save('my_user_model.h5')





fig1 = plt.figure()

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves :RNN - GRU',fontsize=16)

plt.show()



scores = model.evaluate(X_test, Y_test, verbose=0)

print("\nAccuracy: %.2f%%" % (scores[1]*100))
embedding_dimensions = 1000

epochs = 10

#batch_size = 512

batch_size = 256



model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(GRU(units=100, return_sequences=True))

model.add(Dropout(0.2))

model.add(GRU(100))

model.add(Dropout(0.2))

#model.add(Dense(10, activation="softmax"))

model.add(Dense(11, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())

history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.04,shuffle=True)



model.save('my_user_model.h5')





fig1 = plt.figure()

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves :RNN - GRU',fontsize=16)

plt.show()



scores = model.evaluate(X_test, Y_test, verbose=0)

print("\nAccuracy: %.2f%%" % (scores[1]*100))
embedding_dimensions = 250

epochs = 8

#batch_size = 512

batch_size = 128



model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(GRU(units=100, return_sequences=True))

model.add(Dropout(0.1))

model.add(GRU(100))

model.add(Dropout(0.1))

#model.add(Dense(10, activation="softmax"))

model.add(Dense(11, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())

history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05,shuffle=True)



model.save('my_user_model.h5')





fig1 = plt.figure()

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves :RNN - GRU',fontsize=16)

plt.show()



scores = model.evaluate(X_test, Y_test, verbose=0)

print("\nAccuracy: %.2f%%" % (scores[1]*100))
embedding_dimensions = 400

epochs = 10

#batch_size = 512

batch_size = 128



model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(GRU(units=100, return_sequences=True))

model.add(Dropout(0.1))

model.add(GRU(100))

model.add(Dropout(0.1))

#model.add(Dense(10, activation="softmax"))

model.add(Dense(11, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())

history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05,shuffle=True)



model.save('my_user_model.h5')





fig1 = plt.figure()

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves :RNN - GRU',fontsize=16)

plt.show()



scores = model.evaluate(X_test, Y_test, verbose=0)

print("\nAccuracy: %.2f%%" % (scores[1]*100))
embedding_dimensions = 350

epochs = 10

#batch_size = 512

batch_size = 128



model = Sequential()

model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(GRU(units=100, return_sequences=True))

model.add(Dropout(0.2))

model.add(GRU(100))

model.add(Dropout(0.2))

#model.add(Dense(10, activation="softmax"))

model.add(Dense(11, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())

history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05,shuffle=True)



model.save('my_user_model.h5')





fig1 = plt.figure()

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves :RNN - GRU',fontsize=16)

plt.show()



scores = model.evaluate(X_test, Y_test, verbose=0)

print("\nAccuracy: %.2f%%" % (scores[1]*100))
yhat = model.predict_classes(X_predict, verbose=0)
print(yhat[:10], X_predict[:10])
print(len(yhat))
print(len(X_predict), X_predict.shape)
predict_result=[]

for i in range (len(yhat)):

    predict_result.append([i, yhat[i]])

    

import datetime

## save result to csv

df=pd.DataFrame(predict_result)

dt=datetime.datetime.now()

df.to_csv(path_or_buf='userclassification{}.csv'.format(dt.strftime("%Y%m%d%H")), index = False,header=None,encoding='ascii')  

    
