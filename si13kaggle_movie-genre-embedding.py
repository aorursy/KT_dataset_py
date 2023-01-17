# чистит сессию в Keras и TF

def reset_tf_session():

    curr_session = tf.get_default_session()

    # close current session

    if curr_session is not None:

        curr_session.close()

    # reset graph

    K.clear_session()

    # create new session

    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    s = tf.InteractiveSession(config=config)

    K.set_session(s)

    return s
import matplotlib.pyplot as plt

plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import sklearn

import tensorflow as tf

import keras

from keras import backend as K

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

print(tf.__version__)

print(keras.__version__)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# считаем данные

train_data_np = pd.read_csv("../input/train_data.txt",delimiter=' ::: ',header=None,names=['id','title','genre','desc'])

predict_data_np = pd.read_csv("../input/test_data.txt",delimiter=' ::: ',header=None,names=['id','title','desc'])



train_data_np.shape

predict_data_np.head()

genres = np.sort(train_data_np.genre.unique())

genres = genres.tolist()

# конвертируем метки в one-hot

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

lb.fit(genres)

y = lb.transform(train_data_np.genre)

print(lb.classes_)

print(train_data_np.genre[0])

y[0]
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split



desc = train_data_np['desc'].values



#tokenizer = Tokenizer(num_words=20000) # ML8: 10000,20000 получше, None - похуже

tokenizer = Tokenizer(num_words=None,lower=True) # ML9: 10000,20000 получше, None - похуже?



tokenizer.fit_on_texts(desc)

# remove frequent words - вроде ничего не дает, отключил для LSTM

deleted = 0

high_count_words = [w for w,c in tokenizer.word_counts.items() if c > 10.0*train_data_np.shape[0]]

for w in high_count_words:

    del tokenizer.word_index[w]

    del tokenizer.word_docs[w]

    del tokenizer.word_counts[w]

    deleted+=1

    print("Delete ", w)

print("Delete ", deleted, " words from tokenizer")



desc_train, desc_test, y_train, y_test = train_test_split(desc, y, test_size=0.05, random_state=1000)



X_train = tokenizer.texts_to_sequences(desc_train)

X_test = tokenizer.texts_to_sequences(desc_test)







vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index





print(desc_train[2])

print(X_train[2])

# maxlen parameter to specify how long the sequences should be. This cuts sequences that exceed that number

from keras.preprocessing.sequence import pad_sequences



maxlen = 100



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



print(X_train[0, :])
# Модель для 8 урока, без рекуррентных сетей. 

# Переобучается и не разгонятеся более 0.6 на validation accuracy  

from keras.models import Sequential

from keras import layers



def make_model_8():

    s = reset_tf_session()

    embedding_dim = 50 # initial 50



    model = Sequential()

    model.add(layers.Embedding(input_dim=vocab_size, 

                               output_dim=embedding_dim, 

                               input_length=maxlen))

    #model.add(layers.Flatten())

    #model.add(keras.layers.Conv1D(30,1,activation="relu"))

    model.add(keras.layers.GlobalAveragePooling1D())

    #model.add(layers.Dense(500, activation='relu')) # initial: не было

    model.add(layers.Dense(32, activation='relu')) # initial: 32, val_acc=0.55

    model.add(layers.Dense(len(genres), activation='softmax'))

    model.compile(optimizer='adam',

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    model.summary()

    return model

# ML9: добавляем LSTM

def make_model_9():

    s = reset_tf_session()

    embedding_dim = 50 # initial 50



    model = Sequential()

    model.add(layers.Embedding(input_dim=vocab_size, 

                               output_dim=embedding_dim, 

                               input_length=maxlen))

   

    #model.add(keras.layers.GlobalAveragePooling1D())

    # dropout - немного замедляет переобучение: val_acc 0.44 -> 0.54

    #model.add(layers.LSTM(32, return_sequences=True,dropout=0.7,recurrent_dropout=0.5)) 

    #model.add(keras.layers.CuDNNLSTM(4, return_sequences=True))

    model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True, dropout=0.7,recurrent_dropout=0.7)))

    model.add(layers.Flatten())



    #model.add(layers.Dense(500, activation='relu')) # initial: 32



    model.add(layers.Dense(len(genres), activation='softmax'))

    model.compile(optimizer='adam',

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    model.summary()

    return model
#model = make_model_9()

model = make_model_8()



history = model.fit(X_train, y_train,

                    epochs=20,

                    verbose=True,

                    validation_data=(X_test, y_test),

                    batch_size=1000)

plot_history(history)
plot_history(history)
y_predict_train = model.predict_proba(X_train)
# проверка

T=201

y_predict = y_predict_train

print(y_predict[T,:])

y_predict_max = np.argmax(y_predict[T,:])

print(y_predict_max, lb.classes_[y_predict_max], y_predict[T,y_predict_max])

print( np.argmax(y_train[T]),y_train[T])

print(train_data_np.genre[T])



# применяем модель к predict_data_np id, title, desc

desc_predict = predict_data_np['desc'].values

X_predict = tokenizer.texts_to_sequences(desc_predict)

X_predict = pad_sequences(X_predict, padding='post', maxlen=maxlen)

y_predict = model.predict_proba(X_predict)



T=100

print(predict_data_np.iloc[T])

y_predict_max = np.argmax(y_predict[T,:])

print(y_predict_max, lb.classes_[y_predict_max], y_predict[T,y_predict_max])
y_predict.shape[0]

#печатаем ответ



genres_predict = []

ids = []

for i in range(y_predict.shape[0]):

#for i in range(10):

    y_predict_max = np.argmax(y_predict[i,:])

    #print (i,y_predict_max,lb.classes_[y_predict_max])

    genres_predict.extend([lb.classes_[y_predict_max]])



submission = pd.DataFrame({'id':predict_data_np['id'].values,

                           'genre':genres_predict,

                           'title':predict_data_np['title'].values},

                          columns=['id', 'genre','title'])

submission.to_csv('submission.csv', index=False,columns=['id', 'genre'])

print('Save submit')

submission.head()

with open("submission.csv") as myfile:

    head = [next(myfile) for x in range(10)]

print(head)