# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from keras.preprocessing.sequence import pad_sequences

print(os.listdir("../input"))
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.layers import Conv1D,MaxPooling1D,Dense,Flatten,Dropout,Input,Concatenate
from keras.models import Sequential,Model,load_model
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler,EarlyStopping


import keras
# Any results you write to the current directory are saved as output.
X_train = pd.read_json("../input/train.json")
X_test = pd.read_json("../input/test.json")
#X_train
Y_train = X_train['cuisine']
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_train = keras.utils.to_categorical(Y_train)

def generate_text(data):
    ingredients = data['ingredients']
    text_data = list()
    for doc in ingredients:
        str_arr = list()
        for s in doc:
            str_arr.append(s.replace(' ', ''))
        text_data.append(" ".join(str_arr).lower())
    # text_data = [" ".join(doc).lower() for doc in ingredients]
    return text_data

X_train_text = generate_text(X_train)
X_test_text = generate_text(X_test)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train_text)
X_test_counts = count_vect.transform(X_test_text)
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False)
X_train_tf = tf_transformer.fit_transform(X_train_counts)

#X_train_tf = np.expand_dims(X_train_tf, axis=2) # reshape (569, 30) to (569, 30, 1) 
#np.reshape(X_train_tf, ( X_train_tf.shape[0], X_train_tf.shape[1],1))
X_train_tf = X_train_tf.reshape((X_train_tf.shape[0], X_train_tf.shape[1],1))
#b = np.zeros((X_train_tf.shape[0], X_train_tf.shape[1], X_train_tf.max() + 1))



print(X_train_tf[0])

X_test_df  = tf_transformer.transform(X_test_counts)
X_test_df.shape
inp = Input(shape=(6782,), dtype='float32')
reshape = Reshape(target_shape=(6782,1))(inp)

conc=[]
normal = BatchNormalization()(reshape)
conv = Conv1D(128, 3, padding='same', activation='relu', strides=1)(normal)
drop = Dropout(0.75)(conv)
conc.append(drop)


normal1 = BatchNormalization()(drop)
conv1 = Conv1D(128, 4, padding='same', activation='relu', strides=1)(normal1)
drop1 = Dropout(0.75)(conv1)


conc.append(drop1)
    
    

concatenate = Concatenate()(conc)
flatten = Flatten()(concatenate)
drop = Dropout(0.75)(flatten)
outp = Dense(20, activation='softmax')(drop)

model = Model(inputs=inp, outputs=outp)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=2, verbose=1)


history = model.fit(X_train_tf, Y_train, epochs=15, callbacks=[annealer,lr_reduce],batch_size=128, validation_split=0.1)

import matplotlib.pyplot as plt

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_test = model.predict(X_test_df)
y_predict = le.inverse_transform([np.argmax(pred) for pred in y_test])
y_predict
test_id = [doc for doc in X_test['id']]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_predict}, columns=['id', 'cuisine'])
sub.to_csv('output.csv', index=False)
