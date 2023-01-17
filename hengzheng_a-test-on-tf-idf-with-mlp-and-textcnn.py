import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Input, Activation, Conv1D
from keras.layers import Dropout, MaxPooling1D, Flatten, Concatenate, Reshape
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils

import re
import random
import os
print(os.listdir("../input"))
train_df = pd.read_json("../input/train.json")
train_df.head()
test_df = pd.read_json("../input/test.json")
test_df.head()
# concat all

test_df["cuisine"] = "unknown"

df = pd.concat([train_df, test_df], ignore_index=True)
df.head()
print(df.shape)
df.ingredients = df.ingredients.apply(lambda x: (" ".join(x)).lower())
df.ingredients = df.ingredients.apply(lambda x: re.sub(r'[^\w\d ,]', '', x))
df.head()
train_set = df[df.cuisine != 'unknown']
test_set = df[df.cuisine == 'unknown']
tfidf = TfidfVectorizer(binary=True)

x = tfidf.fit_transform(train_set.ingredients).todense()
x_test = tfidf.transform(test_set.ingredients).todense()
print(x.shape, x_test.shape)
lb = LabelEncoder()
y = lb.fit_transform(train_set.cuisine)
y = np_utils.to_categorical(y)
print(y.shape)
seed = 29
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
MLP = Sequential()
MLP.add(Dense(512, input_shape=(3073, ), activation='relu'))
MLP.add(Dropout(0.5))
MLP.add(Dense(y_train.shape[1], activation='softmax'))
MLP.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

MLP.summary()
file_path = "mlp.hdf5"
check_point = ModelCheckpoint(file_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
early_stop = EarlyStopping(monitor="val_acc", mode="max", patience=5)

mlp_history = MLP.fit(x_train, 
                      y_train, 
                      batch_size=128, 
                      epochs=50, 
                      validation_data=(x_val, y_val),
                      callbacks=[check_point, early_stop])
mlp_best = load_model('mlp.hdf5')
inp = Input(shape=(3073,), dtype='float32')
reshape = Reshape(target_shape=(7,439))(inp)

stacks = []
for kernel_size in [2, 3, 4]:
    conv = Conv1D(128, kernel_size, padding='same', activation='relu', strides=1)(reshape)
    pool = MaxPooling1D(pool_size=3)(conv)
    drop = Dropout(0.5)(pool)
    stacks.append(drop)

merged = Concatenate()(stacks)
flatten = Flatten()(merged)
drop = Dropout(0.5)(flatten)
outp = Dense(y_train.shape[1], activation='softmax')(drop)

TextCNN = Model(inputs=inp, outputs=outp)
TextCNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

TextCNN.summary()
file_path = "textcnn.hdf5"
check_point = ModelCheckpoint(file_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
early_stop = EarlyStopping(monitor="val_acc", mode="max", patience=5)

textcnn_history = TextCNN.fit(x_train, 
                              y_train, 
                              batch_size=128, 
                              epochs=50, 
                              validation_data=(x_val, y_val),
                              callbacks=[check_point, early_stop])
textcnn_best = load_model('textcnn.hdf5')
mlp_pred = mlp_best.predict(x_test)
textcnn_pred = textcnn_best.predict(x_test)

ensemble_pred = (mlp_pred + textcnn_pred).argmax(axis=1)
final = lb.inverse_transform(ensemble_pred)
sub = pd.DataFrame({'id': test_df.id, 'cuisine': final})
sub.to_csv('tfidf-textcnn.csv', index = False)
!head tfidf-textcnn.csv