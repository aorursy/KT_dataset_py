# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.callbacks import TensorBoard

from keras.models import Sequential, load_model

from keras.layers import LSTM, Dense, Embedding

from keras.optimizers import Nadam

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load pandas frame and output examples

df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

df.head(5)
# Setup TensorBoard

tBoard = TensorBoard(log_dir='graphs/', histogram_freq=0,

                     write_graph=True, write_images=True)

# define data

chance_admit = df[df.columns[8]].values

features = df.drop(df.columns[[0, 8]], axis=1).values

categorical = []

print("Raw chances:\t",chance_admit[:10])

print("Raw features:\t",features[:10])



treshold_admittable = 0.8



for row in chance_admit:

    if row >= treshold_admittable:

        categorical.append(1)

    else:

        categorical.append(0)

categorical = to_categorical(categorical, num_classes=2)

print("Categorical chances:\t",categorical[:10])
# split train/test

X_train, X_test, y_train, y_test = train_test_split(features, categorical, test_size=0.2, random_state=42)
# define DNN

model=None



model = Sequential()

model.add(Embedding(df['GRE Score'].max()+1, 80, input_length=7))

model.add(LSTM(80, dropout=0.3, recurrent_dropout=0.4, return_sequences=True))

model.add(LSTM(80))

model.add(Dense(2, activation='softmax', kernel_regularizer='l1'))

model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.0001), metrics=['categorical_accuracy'])

print("Model summary:")

model.summary()

# train neural net

history = model.fit(X_train, y_train, batch_size=20, epochs=20,validation_data=(X_test,y_test), callbacks=[tBoard])

y_pred = model.predict(X_test)

y_test = np.argmax(y_test, axis=1)

y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred))