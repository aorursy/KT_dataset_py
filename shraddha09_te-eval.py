# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pickle
import tensorflow as tf
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

with open("/kaggle/input/project/3way_df.pickle", "rb") as f:
    df = pickle.load(f)
    
columns = {'student-ans','ref-ans'}
embeddings = []
for row in df.index.values:
    r = np.empty((0,512))
    for col in columns:
        r = np.vstack((r,np.array(embed([df.loc[row,col]]))))
    embeddings.append(r)

embeddings = np.array(embeddings)

with open("google_encoder_8910_2_512.pickle","wb") as f:
    pickle.dump(embeddings,f)
import pickle

with open("/kaggle/input/project/embeddings_hstack_8910_1024.pickle","rb") as f:
    x = pickle.load(f)

with open("/kaggle/input/project/google_encoder_8910_2_512.pickle","rb") as f:
    embeddings = pickle.load(f)
    
with open("/kaggle/input/project/3way_df.pickle","rb") as f:
    df = pickle.load(f)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(df['score'].unique())

print(le.transform(['correct','contradictory','incorrect']))

y = le.transform(df['score'])
#only for svm
x = np.empty((0,2*512))
for row in range(len(embeddings)):
    r = np.hstack((embeddings[row][0], embeddings[row][1])) 
    x = np.vstack((x,r))

x.shape

xf = np.empty((0,2*512))
for row in range(len(embeddings)):
    #vector component wise product, absolute difference
    x1 = np.multiply(embeddings[row][0], embeddings[row][1])
    x2 = np.absolute(embeddings[row][0]-embeddings[row][1])
    r = np.hstack((x1,x2))
    xf = np.vstack((xf,r))

xf.shape

with open("embeddings_hstack_8910_1024.pickle","wb") as f:
    pickle.dump(x,f)
#ANN implementation

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=X_train[0].shape))
model.add(Dense(32, activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='adam', 
             loss='categorical_crossentropy',
             metrics=['accuracy'])

print(model.summary())

early_stops = EarlyStopping(patience=3, monitor='val_loss')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_split=0.05,
                   callbacks=[early_stops])
prediction = model.predict(X_test)

ann_acc = 0;

for val in range(len(X_test)):
    if(np.argmax(prediction[val]) == np.argmax(y_test[val])):
        ann_acc += 1

ann_acc = ann_acc/len(X_test)

print(ann_acc)
# SVM implementation

from sklearn.svm import SVC

model = SVC(decision_function_shape='ovo') # LinearSVC(tol=1.0e-6,max_iter=5000)
model.fit(X_train, y_train)
predicted_labels = model.predict(X_test)

acc = 0
for r in range(len(y_test)):
    if(predicted_labels[r]==y_test[r]):
        acc += 1
print(acc/len(y_test))
from tensorflow.keras.utils import to_categorical

y = to_categorical(y)
lstm_x = x.reshape(x.shape[0],x.shape[1],1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(lstm_x, y, test_size=0.20)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPool1D
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape = X_train[0].shape,
                             return_sequences=True, recurrent_dropout=0.2)))
#model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)))
#model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)))
#model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)))
model.add(GlobalAveragePooling1D())
#model.add(GlobalMaxPool1D())
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', 
             loss='categorical_crossentropy',
             metrics=['accuracy'])

#print(model.summary())

early_stops = EarlyStopping(patience=3, monitor='val_loss')

history = model.fit(X_train, y_train, epochs=50, batch_size=100, 
                    validation_split=0.05,
                   callbacks=[early_stops])
prediction = model.predict(X_test)

ann_acc = 0;

for val in range(len(X_test)):
    if(np.argmax(prediction[val]) == np.argmax(y_test[val])):
        ann_acc += 1

ann_acc = ann_acc/len(X_test)

print(ann_acc)