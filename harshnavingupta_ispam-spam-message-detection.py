import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objs as go
import os

dataset_path = ""

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        dataset_path = os.path.join(dirname, filename)
dataset = pd.read_csv(dataset_path, delimiter=',', encoding='latin-1')
dataset['Category'] = dataset['v1']

dataset['Message'] = dataset['v2']

dataset.drop(['v1', 'v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
dataset.head()
dataset.shape
dataset.isnull().sum()
msgs = dataset['Message']

category = dataset['Category']
msgs_len = []

for m in msgs:

    msgs_len.append(len(m))

fig = go.Figure(data=[go.Histogram(x=msgs_len)])

fig.show()
dataset = dataset[dataset['Message'].map(len) <= 200]

msgs = dataset['Message']

category = dataset['Category']
category.value_counts()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(category)

category = encoder.fit_transform(category)
X = msgs

y = category



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)
print('X Train Shape : ' + str(X_train.shape))

print('Y Train Shape : ' + str(y_train.shape))

print('X Test Shape : ' + str(X_test.shape))

print('Y Test Shape : ' + str(y_test.shape))
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)

X_test = vectorizer.transform(X_test)
print('X Train Shape After Vectorization : ' + str(X_train.shape))

print('X Test Shape After Vectorization : ' + str(X_test.shape))
from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(32, input_dim=7310, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam',

             loss='binary_crossentropy',

             metrics = ['accuracy'])

model.summary()
num_epochs = 50

history = model.fit(X_train, y_train,

                   validation_data = (X_test, y_test),

                   epochs = num_epochs,

                   batch_size = 200,

                   verbose=True)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']
epoch_range = list(range(1, num_epochs + 1))

fig = go.Figure()

fig.add_trace(go.Scatter(x=epoch_range, y=acc, name='Train Accuracy', mode='lines+markers'))

fig.add_trace(go.Scatter(x=epoch_range, y=val_acc, name='Test Accuracy', mode='lines+markers'))

fig.update_layout(title='Train & Test Accuracy Trend',

                   xaxis_title='Epochs',

                   yaxis_title='Accuracy Of Model')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=epoch_range, y=loss, name='Train Loss', mode='lines+markers'))

fig.add_trace(go.Scatter(x=epoch_range, y=val_loss, name='Test Loss', mode='lines+markers'))

fig.update_layout(title='Train & Test Loss Trend',

                   xaxis_title='Epochs',

                   yaxis_title='Loss')

fig.show()
bw_model = Sequential()

bw_model.add(Dense(32, input_dim=7310, activation='relu'))

bw_model.add(Dense(16, activation='relu'))

bw_model.add(Dense(4, activation='relu'))

bw_model.add(Dense(1, activation='sigmoid'))



bw_model.compile(optimizer='adam',

             loss='binary_crossentropy',

             metrics = ['accuracy'])



num_epochs = 7

history = bw_model.fit(X_train, y_train,

                   validation_data = (X_test, y_test),

                   epochs = num_epochs,

                   batch_size = 200,

                   verbose=False)



loss, train_acc = bw_model.evaluate(X_train, y_train, verbose=False)

loss, test_acc = bw_model.evaluate(X_test, y_test, verbose=False)
print('Training Accuracy : ' + str(train_acc))

print('Testing Accuracy : ' + str(test_acc))
from sklearn.metrics import f1_score

pred = bw_model.predict_classes(X_train)

train_f1 = f1_score(y_train, pred)

pred = bw_model.predict_classes(X_test)

test_f1 = f1_score(y_test, pred)



print('Train F1 Score : ' + str(train_f1))

print('Test F1 Score : ' + str(test_f1))
bw_model_json = bw_model.to_json()

with open("bw_model.json", "w") as json_file:

    json_file.write(bw_model_json)

bw_model.save_weights("bw_model.h5")
X = msgs

y = category



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)
print('X Train Shape : ' + str(X_train.shape))

print('Y Train Shape : ' + str(y_train.shape))

print('X Test Shape : ' + str(X_test.shape))

print('Y Test Shape : ' + str(y_test.shape))
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = 10000)

tokenizer.fit_on_texts(X_train)



X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)



vocab_size = len(tokenizer.word_index) + 1
from keras.preprocessing.sequence import pad_sequences

maxlen = 200



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
embedding_dim = 300



from keras.models import Sequential

from keras.layers import Dense, GlobalMaxPool1D, Embedding



model = Sequential()

model.add(Embedding(input_dim = vocab_size,

                   output_dim = embedding_dim,

                   input_length = maxlen))

model.add(GlobalMaxPool1D())

model.add(Dense(16, activation='relu'))

model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])

model.summary()
num_epochs = 50

history = model.fit(X_train, y_train,

                   epochs=num_epochs,

                   validation_data = (X_test, y_test),

                   batch_size = 200)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']
epoch_range = list(range(1, num_epochs + 1))

fig = go.Figure()

fig.add_trace(go.Scatter(x=epoch_range, y=acc, name='Train Accuracy', mode='lines+markers'))

fig.add_trace(go.Scatter(x=epoch_range, y=val_acc, name='Test Accuracy', mode='lines+markers'))

fig.update_layout(title='Train & Test Accuracy Trend',

                   xaxis_title='Epochs',

                   yaxis_title='Accuracy Of Model')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=epoch_range, y=loss, name='Train Loss', mode='lines+markers'))

fig.add_trace(go.Scatter(x=epoch_range, y=val_loss, name='Test Loss', mode='lines+markers'))

fig.update_layout(title='Train & Test Loss Trend',

                   xaxis_title='Epochs',

                   yaxis_title='Loss')

fig.show()
embedding_dim = 300



em_model = Sequential()

em_model.add(Embedding(input_dim = vocab_size,

                   output_dim = embedding_dim,

                   input_length = maxlen))

em_model.add(GlobalMaxPool1D())

em_model.add(Dense(16, activation='relu'))

em_model.add(Dense(4, activation='relu'))

em_model.add(Dense(1, activation='sigmoid'))



em_model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])



num_epochs = 21

history = em_model.fit(X_train, y_train,

                   epochs=num_epochs,

                   validation_data = (X_test, y_test),

                   batch_size = 200)
loss, train_acc = em_model.evaluate(X_train, y_train, verbose=False)

loss, test_acc = em_model.evaluate(X_test, y_test, verbose=False)



print('Train Accuracy : ' + str(train_acc))

print('Test Accuracy : ' + str(test_acc))
from sklearn.metrics import f1_score

pred = em_model.predict_classes(X_train)

train_f1 = f1_score(y_train, pred)

pred = em_model.predict_classes(X_test)

test_f1 = f1_score(y_test, pred)



print('Train F1 Score : ' + str(train_f1))

print('Test F1 Score : ' + str(test_f1))
em_model_json = em_model.to_json()

with open("em_model.json", "w") as json_file:

    json_file.write(em_model_json)

em_model.save_weights("em_model.h5")