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
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
train = pd.read_csv('/kaggle/input/odia-news-dataset/train.csv')
test = pd.read_csv('/kaggle/input/odia-news-dataset/valid.csv')
train.label.unique()
train.head()
test.head()
print(train.shape)
print(test.shape)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Conv1D,MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
tokenizer = Tokenizer()
# entire training set sentences to be put as corpus
corpus = train.headings.values

print("Total sentences = ",len(corpus))
print(corpus[2])
print(corpus[1])
#Tokenizing the texts
#corpus = [item for sublist in corpus for item in sublist]
tokenizer.fit_on_texts(corpus)
len(tokenizer.word_index)
#Total number of words in the vocabulary
total_words = len(tokenizer.word_index) + 1
print("total_words = ",total_words)
tokenizer.word_index
# pad sequences 
sequences = tokenizer.texts_to_sequences(corpus)
max_sequence_len = max([len(x) for x in sequences])
padded_seq = pad_sequences(sequences, maxlen=max_sequence_len, padding='post', truncating='post')
print(padded_seq.shape)
len(padded_seq)
classes = len(train.label.unique())

test_portion = 0.3
training_size = len(padded_seq)

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
labels = onehot_encoder.fit_transform(np.asarray(train.label.values).reshape(-1, 1))
print(labels.shape)

split = int(test_portion * training_size)

#Train Validation split
valid_sequences = padded_seq[0:split]
training_sequences = padded_seq[split:training_size]
valid_labels = labels[0:split]
training_labels = labels[split:training_size]
print(valid_sequences.shape)
print(valid_labels.shape)
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150, return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(training_padded, training_labels, epochs=15,
                    validation_data=(valid_padded, valid_labels), verbose=1)
#TODO
#checkpoint
#callback
embedding_dim = 100

model = Sequential([
    Embedding(total_words, embedding_dim, input_length=max_sequence_len-1),#weights=[embeddings_matrix], trainable=False
    Dropout(0.2),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(pool_size=4),
    LSTM(64),
    Dense(classes, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 10

training_padded = np.array(training_sequences)
training_labels = np.array(training_labels)
valid_padded = np.array(valid_sequences)
valid_labels = np.array(valid_labels)

history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(valid_padded, valid_labels), verbose=2)

print("Training Complete")
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()
seed_text = test.headings.tolist()

#Use the tokenizer created with train data to convert the test data to sequences
token_list = tokenizer.texts_to_sequences(seed_text)
#Padding for the sequences to be of equal length, the parameters should match whatever was done during training
token_list = pad_sequences(token_list, maxlen=max_sequence_len, padding='post')

predicted = model.predict(token_list, verbose=0)
print(test.shape)
print(predicted.shape)
#We inverse transform the target / label to its original representation
predicted = onehot_encoder.inverse_transform(predicted)
from sklearn.metrics import accuracy_score
print("Accuracy = ", accuracy_score(test.label, predicted))