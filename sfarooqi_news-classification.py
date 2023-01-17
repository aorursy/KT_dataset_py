import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir('../input'))
#Read the data in dataframe
df = pd.read_json('../input/news-category-dataset/News_Category_Dataset.json', lines=True)
print(df.shape)
df.head()
df.groupby(by='category')['category'].count().sort_values(ascending=False)
#Merge `THE WORLDPOST` and `WORLDPOST` into single category
df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
df.groupby(by='category')['category'].count().sort_values(ascending=False)
#Number of articles published by month
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.groupby(pd.Grouper(key='date', freq='M'))['date'].count().sort_values(ascending=False)
#Average articles per month
df.groupby(pd.Grouper(key='date', freq='M'))['date'].count().mean()
#Popular category per month
df.groupby(pd.Grouper(key='date', freq='M'))['category'].agg(lambda x:x.value_counts().index[0])
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

#Tokenize headline
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.headline)
X = tokenizer.texts_to_sequences(df.headline)
df['words'] = X
df.head()
#Use GLOVE pretrained word-embeddings
EMBEDDING_DIMENSION=100
embeddings_index = {}
f = open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
#Create a weight matrix for words in training docs
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, 100))

for word, index in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

#Create embedding layer from embedding matrix
embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIMENSION,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=50, trainable=False)
from keras.utils import np_utils
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

#Prepare training and test data
X = np.array(list(sequence.pad_sequences(df.words, maxlen=50)))

category_dict = dict((i,k) for k,i in enumerate(list(df.groupby('category').groups.keys())))
df['labels'] = df['category'].apply(lambda x: category_dict[x])
Y = np_utils.to_categorical(list(df.labels))
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Flatten

#RNN with LSTM
model = Sequential()

model.add(embedding_layer)
model.add(LSTM(300, dropout=0.25, recurrent_dropout=0.25))
model.add(Dropout(0.25))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(len(category_dict), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model_history = model.fit(X, Y, batch_size=128, validation_split=0.4, epochs=15)
import matplotlib.pyplot as plt

acc = model_history.history['acc']
val_acc = model_history.history['val_acc']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()

plt.show()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.short_description)
X = tokenizer.texts_to_sequences(df.short_description)
df['short_description_tokenize'] = X
df.head()
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, 100))

for word, index in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIMENSION,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=50, trainable=False)
X = np.array(list(sequence.pad_sequences(df.short_description_tokenize, maxlen=50)))
model = Sequential()

model.add(embedding_layer)
model.add(LSTM(300, dropout=0.25, recurrent_dropout=0.25))
model.add(Dropout(0.25))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(len(category_dict), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model_history = model.fit(X, Y, batch_size=128, validation_split=0.3, epochs=15)
import matplotlib.pyplot as plt

acc = model_history.history['acc']
val_acc = model_history.history['val_acc']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()

plt.show()
