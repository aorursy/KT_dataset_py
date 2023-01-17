# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences











from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import SimpleRNN

from keras.layers import Embedding

from keras.layers import Bidirectional

from keras.layers import BatchNormalization

from keras.layers import Dropout



from keras.utils import to_categorical

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv", names=['comment', 'label'], header=0, encoding='utf-8')
df.head()
document_lenghts = list(map(len, df.comment.values))

print(np.max(document_lenghts))

print(np.min(document_lenghts))

print('mean_size:',np.mean(document_lenghts))

print('median_size:',np.median(document_lenghts))
df.label.value_counts()
dictionary_length = 1000

input_length = 100



tokenizer = Tokenizer(num_words=dictionary_length)

tokenizer.fit_on_texts(df.comment.values)
post_seq = tokenizer.texts_to_sequences(df.comment.values)
print(len(post_seq))

print(post_seq[0])

print(len(post_seq[0]))
post_seq_padded = pad_sequences(post_seq, maxlen=input_length)
print(len(post_seq_padded))

print(post_seq_padded[0])

print(len(post_seq_padded[0]))
x_original = post_seq_padded

x_original = np.array(x_original)



y_original = df['label'].values

y_original = 1*(y_original=='positive')

y_original = np.array(y_original)









x, y = shuffle(x_original, y_original, random_state=23)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=23)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=23)
print("train set:", x_train.shape)

print("validation set:", x_val.shape)

print("test set:", x_test.shape)
model = Sequential()

model.add(Embedding(dictionary_length, 2, input_length=input_length))

model.add(Dense(32,activation="relu"))

model.add(Dense(32,activation="relu"))

model.add(Bidirectional(SimpleRNN(16, return_sequences=True)))

model.add(Bidirectional(SimpleRNN(16, return_sequences=True)))

model.add(Bidirectional(SimpleRNN(16, return_sequences=False)))

model.add(Dense(1, activation='sigmoid'))



model.summary()
model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model.fit(x=x_train, y=y_train, batch_size=256, verbose=1, epochs=5, validation_data=(x_val, y_val))
history_dict = history.history



acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']

loss = history_dict['loss']

val_loss = history_dict['val_loss']



epochs = range(1, len(acc) + 1)



plt.figure(figsize=(8,5))

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



plt.figure(figsize=(8,5))

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')

plt.ylim((0.5,1))

plt.show()
bad_comment = tokenizer.texts_to_sequences(["Wow, this is the worst film I ever seen. This film is really bad"])

good_comment = tokenizer.texts_to_sequences(["Not so bad, it is not a masterpiece but I liked it"])



bad_comment = pad_sequences(bad_comment, maxlen=input_length)

good_comment = pad_sequences(good_comment, maxlen=input_length)



#print(bad_comment)

print(model.predict(bad_comment))

print(model.predict(good_comment))
e = model.layers[0]

weights = e.get_weights()[0]

print(weights.shape)

plt.figure(figsize=(12,9))

plt.plot(weights[:,0], weights[:,1], 'bo')

plt.title('Word embedding')

plt.show()