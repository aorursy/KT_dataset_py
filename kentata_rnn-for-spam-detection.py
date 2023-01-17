from keras.layers import SimpleRNN, Embedding, Dense, LSTM

from keras.models import Sequential



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns; sns.set()
data = pd.read_csv("../input/SPAM text message 20170820 - Data.csv")
texts = []

labels = []

for i, label in enumerate(data['Category']):

    texts.append(data['Message'][i])

    if label == 'ham':

        labels.append(0)

    else:

        labels.append(1)



texts = np.asarray(texts)

labels = np.asarray(labels)





print("number of texts :" , len(texts))

print("number of labels: ", len(labels))
from keras.layers import SimpleRNN, Embedding, Dense, LSTM

from keras.models import Sequential



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# number of words used as features

max_features = 10000

# cut off the words after seeing 500 words in each document(email)

maxlen = 500





# we will use 80% of data as training, 20% as validation data

training_samples = int(5572 * .8)

validation_samples = int(5572 - training_samples)

# sanity check

print(len(texts) == (training_samples + validation_samples))

print("The number of training {0}, validation {1} ".format(training_samples, validation_samples))



tokenizer = Tokenizer()

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)



word_index = tokenizer.word_index

print("Found {0} unique words: ".format(len(word_index)))



data = pad_sequences(sequences, maxlen=maxlen)



print("data shape: ", data.shape)



np.random.seed(42)

# shuffle data

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]





texts_train = data[:training_samples]

y_train = labels[:training_samples]

texts_test = data[training_samples:]

y_test = labels[training_samples:]
model = Sequential()

model.add(Embedding(max_features, 32))

model.add(SimpleRNN(32))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history_rnn = model.fit(texts_train, y_train, epochs=10, batch_size=60, validation_split=0.2)
acc = history_rnn.history['acc']

val_acc = history_rnn.history['val_acc']

loss = history_rnn.history['loss']

val_loss = history_rnn.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, '-', color='orange', label='training acc')

plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()



plt.plot(epochs, loss, '-', color='orange', label='training acc')

plt.plot(epochs, val_loss,  '-', color='blue', label='validation acc')

plt.title('Training and validation loss')

plt.legend()

plt.show()
pred = model.predict_classes(texts_test)

acc = model.evaluate(texts_test, y_test)

proba_rnn = model.predict_proba(texts_test)

from sklearn.metrics import confusion_matrix

print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))

print(confusion_matrix(pred, y_test))
model = Sequential()

model.add(Embedding(max_features, 32))

model.add(LSTM(32))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history_ltsm = model.fit(texts_train, y_train, epochs=10, batch_size=60, validation_split=0.2)
acc = history_ltsm.history['acc']

val_acc = history_ltsm.history['val_acc']

loss = history_ltsm.history['loss']

val_loss = history_ltsm.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, '-', color='orange', label='training acc')

plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()



plt.plot(epochs, loss, '-', color='orange', label='training acc')

plt.plot(epochs, val_loss,  '-', color='blue', label='validation acc')

plt.title('Training and validation loss')

plt.legend()

plt.show()
pred = model.predict_classes(texts_test)

acc = model.evaluate(texts_test, y_test)

proba_ltsm = model.predict_proba(texts_test)

from sklearn.metrics import confusion_matrix

print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))

print(confusion_matrix(pred, y_test))
ensemble_proba = 0.3 * proba_rnn + 0.7 * proba_ltsm
ensemble_proba[:5]
ensemble_class = np.array([1 if i >= 0.5 else 0 for i in ensemble_proba])
ensemble_class[:5]
print(confusion_matrix(pred, y_test))