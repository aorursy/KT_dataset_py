#Importing the useful methods from the RNN library keras (sublibrary of tensorflow)

from keras.layers import SimpleRNN, Embedding, Dense, LSTM, GRU

from keras.models import Sequential



#Importing th basic math,plotting and data manipulation libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns; sns.set()
#Reading the data from the csv file and taking a look at it

data = pd.read_csv("../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")

data.head(5)
#Generating the training/test data for the model and assigning numerical labels 0(nonspam), 1(spam).



emails = []

labels = []

for i, label in enumerate(data['Category']):

    emails.append(data['Message'][i])

    if label == 'ham':

        labels.append(0)

    else:

        labels.append(1)



emails = np.asarray(emails)

labels = np.asarray(labels)





print("number of emails :" , len(emails))

print("number of labels: ", len(labels))
#from keras.layers import SimpleRNN, Embedding, Dense, LSTM

#from keras.models import Sequential



#Import libraries for preprocessing text

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# number of words used as features

max_features = 10000

# cut off the words after seeing 500 words in each document(email)

maxlen = 500





# we will use 80% of data as training, 20% as validation data

training_samples = int(len(emails) * .8)

test_samples = int(len(emails) - training_samples)



print("The number of training {0}, validation {1} ".format(training_samples, test_samples))



tokenizer = Tokenizer()

tokenizer.fit_on_texts(emails)

sequences = tokenizer.texts_to_sequences(emails)





word_index = tokenizer.word_index

print("Found {0} unique words: ".format(len(word_index)))



first_email = []

for i in sequences[0]:

    first_email.append([word for word, index in word_index.items() if index == i][0]);

print(first_email)



data = pad_sequences(sequences, maxlen=maxlen)



print("data shape: ", data.shape)



np.random.seed(23)

# shuffle data

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]





emails_train = data[:training_samples]

y_train = labels[:training_samples]

emails_test = data[training_samples:]

y_test = labels[training_samples:]
#Define the RNN model



model = Sequential()

model.add(Embedding(max_features, 32))

model.add(SimpleRNN(32))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history_rnn = model.fit(emails_train, y_train, epochs=10, batch_size=60, validation_split=0.2)
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



plt.plot(epochs, loss, '-', color='orange', label='training loss')

plt.plot(epochs, val_loss,  '-', color='blue', label='validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
pred = model.predict_classes(emails_test)

acc = model.evaluate(emails_test, y_test)

proba_rnn = model.predict_proba(emails_test)

from sklearn.metrics import confusion_matrix

print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))

print(confusion_matrix(pred, y_test))
model = Sequential()

model.add(Embedding(max_features, 32))

model.add(LSTM(32))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history_ltsm = model.fit(emails_train, y_train, epochs=10, batch_size=30, validation_split=0.2)
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



plt.plot(epochs, loss, '-', color='orange', label='training loss')

plt.plot(epochs, val_loss,  '-', color='blue', label='validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
pred = model.predict_classes(emails_test)

acc = model.evaluate(emails_test, y_test)

proba_ltsm = model.predict_proba(emails_test)

from sklearn.metrics import confusion_matrix

print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))

print(confusion_matrix(pred, y_test))