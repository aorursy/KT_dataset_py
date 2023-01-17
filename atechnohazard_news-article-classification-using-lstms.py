import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_score, recall_score

from mlxtend.plotting import plot_confusion_matrix



from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.utils import plot_model



physical_devices = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
TRAIN_FILE_PATH = '/kaggle/input/ag-news-classification-dataset/train.csv'

TEST_FILE_PATH = '/kaggle/input/ag-news-classification-dataset/test.csv'



data = pd.read_csv(TRAIN_FILE_PATH)

testdata = pd.read_csv(TEST_FILE_PATH)



X_train = data['Title'] + " " + data['Description'] # Combine title and description (better accuracy than using them as separate features)

y_train = data['Class Index'].apply(lambda x: x-1).values # Class labels need to begin from 0



x_test = testdata['Title'] + " " + testdata['Description'] # Combine title and description (better accuracy than using them as separate features)

y_test = testdata['Class Index'].apply(lambda x: x-1).values # Class labels need to begin from 0



maxlen = X_train.map(lambda x: len(x.split())).max() # max length of sentences in train dataset

data.head()
vocab_size = 10000 # arbitrarily chosen

embed_size = 32 # arbitrarily chosen



# Create and fit tokenizer

tok = Tokenizer(num_words=vocab_size)

tok.fit_on_texts(X_train.values)



# Tokenize data

X_train = tok.texts_to_sequences(X_train)

x_test = tok.texts_to_sequences(x_test)



# Pad data

X_train = pad_sequences(X_train, maxlen=maxlen)

x_test = pad_sequences(x_test, maxlen=maxlen)
model = Sequential()

model.add(Embedding(vocab_size, embed_size, input_length=maxlen))

model.add(Bidirectional(LSTM(128, return_sequences=True))) # bidirectional LSTMs since this isn't a timeseries problem

model.add(Bidirectional(LSTM(64, return_sequences=True)))

model.add(GlobalMaxPooling1D())

model.add(Dense(1024))

model.add(Dropout(0.25))

model.add(Dense(512))

model.add(Dropout(0.25))

model.add(Dense(256))

model.add(Dropout(0.25))

model.add(Dense(4, activation='softmax'))

model.summary()
callbacks = [

    EarlyStopping(

        monitor='val_accuracy',

        min_delta=1e-4,

        patience=4,

        verbose=1

    ),

    ModelCheckpoint(

        filepath='weights.h5',

        monitor='val_accuracy', 

        mode='max', 

        save_best_only=True,

        save_weights_only=True,

        verbose=1

    )

]
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # sparse categorical crossentropy loss because data is not one-hot encoded

model.fit(X_train, y_train, batch_size=256, validation_data=(x_test, y_test), epochs=20, callbacks=callbacks)
model.load_weights('weights.h5')

model.save('model.hdf5')
labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']



test = ['New evidence of virus risks from wildlife trade', 'Coronavirus: Bank pumps £100bn into UK economy to aid recovery', 

        'Trump\'s bid to end Obama-era immigration policy ruled unlawful', 'David Luiz’s future with Arsenal to be decided this week']

test_seq = pad_sequences(tok.texts_to_sequences(test), maxlen=maxlen)

test_preds = [labels[np.argmax(i)] for i in model.predict(test_seq)]



for news, label in zip(test, test_preds):

    print('{} - {}'.format(news, label))
preds = [np.argmax(i) for i in model.predict(x_test)]

cm  = confusion_matrix(y_test, preds)

plt.figure()

plot_confusion_matrix(cm, figsize=(16,12), hide_ticks=True, cmap=plt.cm.Blues)

plt.xticks(range(4), labels, fontsize=12)

plt.yticks(range(4), labels, fontsize=12)

plt.show()
print("Recall of the model is {:.2f}".format(recall_score(y_test, preds, average='micro')))

print("Precision of the model is {:.2f}".format(precision_score(y_test, preds, average='micro')))