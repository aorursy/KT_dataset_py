# libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

np.random.seed(32)





from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.manifold import TSNE



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout

from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from keras.callbacks import EarlyStopping



%matplotlib inline
data = pd.read_csv("../input/arabic-news-texts-corpus/arabic_categorization_data.csv")
data.head()
train_text, test_text, train_y, test_y = train_test_split(data['text'],data['type'],test_size = 0.2, random_state=2019)
MAX_NB_WORDS = 20000



# get the raw text data

texts_train = train_text.astype(str)

texts_test = test_text.astype(str)



# finally, vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False)

tokenizer.fit_on_texts(texts_train)

sequences = tokenizer.texts_to_sequences(texts_train)

sequences_test = tokenizer.texts_to_sequences(texts_test)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
type(tokenizer.word_index), len(tokenizer.word_index)
index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())
" ".join([index_to_word[i] for i in sequences[0]])
seq_lens = [len(s) for s in sequences]

print("average length: %0.1f" % np.mean(seq_lens))

print("max length: %d" % max(seq_lens))
%matplotlib inline

import matplotlib.pyplot as plt



plt.hist(seq_lens, bins=50);



plt.hist([l for l in seq_lens if l < 400], bins=50);



# pad vectors to maximum length

MAX_SEQUENCE_LENGTH = 300



# pad sequences with 0s

x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', x_train.shape)

print('Shape of data test tensor:', x_test.shape)

# encode y data labels

encoder = LabelEncoder()

encoder.fit(train_y)

y_train = encoder.transform(train_y)

y_test = encoder.transform(test_y)


# Converts the labels to a one-hot representation

N_CLASSES = np.max(y_train) + 1

y_train = to_categorical(y_train, N_CLASSES)

y_test = to_categorical(y_test, N_CLASSES)

print('Shape of label tensor:', y_train.shape)
from keras.layers import Dense, Input, Flatten

from keras.layers import GlobalAveragePooling1D, Embedding

from keras.models import Model



EMBEDDING_DIM = 50



# input: a sequence of MAX_SEQUENCE_LENGTH integers

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)



embedded_sequences = embedding_layer(sequence_input)



average = GlobalAveragePooling1D()(embedded_sequences)

predictions = Dense(N_CLASSES, activation='softmax')(average)



model = Model(sequence_input, predictions)

model.compile(loss='categorical_crossentropy',

              optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, validation_split=0.2,

          nb_epoch=40, batch_size=128, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
output_test = model.predict(x_test)

print("test auc:", roc_auc_score(y_test,output_test))
# Evaluate the accuracy of our trained model

score = model.evaluate(x_test, y_test,

                       batch_size=64, verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# Here's how to generate a prediction on individual examples

text_labels = encoder.classes_ 



for i in range(50,80):

    prediction = model.predict(np.array([x_test[i]]))

    predicted_label = text_labels[np.argmax(prediction)]

    print(texts_test.iloc[i], "...")

    print('Actual label:' + test_y.iloc[i])

    print("Predicted label: " + predicted_label + "\n")  
# input: a sequence of MAX_SEQUENCE_LENGTH integers

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)



x = LSTM(128, dropout=0.5, recurrent_dropout=0.2)(embedded_sequences)

predictions = Dense(N_CLASSES, activation='softmax')(x)





model = Model(sequence_input, predictions)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])
model.fit(x_train, y_train, validation_split=0.1,

          nb_epoch=3, batch_size=64)
output_test = model.predict(x_test)

print("test auc:", roc_auc_score(y_test,output_test))
# Evaluate the accuracy of our trained model

score = model.evaluate(x_test, y_test,

                       batch_size=64, verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# input: a sequence of MAX_SEQUENCE_LENGTH integers

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)



# 1D convolution with 64 output channels

x = Conv1D(64, 5)(embedded_sequences)

# MaxPool divides the length of the sequence by 5

x = MaxPooling1D(5)(x)

x = Dropout(0.5)(x)

x = Conv1D(64, 5)(x)

x = MaxPooling1D(5)(x)

# LSTM layer with a hidden size of 64

x = Dropout(0.3)(x)

x = LSTM(32)(x)

predictions = Dense(N_CLASSES, activation='softmax')(x)



model = Model(sequence_input, predictions)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])
model.fit(x_train, y_train, validation_split=0.1,

          nb_epoch=3, batch_size=128)
# Evaluate the accuracy of our trained model

score = model.evaluate(x_test, y_test,

                       batch_size=64, verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
output_test = model.predict(x_test)

print("test auc:", roc_auc_score(y_test,output_test))
# Here's how to generate a prediction on individual examples

text_labels = encoder.classes_ 



for i in range(10):

    prediction = model.predict(np.array([x_test[i]]))

    predicted_label = text_labels[np.argmax(prediction)]

    print(texts_test.iloc[i], "...")

    print('Actual label:' + test_y.iloc[i])

    print("Predicted label: " + predicted_label + "\n")  