# import packages

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

import re

import pandas as pd

pd.set_option('display.max_colwidth', -1)
# read dataset

tweets = pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")

tweets.head(4)
tweets['tweet_len'] = tweets['text'].apply(len)

tweets.groupby(['tweet_len', 'airline_sentiment']).size().unstack().plot(kind='line', stacked=False)
data = tweets[['text','airline_sentiment']]

print(data['airline_sentiment'][:10])

#cleaning data set. Consider only the positive and negative ones

data = data[data.airline_sentiment != 'neutral']

print(data['airline_sentiment'][:10])

print(data['text'][:10])
# cleaning data set

data['text'] = data['text'].apply(lambda x: x.lower()) # convert all to lower

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x))) # remove anything except alphanumeric

print(data['text'][:10])

data['airline_sentiment'].value_counts().plot(kind='bar')

data['airline_sentiment'].value_counts()
#tokenization



mnax_words = 1500

tokenizer = Tokenizer(num_words=mnax_words, split=' ')

tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)

# print('Tokenized sentences', X[:5])

X = pad_sequences(X)

# print(X[:5])
enbedding_out_dim = 256

lstm_out_dim = 256



model = Sequential()

model.add(Embedding(mnax_words, enbedding_out_dim,input_length = X.shape[1]))

model.add(LSTM(lstm_out_dim))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
# data set to train

Y = pd.get_dummies(data['airline_sentiment']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 50)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
X_val = X_train[:500]

Y_val = Y_train[:500]
partial_X_train = X_train[500:]

partial_Y_train = Y_train[500:]
# train the net

batch_size = 512

history = model.fit(X_train,Y_train, 

                    epochs = 20, 

                    batch_size=batch_size,

                    validation_data=(X_val, Y_val))
loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# validation

positive_count, negative_count, positive_correct, neg_correct = 0, 0, 0, 0

for x in range(len(X_val)):



    result = model.predict(X_val[x].reshape(1, X_test.shape[1]), batch_size=1)[0]



    if np.argmax(result) == np.argmax(Y_val[x]):

        if np.argmax(Y_val[x]) == 0:

            neg_correct += 1

        else:

            positive_correct += 1



    if np.argmax(Y_val[x]) == 0:

        negative_count += 1

    else:

        positive_count += 1

print("positive accuracy", positive_correct / positive_count * 100, "%")

print("negative accuracy", neg_correct / negative_count * 100, "%")