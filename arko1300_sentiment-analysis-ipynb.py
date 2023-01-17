#%tensorflow_version 2.x

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, SpatialDropout1D, Embedding

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

import re
data = pd.read_csv('../input/Sentiment.csv')

#keep necessary columns

data = data[['text', 'sentiment']]
data = data[data.sentiment != "Neutral"]

data['text'] = data['text'].apply(lambda x: x.lower())

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-Z0-9\s]', '', x)))



print(data[data['sentiment'] == 'Positive'].size)

print(data[data['sentiment'] == 'Negative'].size)
#Replacing RT from the text

for idx, row in data.iterrows():

    row[0] = row[0].replace('rt', ' ')
max_features = 2000

tokenizer = Tokenizer(num_words=max_features, split=' ')

tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)
embed_dim = 128

lstm_out = 196



model = Sequential()

model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.4))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

#model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.3, return_sequences=True))

#model.add(LSTM(364, dropout=0.2, recurrent_dropout=0.3))

model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Y = pd.get_dummies(data['sentiment']).values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print(X_train.shape, ' ', y_train.shape)

print(X_test.shape, ' ', y_test.shape)
batch_size = 32

model.fit(X_train, y_train, epochs=20, batch_size=batch_size, verbose=2)
validation_size=  1500

X_validate = X_test[-validation_size:]

y_validate = y_test[-validation_size:]

score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)



print("score: %.2f"%(score))

print("acc: %.2f"%(acc))
pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(X_validate)):

    result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)

    if np.argmax(result) == np.argmax(y_validate[x]):

        if np.argmax(y_validate[x] == 0):

            neg_correct += 1

        else:

            pos_correct += 1

    

    if np.argmax(y_validate[x]) == 0:

        neg_cnt += 1

    else:

        pos_cnt += 1



print("pos_acc", pos_correct/pos_cnt*100, "%")

print("neg_acc", neg_correct/neg_cnt*100, "%")
twt = input('Enter Tweet: ')

twt = tokenizer.texts_to_sequences(twt)



twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)

#print(twt)

sentiment = model.predict(twt, batch_size=1, verbose=2)[0]

if(np.argmax(sentiment)==0):

    print("Negative")

elif(np.argmax(sentiment)==1):

    print("Positive")