import re

import numpy as np

import pandas as pd 



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from keras.datasets import imdb



from keras.utils.np_utils import to_categorical



import warnings

warnings.filterwarnings('ignore')

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
max_features = 1000

maxlen = 80  # cut texts after this number of words (among top max_features most common words)

batch_size = 32



# save np.load

#np_load_old = np.load



# modify the default parameters of np.load

#np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')

print(len(x_test), 'test sequences')



#np.load = np_load_old



print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)
x_train[0]
INDEX_FROM=3   # word index offset



word_to_id = imdb.get_word_index()

word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}

word_to_id["<PAD>"] = 0

word_to_id["<START>"] = 1

word_to_id["<UNK>"] = 2



id_to_word = {value:key for key,value in word_to_id.items()}

print(' '.join(id_to_word[id] for id in x_train[10] ))
print('Build model...')

model = Sequential()

model.add(Embedding(max_features, 8))

model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))

model.summary()
# try using different optimizers and different optimizer configs

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Write the training input and output, batch size, and testing input and output



model.fit(x_train, y_train, 

          batch_size=batch_size, 

          epochs=1, 

          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)
prediction = model.predict(x_test[22220:22221])

print('Prediction value:',prediction[0])

print('Test Label:',y_test[22220:22221])
# Credits to Peter Nagy
!wget https://notebooks.azure.com/vipulmishra/projects/labgail/raw/Senti.csv
import pandas as pd

data = pd.read_csv('Senti.csv')

# Keeping only the neccessary columns

data = data[['text','sentiment']]
data.head(10)
data = data[data.sentiment != "Neutral"]

data['text'] = data['text'].apply(lambda x: x.lower())

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))



for idx,row in data.iterrows():

    row[0] = row[0].replace('rt',' ')

    

max_fatures = 2000

tokenizer = Tokenizer(nb_words=max_fatures, split=' ')

tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)
Y = pd.get_dummies(data['sentiment']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

print('Shape of training samples:',X_train.shape,Y_train.shape)

print('Shape of testing samples:',X_test.shape,Y_test.shape)
model = Sequential()

model.add(Embedding(max_fatures, 128 ,input_length = X.shape[1], dropout=0.2))

model.add(LSTM(128))

model.add(Dense(2, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
batch_size = 32

model.fit(X_train, Y_train, epochs = 5, batch_size=batch_size, verbose = 2)
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("Score: %.2f" % (score))

print("Accuracy: %.2f" % (acc))
text = 'We are going to Delhi'

tester = np.array([text])

tester = pd.DataFrame(tester)

tester.columns = ['text']



tester['text'] = tester['text'].apply(lambda x: x.lower())

tester['text'] = tester['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))



max_fatures = 2000

test = tokenizer.texts_to_sequences(tester['text'].values)

test = pad_sequences(test)



if X.shape[1]>test.shape[1]:

    test = np.pad(test[0], (X.shape[1]-test.shape[1],0), 'constant')

    

test = np.array([test])



prediction = model.predict(test)

print('Prediction value:',prediction[0])
model = Sequential()

model.add(Embedding(max_features, 8))

model.add(LSTM(8, dropout=0.0, recurrent_dropout=0.0))

model.add(Dense(1, activation='sigmoid'))

model.summary()



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))



score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)
# Write your code here 



# Use the same layer design from the above cell 
model = Sequential()

model.add(Embedding(max_features, 4))

model.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))

model.add(Dense(1, activation='sigmoid'))

model.summary()



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))



score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)
model = Sequential()

model.add(Embedding(max_features, 32))

model.add(LSTM(8, dropout=0.5, recurrent_dropout=0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))



score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)
# Write your code here 



# Use the same model design from the above cell 
model = Sequential()

model.add(Embedding(max_features, 8))

model.add(LSTM(8, dropout=0.0, recurrent_dropout=0.0, return_sequences=True))

model.add(LSTM(8, dropout=0.0, recurrent_dropout=0.0, return_sequences=True))

model.add(LSTM(8, dropout=0.0, recurrent_dropout=0.0))

model.add(Dense(1, activation='sigmoid'))

model.summary()



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))



score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)
# Write your code here 



# Use the same node design from the above cell 