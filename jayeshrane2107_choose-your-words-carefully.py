import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
train = pd.read_csv('../input/train.csv',header=0)
test = pd.read_csv('../input/test.csv',header=0)
train.head()
test.head()
train.shape, test.shape
print(train.isnull().any() ,'\n\n', test.isnull().any())
train_features = train['comment_text']
test_features = test['comment_text']

classes = list(train.columns.values[2:])
train_labels = train[classes]
classes
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_features)
train_tokenized_list = tokenizer.texts_to_sequences(train_features)
test_tokenized_list = tokenizer.texts_to_sequences(test_features)
train_features[5]
train_tokenized_list[5]
totalNumWords = [len(comment) for comment in train_tokenized_list]
totalNumWords[5]
plt.hist(totalNumWords,bins=np.arange(0,510,10))
plt.show()
pad_len = 200
train_pad = pad_sequences(train_tokenized_list,maxlen=pad_len,padding='post')
test_pad = pad_sequences(test_tokenized_list,maxlen=pad_len,padding='post')

train_pad[5]
inp = Input(shape=(pad_len,))
embed_size = 128
x = Embedding(max_features,embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
batch_size = 32
epochs = 1
model.fit(train_pad,train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.summary()
pred = model.predict(test_pad)
test_labels = pd.DataFrame({'id':test['id'],classes[0]:pred[:,0], classes[1]:pred[:,1], classes[2]:pred[:,2], classes[3]:pred[:,3], classes[4]:pred[:,4], classes[5]:pred[:,5]},columns=['id','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
test_labels.to_csv('Toxic_comment.csv',index=False)
test_labels.head()
