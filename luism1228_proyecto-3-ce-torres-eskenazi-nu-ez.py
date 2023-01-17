import os

import time

import numpy as np 

import pandas as pd 

from tqdm import tqdm

import math

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

plt.style.use('dark_background')





from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Input

from keras.layers import Embedding

from keras.models import Model

from keras.initializers import Constant

from keras.layers import LSTM
train = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
x_train = train['question_text']

y_train= train['target']

x_test = test["question_text"].fillna("dieter").values
token = Tokenizer()

token.fit_on_texts(x_train)

seq = token.texts_to_sequences(x_train)
pad_seq = pad_sequences(seq,maxlen=300)

vocab_size = len(token.word_index)+1
embedding_vector = {}

f = open('../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt')

for line in tqdm(f):

    value = line.split(' ')

    word = value[0]

    coef = np.array(value[1:],dtype = 'float32')

    embedding_vector[word] = coef
embedding_matrix = np.zeros((vocab_size,300))

for word,i in tqdm(token.word_index.items()):

    embedding_value = embedding_vector.get(word)

    if embedding_value is not None:

        embedding_matrix[i] = embedding_value


model = Sequential()



model.add(Embedding(vocab_size,300,weights = [embedding_matrix],input_length=300,trainable = False))



model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))



model.add(Dense(1,activation = 'sigmoid'))



model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])





print(model.summary())

history = model.fit(pad_seq,y_train,epochs = 4,batch_size=256,validation_split=0.2)
plt.clf()

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'g', label='Training loss')

plt.plot(epochs, val_loss, 'y', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
x_test = test['question_text']

x_test = token.texts_to_sequences(x_test)

x_test = pad_sequences(x_test,maxlen=300)

y_pred = model.predict(x_test, batch_size=1024)

yf=(y_pred > 0.5).astype(int).reshape(x_test.shape[0])
submit_df = pd.DataFrame({"qid": test["qid"], "prediction": yf})

submit_df.to_csv("submission.csv", index=False)

submit_df