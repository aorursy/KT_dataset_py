import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from nltk.corpus import stopwords

import re

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Embedding

from tensorflow.keras.datasets import imdb
(x_train, y_train),(x_test, y_test) = imdb.load_data(skip_top=20, num_words=50000) # 상위 20개 단어는 건너뜀, 훈련에 사용할 단어 개수=50000

print(x_train.shape, y_train.shape)
print(x_train[0])
for i in range(len(x_train)):

    x_train[i] = [w for w in x_train[i] if w>2]
# 영단어와 정수로 구성된 딕셔너리 반환

word_to_index = imdb.get_word_index()

# 정수를 영단어로 변환

index_to_word = {word_to_index[k]: k for k in word_to_index}    



index_to_word
def decode_review(data):

    return ' '.join([index_to_word.get(i,'?') for i in data])



decode_review(x_train[0])
# 불용어 처리

word = index_to_word.values()

stop_words = set(stopwords.words('english'))



word_clean=[]



for w in word:

    if w not in stop_words:

        word_clean.append(w) 
np.unique(y_train, return_counts=True)
maxlen = 300 # 최대 샘플 길이

x_train_seq = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test_seq = sequence.pad_sequences(x_test, maxlen=maxlen)



print(x_train_seq.shape, x_test_seq.shape)
print(x_train_seq[0])
model = keras.Sequential()

model.add(keras.layers.Embedding(50000,16))

model.add(keras.layers.SimpleRNN(16))

model.add(keras.layers.Dense(16, activation='relu'))

model.add(keras.layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train_seq, y_train, epochs=10, batch_size=32, validation_split=0.5)
results = model.evaluate(x_test_seq, y_test, verbose=2)



print(results)
plt.plot(hist.history['loss'],'r',label='loss')

plt.plot(hist.history['val_loss'],'b',label='val_loss')

plt.legend()

plt.show()
plt.plot(hist.history['accuracy'],'r',label='accuracy')

plt.plot(hist.history['val_accuracy'],'b',label='val_accuracy')

plt.legend()

plt.show()