import pandas as pd

import numpy as np

import keras

from keras.models import Sequential

from keras.layers import LSTM, Dense, GlobalAvgPool1D, Dropout, Embedding,Bidirectional, Flatten, CuDNNLSTM, Conv1D, MaxPooling1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

import random

import matplotlib.pyplot as plt
training_set = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
training_set = training_set.drop(['id'], axis=1)
print("Number of training records :",len(training_set))

print("Columns :")

for i in training_set:

    print("\t"+i)
#plot 2

columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']  

count_ones = []

for i in columns:

    count_ones.append(training_set[training_set[i]==1][i].count())

y_pos = np.arange(len(columns))

plt.bar(y_pos, count_ones, align="center", alpha=0.5)

plt.xticks(y_pos, columns)

plt.ylabel("Number of Ones")

plt.title("Number of Ones")

plt.show()



#plot 1

count_zeros = []

for i in columns:

    count_zeros.append(training_set[training_set[i]==0][i].count())

y_pos = np.arange(len(columns))

plt.bar(y_pos, count_zeros, align="center", alpha=0.5)

plt.xticks(y_pos, columns)

plt.ylabel("Number of Zeros")

plt.title("Number of Zeros")

plt.show()
for i in range(1):

    j = random.randint(0, 10000)

    print(training_set.values[j])

    
f = open("../input/glove-embeddings/glove.6B.300d.txt")
embedding_matrix = {}

for line in tqdm(f):

    temp = line.split(" ")

    word = temp[0]

    embeds = np.array(temp[1:], dtype='float32')

    embedding_matrix[word] = embeds
x = training_set['comment_text']

y = training_set[columns]
token = Tokenizer(num_words=20000)

token.fit_on_texts(x)

seq = token.texts_to_sequences(x)
padded_seq = pad_sequences(seq, maxlen=40)
vocab_size = len(token.word_index)+1

print(vocab_size)
embeddings = np.zeros((vocab_size, 300))

for word, i in tqdm(token.word_index.items(), position=0):

    embeds = embedding_matrix.get(word)

    if embeds is not None:

        embeddings[i] = embeds
model1 = Sequential()

model1.add(Embedding(vocab_size, 300, weights = [embeddings],

                     input_length=40, trainable=False))

model1.add(Conv1D(128, 5, activation='relu'))

model1.add(MaxPooling1D(5))

model1.add(Conv1D(128, 5, activation='relu'))

model1.add(MaxPooling1D(3))

model1.add(Flatten())

model1.add(Dense(128, activation='relu'))

model1.add(Dense(1, activation='sigmoid'))



model1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])



model1.summary()
model1.fit(padded_seq, training_set['toxic'], epochs=3, batch_size=32, validation_split=0.2)
model2 = Sequential()

model2.add(Embedding(vocab_size, 300, weights = [embeddings],

                     input_length=40, trainable=False))

model2.add(Conv1D(128, 5, activation='relu'))

model2.add(MaxPooling1D(5))

model2.add(Conv1D(128, 5, activation='relu'))

model2.add(MaxPooling1D(3))

model2.add(Flatten())

model2.add(Dense(128, activation='relu'))

model2.add(Dense(1, activation='sigmoid'))



model2.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])



model2.summary()
model2.fit(padded_seq, training_set['severe_toxic'], epochs=2, batch_size=32, validation_split=0.2)
model3 = Sequential()

model3.add(Embedding(vocab_size, 300, weights = [embeddings],

                     input_length=40, trainable=False))

model3.add(Conv1D(128, 5, activation='relu'))

model3.add(MaxPooling1D(5))

model3.add(Conv1D(128, 5, activation='relu'))

model3.add(MaxPooling1D(3))

model3.add(Flatten())

model3.add(Dense(128, activation='relu'))

model3.add(Dense(1, activation='sigmoid'))



model3.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])



model3.summary()
model3.fit(padded_seq, training_set['obscene'], epochs=2, batch_size=32, validation_split=0.2)
model4 = Sequential()

model4.add(Embedding(vocab_size, 300, weights = [embeddings],

                     input_length=40, trainable=False))

model4.add(Conv1D(128, 5, activation='relu'))

model4.add(MaxPooling1D(5))

model4.add(Conv1D(128, 5, activation='relu'))

model4.add(MaxPooling1D(3))

model4.add(Flatten())

model4.add(Dense(128, activation='relu'))

model4.add(Dense(1, activation='sigmoid'))



model4.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])



model4.summary()
model4.fit(padded_seq, training_set['threat'], epochs=1, batch_size=32, validation_split=0.2)
model5 = Sequential()

model5.add(Embedding(vocab_size, 300, weights = [embeddings],

                     input_length=40, trainable=False))

model5.add(Conv1D(128, 5, activation='relu'))

model5.add(MaxPooling1D(5))

model5.add(Conv1D(128, 5, activation='relu'))

model5.add(MaxPooling1D(3))

model5.add(Flatten())

model5.add(Dense(128, activation='relu'))

model5.add(Dense(1, activation='sigmoid'))



model5.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])



model5.summary()
model5.fit(padded_seq, training_set['insult'], epochs=2, batch_size=32, validation_split=0.2)
model6 = Sequential()

model6.add(Embedding(vocab_size, 300, weights = [embeddings],

                     input_length=40, trainable=False))

model6.add(Conv1D(128, 5, activation='relu'))

model6.add(MaxPooling1D(5))

model6.add(Conv1D(128, 5, activation='relu'))

model6.add(MaxPooling1D(3))

model6.add(Flatten())

model6.add(Dense(128, activation='relu'))

model6.add(Dense(1, activation='sigmoid'))



model6.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])



model6.summary()
model6.fit(padded_seq, training_set['identity_hate'], epochs=1, batch_size=32, validation_split=0.2)
test_set = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
x_test = test_set['comment_text']

token = Tokenizer(num_words=20000)

token.fit_on_texts(x_test)

seq = token.texts_to_sequences(x_test)
test_padded_seq = pad_sequences(seq, maxlen=40)
toxic = model1.predict(test_padded_seq)

severe_toxic = model2.predict(test_padded_seq)

obscene = model3.predict(test_padded_seq)

threat = model4.predict(test_padded_seq)

insult = model5.predict(test_padded_seq)

identity_hate = model6.predict(test_padded_seq)
toxic = [1 if i>=0.5 else 0 for i in toxic]

severe_toxic = [1 if i>=0.5 else 0 for i in severe_toxic]

obscene = [1 if i>=0.5 else 0 for i in obscene]

threat = [1 if i>=0.5 else 0 for i in threat]

insult = [1 if i>=0.5 else 0 for i in insult]

identity_hate = [1 if i>=0.5 else 0 for i in identity_hate]
id = test_set['id']
df = pd.DataFrame({'id':id,

                   'toxic':toxic,

                   'severe_toxic':severe_toxic,

                   'obscene':obscene,

                   'threat':threat,

                   'insult':insult,

                   'identity_hate':identity_hate})
df.to_csv("submission.csv", index=False)