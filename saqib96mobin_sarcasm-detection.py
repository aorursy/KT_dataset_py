from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, CuDNNLSTM, GlobalMaxPool1D, CuDNNGRU

from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
import json



def parse_data(file):

    for l in open(file, 'r'):

        yield json.loads(l)



data = list(parse_data("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json"))



Sentences = []

Label = []

Url = []



for i in data:

    Sentences.append(i["headline"])

    Label.append(i["is_sarcastic"])

    Url.append(i["article_link"])
max_words = 10000

max_len = 25

emb_size=300



tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")

tokenizer.fit_on_texts(Sentences)
word_index = tokenizer.word_index

print(len(word_index))
sequences = tokenizer.texts_to_sequences(Sentences)

padded = pad_sequences(sequences, padding="post")

print(Sentences[2])

print(sequences[2])

print(padded[2])

print(padded.shape)
x = padded

y = Label

print(len(x))

print(len(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(len(x_train))

print(len(y_train))

print(len(x_test))

print(len(y_test))
EMBEDDING_FILE = '../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'



embedding_index = {}



f = open(EMBEDDING_FILE)



for line in tqdm(f):

    values = line.split()

    word = values[0]

    

    embedding = np.asarray(values[1:], dtype='float32')

    embedding_index[word] = embedding 

    

f.close()
all_emb = np.stack(embedding_index.values())

emb_mean = all_emb.mean()

emb_std = all_emb.std()

emb_size = all_emb.shape[1]

print(emb_size)
num_words = min(max_words, len(word_index))



embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words,emb_size))



for word, i in word_index.items():

    if i>= max_words:

        continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
model = Sequential()

model.add(Embedding(max_words, emb_size, weights = [embedding_matrix], trainable = False))

model.add(CuDNNGRU(128, return_sequences=True))

model.add(Dropout(0.7))

model.add(CuDNNGRU(64, return_sequences=True))

model.add(Dropout(0.5))

model.add(GlobalMaxPool1D())

model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.2, verbose=1 )
res = model.evaluate(x_test, y_test)

print('Test Set \n Loss: {:0.3f} \n Accuracy: {:0.3f}'.format(res[0], res[1]))
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))

fig.suptitle('Performance of model')



ax1.plot(history.history['acc'])

ax1.plot(history.history['val_acc'])

ax1.set_title('Model Accuracy')

ax1.legend(['train', 'test'])



ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

ax2.set_title('Model Loss')

ax2.legend(['train', 'test'])



plt.show()