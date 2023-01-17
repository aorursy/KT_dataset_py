import numpy as np
import pandas as pd
from collections import defaultdict

import keras
import keras.backend as K
from keras.layers import Dense, GlobalAveragePooling1D, Embedding, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
np.random.seed(7)
df = pd.read_csv('./../input/train.csv')
df_test = pd.read_csv('./../input/test.csv')
df_full = df.append(df_test, sort=False)

a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}
y = np.array([a2c[a] for a in df.author])
y = to_categorical(y)
y
counter = {name : defaultdict(int) for name in set(df.author)}
for (text, author) in zip(df.text, df.author):
    text = text.replace(' ', '')
    for c in text:
        counter[author][c] += 1

chars = set()
for v in counter.values():
    chars |= v.keys()
    
names = [author for author in counter.keys()]

print('c ', end='')
for n in names:
    print(n, end='   ')
print()
for c in chars:    
    print(c, end=' ')
    for n in names:
        print(counter[n][c], end=' ')
    print()

def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text
def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    docs = []
    for doc in df.text:
        doc = preprocess(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    
    return docs
min_count = 2

docs = create_docs(df)
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])
num_words
tokenizer = Tokenizer(num_words=num_words, lower=False,  filters='')
tokenizer.fit_on_texts(docs_full)
docs = tokenizer.texts_to_sequences(docs)
print("Samples Number:", len(docs))
print("Sample 1:\n{}\nSample 2:\n{}".format(docs[0], docs[1]))
max_size = 0 
for ind, text in enumerate(docs):
    max_size = len(text) if len(text) > max_size else max_size
    
print("Max number of words in a sample for full dataset:", max_size)
maxlen = 256
docs = pad_sequences(sequences=docs, maxlen=max_size)
print("Samples Number:", len(docs))
print("Sample 1:\n{}\nSample 2:\n{}".format(docs[0][-40:], docs[1][-40:]))
num_words
docs.shape
input_dim =  np.max(docs) + 1
embedding_dims = 20
input_dim
x_train, x_test, y_train, y_test = train_test_split(docs, y, test_size=0.20)# random_state=42)
def create_model(embedding_dims=20, optimizer='adam'):
    
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    #model.add(Dense(20, activation='relu')) #tanh 
    #model.add(Dropout(0.5)) #relu 
    #model.add(Dense(15, activation='relu')) #tanh 
    #model.add(Dropout(0.5)) #relu 
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
model = create_model(embedding_dims=20)
model.summary()
epochs = 500
hist = model.fit(x_train, y_train,
                 batch_size=16,
                 validation_data=(x_test, y_test),
                 epochs=epochs,
                 callbacks=[EarlyStopping(patience=2, monitor='val_loss')])
#model = create_model()
#model.summary()
test_df = pd.read_csv('../input/test.csv')
docs = create_docs(test_df)
docs = tokenizer.texts_to_sequences(docs)
docs = pad_sequences(sequences=docs, maxlen=maxlen)
y = model.predict_proba(docs)

result = pd.read_csv('../input/sample_submission.csv')
for a, i in a2c.items():
    result[a] = y[:, i]
result.to_csv('fastText_result_01.csv', index=False)
