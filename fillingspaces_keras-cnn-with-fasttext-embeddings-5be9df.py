import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import keras

from keras import optimizers

from keras import backend as K

from keras import regularizers

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 

from keras.utils import plot_model

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping



from tqdm import tqdm

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer 

import os, re, csv, math, codecs

from sklearn.metrics import f1_score

sns.set_style("whitegrid")

np.random.seed(0)



DATA_PATH = '../input/'

EMBEDDING_DIR = '../input/'



MAX_NB_WORDS = 100000

tokenizer = RegexpTokenizer(r'\w+')

stop_words = set(stopwords.words('english'))

stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#load embeddings

print('loading word embeddings...')

embeddings_index = {}

f = codecs.open('../input/fasttext/wiki.simple.vec', encoding='utf-8')

for line in tqdm(f):

    values = line.rstrip().rsplit(' ')

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('found %s word vectors' % len(embeddings_index))
#load data

train_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge' + '/train.csv', sep=',', header=0)

test_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge' + '/test.csv', sep=',', header=0)

test_df = test_df.fillna('_NA_')



print("num train: ", train_df.shape[0])

print("num test: ", test_df.shape[0])



label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_train = train_df[label_names].values



#visualize word distribution

train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))

max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)

sns.distplot(train_df['doc_len'], hist=True, kde=True, color='b', label='doc len')

plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')

plt.title('comment length'); plt.legend()

plt.show()
raw_docs_train = train_df['comment_text'].tolist()

raw_docs_test = test_df['comment_text'].tolist() 

num_classes = len(label_names)



print("pre-processing train data...")

processed_docs_train = []

for doc in tqdm(raw_docs_train):

    tokens = tokenizer.tokenize(doc)

    filtered = [word for word in tokens if word not in stop_words]

    processed_docs_train.append(" ".join(filtered))

#end for



processed_docs_test = []

for doc in tqdm(raw_docs_test):

    tokens = tokenizer.tokenize(doc)

    filtered = [word for word in tokens if word not in stop_words]

    processed_docs_test.append(" ".join(filtered))

#end for



print("tokenizing input data...")

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)

tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky

word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)

word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)

word_index = tokenizer.word_index

print("dictionary size: ", len(word_index))



#pad sequences

word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)

word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)
#training params

batch_size = 256 

num_epochs = 8 



#model parameters

num_filters = 64 

embed_dim = 300 

weight_decay = 1e-4
#embedding matrix

print('preparing embedding matrix...')

words_not_found = []

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_dim))

for word, i in word_index.items():

    if i >= nb_words:

        continue

    embedding_vector = embeddings_index.get(word)

    if (embedding_vector is not None) and len(embedding_vector) > 0:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

    else:

        words_not_found.append(word)

print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
print("sample words not found: ", np.random.choice(words_not_found, 10))
#CNN architecture

print("training CNN ...")

model = Sequential()

model.add(Embedding(nb_words, embed_dim,

          weights=[embedding_matrix], input_length=max_seq_len, trainable=False))

model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))

model.add(MaxPooling1D(2))

model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))

model.add(GlobalMaxPooling1D())

model.add(Dropout(0.5))

model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Dense(num_classes, activation='sigmoid'))  #multi-label (k-hot encoding)



adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()
#define callbacks

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)

callbacks_list = [early_stopping]
#model training

hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2)
score, acc = model.evaluate(word_seq_train, y_train,

                            batch_size=batch_size)



print(score)

print(acc)



y_test = model.predict(word_seq_train)

print("num test: ", y_test.shape[0])

#generate plots

plt.figure()

plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')

plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')

plt.title('CNN sentiment')

plt.xlabel('Epochs')

plt.ylabel('Cross-Entropy Loss')

plt.legend(loc='upper right')

plt.show()

print(hist.history['loss'])
plt.figure()

plt.plot(hist.history['acc'], lw=2.0, color='b', label='train')

plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')

plt.title('CNN sentiment')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc='upper left')

plt.show()

print(hist.history['val_acc'])