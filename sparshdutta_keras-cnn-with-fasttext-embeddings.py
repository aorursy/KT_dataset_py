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



sns.set_style("whitegrid")

np.random.seed(0)



DATA_PATH = '../input/'

EMBEDDING_DIR = '../input/fasttext-wikinews/'



MAX_NB_WORDS = 100000

tokenizer = RegexpTokenizer(r'\w+')

stop_words = set(stopwords.words('english'))

stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#load embeddings

print('loading word embeddings...')

embeddings_index = {}

f = codecs.open('../input/fasttext-wikinews/wiki-news-300d-1M.vec', encoding='utf-8')

for line in tqdm(f):

    values = line.rstrip().rsplit(' ')

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('found %s word vectors' % len(embeddings_index))
# #load data

# train_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge' + '/train.csv', sep=',', header=0)

# test_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge' + '/test.csv', sep=',', header=0)

# test_df = test_df.fillna('_NA_')



# print("num train: ", train_df.shape[0])

# print("num test: ", test_df.shape[0])



# label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# y_train = train_df[label_names].values



# #visualize word distribution

# train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))

# max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)

# sns.distplot(train_df['doc_len'], hist=True, kde=True, color='b', label='doc len')

# plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')

# plt.title('comment length'); plt.legend()

# plt.show()
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')

newsgroups_test = fetch_20newsgroups(subset='test')



text=[]

label=[]

text.extend(newsgroups_train['data'])

text.extend(newsgroups_test['data'])

label.extend(newsgroups_train['target'])

label.extend(newsgroups_test['target'])



df2=pd.DataFrame({'Text':text,'Label':label})

df2['Text']=df2['Text'].apply(lambda x: x.replace('\n\n',' ').replace('\n',' '))



from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df2, test_size=0.1,stratify = df2['Label'], random_state=42)
def selectN_example_eachClass(dataframe, targetColumn = 'Label', N=10, random_state=42):

    df = pd.DataFrame()

    for classes in np.unique(dataframe[targetColumn]):

        proc_df = dataframe.loc[dataframe[targetColumn] == classes,].sample(N, random_state=random_state)

        df = pd.concat([df,proc_df], axis=0)

    

    df = df.sample(frac=1, random_state=random_state)

        

    return df
df2['Label'].value_counts()
train_data=selectN_example_eachClass(df2, targetColumn='Label', N=1, random_state=42)

test_data=df2.drop(train_data.index)



X_train=train_data

X_test=test_data

# y_train=train_data['Label']

# y_test=test_data['Label']
# raw_docs_train = train_df['comment_text'].tolist()

# raw_docs_test = test_df['comment_text'].tolist() 

# num_classes = len(label_names)



raw_docs_train = X_train['Text'].tolist()

raw_docs_test = X_test['Text'].tolist() 

num_classes = 20





#visualize word distribution

X_train['doc_len'] = X_train['Text'].apply(lambda words: len(words.split(" ")))

max_seq_len = np.round(X_train['doc_len'].mean() + X_train['doc_len'].std()).astype(int)



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

weight_decay = 1e-2 #1e-4
MAX_NB_WORDS
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
from keras.layers import GlobalAveragePooling1D
model = Sequential()

model.add(Embedding(nb_words, embed_dim,

          weights=[embedding_matrix], input_length=max_seq_len, trainable=False))

model.add(GlobalAveragePooling1D())

model.add(Dense(num_classes, activation='softmax'))

adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()
# #CNN architecture

# print("training CNN ...")

# model = Sequential()

# model.add(Embedding(nb_words, embed_dim,

#           weights=[embedding_matrix], input_length=max_seq_len, trainable=False))

# model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))

# model.add(MaxPooling1D(2))

# model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))

# model.add(GlobalMaxPooling1D())

# model.add(Dropout(0.5))

# model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

# model.add(Dense(num_classes, activation='softmax'))  #multi-label (k-hot encoding)



# adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# model.summary()



# # lr=0.001 -> TestAccuracy - 0.75 epochs = 20 TrainSize = 16961, TestSize = 1885 MaxSeqLen =1081

# # lr=0.01 -> TestAccuracy - 0.82 epochs = 20  TrainSize = 16961, TestSize = 1885 MaxSeqLen =1081



#define callbacks

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)

callbacks_list = [early_stopping]
from keras.utils import to_categorical
print(word_seq_train.shape)

print(word_seq_test.shape)

print(max_seq_len)
from collections import Counter
#callbacks=callbacks_list,
#model training

hist = model.fit(word_seq_train, to_categorical(X_train['Label'].values), batch_size=batch_size, epochs=5000,  validation_split=0.0, shuffle=True, verbose=2)
# #model training

# hist = model.fit(word_seq_train, to_categorical(X_train['Label'].values), batch_size=batch_size, epochs=20,  validation_split=0.4, shuffle=True, verbose=2)
max_seq_len
from sklearn.metrics import classification_report
print(classification_report(X_train['Label'], np.argmax(model.predict(word_seq_train), axis=1), target_names=newsgroups_train.target_names))
print(classification_report(X_test['Label'], np.argmax(model.predict(word_seq_test), axis=1), target_names=newsgroups_train.target_names))
#generate plots

plt.figure()

plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')

plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')

plt.title('CNN News Group Categories')

plt.xlabel('Epochs')

plt.ylabel('Cross-Entropy Loss')

plt.legend(loc='upper right')

plt.show()
plt.figure()

plt.plot(hist.history['acc'], lw=2.0, color='b', label='train')

plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')

plt.title('CNN News Group Categories')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc='upper left')

plt.show()