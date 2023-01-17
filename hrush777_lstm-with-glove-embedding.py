# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

print(train_data.shape)

train_data.head()
train_text = train_data['question_text'].values

train_text
import nltk



t_data = list()



for i in range(len(train_text)):

    

    if i % 100000 == 0:

        print(i)



    words = nltk.word_tokenize(train_text[i])



    words=[word.lower() for word in words if word.isalpha()]

    

    # remove single character



    words = [word for word in words if len(word) > 1]

    

    t_data.append(words)
from nltk.stem import WordNetLemmatizer 

  

lemmatizer = WordNetLemmatizer()



data_l = list()

for i in range(len(t_data)):

    temp = list()

    for j in t_data[i]:

        temp.append(lemmatizer.lemmatize(j))

    data_l.append(temp)
vocab = list()



for i in data_l:

    for j in i:

        vocab.append(j)

# no of words in text

len(vocab)
# no of unique words



vocab = set(vocab)

len(vocab)
from keras.preprocessing.text import Tokenizer

# function to build a tokenizer



def tokenization(lines):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer



eng_tokens = tokenization(data_l)

eng_vocab_size = len(eng_tokens.word_index) + 1

print('English Vocabulary Size: %d' % eng_vocab_size)
m = list()

for i in range(len(data_l)):

    m.append(len(data_l[i]))

plt.plot(m)
from keras.preprocessing.sequence import pad_sequences

# encode and pad sequences

def encode_sequences(tokenizer,length,lines):

    # integer encode sequences

    seq = tokenizer.texts_to_sequences(lines)

    # pad sequences with 0 values

    seq = pad_sequences(seq, maxlen=length, padding='post')

    return seq



seq_data = encode_sequences(eng_tokens,60,data_l)
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



print('Found %s word vectors.' % len(embeddings_index))
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = eng_tokens.word_index

nb_words = min(eng_vocab_size - 1, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= eng_vocab_size - 1: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import *

from keras.utils.np_utils import to_categorical



target = train_data['target'].values

target = to_categorical(target)
model = Sequential()

model.add(Embedding(eng_vocab_size - 1,

                    embed_size,

                    weights=[embedding_matrix],

                    input_length=60,

                    trainable=False))

model.add(SpatialDropout1D(0.2))

model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))

model.add(Bidirectional(CuDNNLSTM(32)))

model.add(Dropout(0.25))

model.add(Dense(units=2, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
batch_size = 128

history = model.fit(seq_data, target, epochs=5, batch_size=batch_size, verbose=1, validation_split=0.1)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
test_data = pd.read_csv('../input/test.csv')

print(test_data.shape)

test_data.head()
sam_sub = pd.read_csv('../input/sample_submission.csv')

sam_sub.head()
q_id = test_data['qid'].values

print(q_id.shape)
test_text = test_data['question_text'].values

print(test_text.shape)
t_data_test = list()



for i in range(len(test_text)):

    

    if i % 100000 == 0:

        print(i)



    words = nltk.word_tokenize(test_text[i])



    words=[word.lower() for word in words if word.isalpha()]

    

    # remove single character



    words = [word for word in words if len(word) > 1]

    

    t_data_test.append(words)
data_l_test = list()

for i in range(len(t_data_test)):

    temp = list()

    for j in t_data_test[i]:

        temp.append(lemmatizer.lemmatize(j))

    data_l_test.append(temp)
seq_data_test = encode_sequences(eng_tokens,60,data_l_test)
pred = model.predict_classes(seq_data_test, verbose=1)
pred.shape
q_id = q_id.reshape(-1,1)

print(q_id.shape)

pred = pred.reshape(-1,1)

print(pred.shape)
output = np.array(np.concatenate((q_id, pred), 1))



output = pd.DataFrame(output,columns = ["qid","prediction"])



output.to_csv('submission.csv',index = False)