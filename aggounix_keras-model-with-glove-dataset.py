# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.data_utils import get_file



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

question1 = []

question2 = []

is_duplicate = []

question1 = data["question1"].astype('str') 

question2 = data["question2"].astype('str') 

is_duplicate = data["is_duplicate"]



print (len(question1))
MAX_NB_WORDS = 200000

MAX_SEQUENCE_LENGTH = 25

EMBEDDING_DIM = 300

questions = question1 + question2

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(questions)

question1_word_sequences = tokenizer.texts_to_sequences(question1)

question2_word_sequences = tokenizer.texts_to_sequences(question2)

word_index = tokenizer.word_index



print("Words in index: %d" % len(word_index))
from os.path import expanduser, exists

from zipfile import ZipFile

KERAS_DATASETS_DIR = expanduser('~/.keras/datasets/')

GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'

GLOVE_ZIP_FILE = 'glove.840B.300d.zip'

GLOVE_FILE = 'glove.840B.300d.txt'
# download 

if not exists(KERAS_DATASETS_DIR + GLOVE_ZIP_FILE):    

    zipfile = ZipFile(get_file(GLOVE_ZIP_FILE, GLOVE_ZIP_FILE_URL))

    zipfile.extract(GLOVE_FILE, path=KERAS_DATASETS_DIR)

    

print("Processing", GLOVE_FILE)
from os.path import expanduser, exists

from zipfile import ZipFile

KERAS_DATASETS_DIR = expanduser('~/.keras/datasets/')

GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'

GLOVE_ZIP_FILE = 'glove.840B.300d.zip'

GLOVE_FILE = 'glove.840B.300d.txt'

#Not possible to download on kaggle

#if not exists(KERAS_DATASETS_DIR + GLOVE_ZIP_FILE):    

#    zipfile = ZipFile(get_file(GLOVE_ZIP_FILE, GLOVE_ZIP_FILE_URL))

#    zipfile.extract(GLOVE_FILE, path=KERAS_DATASETS_DIR)

    

print("Processing", GLOVE_FILE)
embeddings_index = {}

'''with open(KERAS_DATASETS_DIR + GLOVE_FILE, encoding='utf-8') as f:

    for line in f:

        values = line.split(' ')

        word = values[0]

        embedding = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = embedding'''



print('Word embeddings: %d' % len(embeddings_index))
q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = np.array(is_duplicate, dtype=int)

print('Shape of question1 data tensor:', q1_data.shape)

print('Shape of question2 data tensor:', q2_data.shape)

print('Shape of label tensor:', labels.shape)
nb_words = min(MAX_NB_WORDS, len(word_index))

word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    if i > MAX_NB_WORDS:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        word_embedding_matrix[i] = embedding_vector



print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))
from keras.models import Model

from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization

from keras.layers.embeddings import Embedding

from keras.regularizers import l2

from keras.callbacks import Callback, ModelCheckpoint

from keras import backend as K

from sklearn.model_selection import train_test_split

X = np.stack((q1_data, q2_data), axis=1)

y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)

Q1_train = X_train[:,0]

Q2_train = X_train[:,1]

Q1_test = X_test[:,0]

Q2_test = X_test[:,1]
question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))

question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
MAX_SEQUENCE_LENGTH = 25

EMBEDDING_DIM = 300

VALIDATION_SPLIT = 0.1

TEST_SPLIT = 0.1

RNG_SEED = 13371447

NB_EPOCHS = 25

DROPOUT = 0.1

BATCH_SIZE = 32

q1 = Embedding(nb_words + 1, 

                 EMBEDDING_DIM, 

                 weights=[word_embedding_matrix], 

                 input_length=MAX_SEQUENCE_LENGTH, 

                 trainable=False)(question1)

q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)

q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1)



q2 = Embedding(nb_words + 1, 

                 EMBEDDING_DIM, 

                 weights=[word_embedding_matrix], 

                 input_length=MAX_SEQUENCE_LENGTH, 

                 trainable=False)(question2)

q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)

q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q2)



merged = concatenate([q1,q2])

merged = Dense(200, activation='relu')(merged)

merged = Dropout(DROPOUT)(merged)

merged = BatchNormalization()(merged)

merged = Dense(200, activation='relu')(merged)

merged = Dropout(DROPOUT)(merged)

merged = BatchNormalization()(merged)

merged = Dense(200, activation='relu')(merged)

merged = Dropout(DROPOUT)(merged)

merged = BatchNormalization()(merged)

merged = Dense(200, activation='relu')(merged)

merged = Dropout(DROPOUT)(merged)

merged = BatchNormalization()(merged)



is_duplicate = Dense(1, activation='sigmoid')(merged)



model = Model(inputs=[question1,question2], outputs=is_duplicate)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
MODEL_WEIGHTS = 'question_pairs_weights.h5'

callbacks = [ModelCheckpoint(MODEL_WEIGHTS, monitor='val_acc', save_best_only=True)]

history = model.fit([Q1_train, Q2_train],

                    y_train,

                    epochs=NB_EPOCHS,

                    validation_split=VALIDATION_SPLIT,

                    verbose=2,

                    batch_size=BATCH_SIZE,

                    callbacks=callbacks)

acc = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],

                    'training': history.history['acc'],

                    'validation': history.history['val_acc']})

ax = acc.iloc[:,:].plot(x='epoch', figsize={5,8}, grid=True)

ax.set_ylabel("accuracy")

ax.set_ylim([0.0,1.0]);
max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))

print('Maximum accuracy at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(max_val_acc))
model.load_weights(MODEL_WEIGHTS)

predictions = model.predict([test_q1, test_q2, test_q1, test_q2], verbose = True)
submission = pd.DataFrame(predictions, columns=['is_duplicate'])

submission.insert(0, 'test_id', test.test_id)

file_name = 'submission_{}.csv'.format(min_loss)

submission.to_csv(file_name, index=False)



submission.head(10)