import gc

import re

import operator 



import numpy as np

import pandas as pd



from gensim.models import KeyedVectors



from sklearn import model_selection



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, Input, Dense, CuDNNGRU,concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.layers import LeakyReLU, CuDNNLSTM

from keras.optimizers import RMSprop, Adam

from keras.models import Model, Sequential

from keras.callbacks import EarlyStopping



import seaborn as sns
import os

print(os.listdir("../input"))
train = pd.read_csv("../input/outprocess3/train_preprocess2.csv")

test = pd.read_csv("../input/outprocess3/test_preprocess2.csv")

print("Train shape : ",train.shape)

test.head()

train_orig = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")

train_orig.head()
train = pd.concat([train,train_orig[['target']]],axis=1)

train.head()
del(train_orig)

gc.collect()
train['target'] = np.where(train['target'] >= 0.5, True, False)
train_df, validate_df = model_selection.train_test_split(train, test_size=0.1)

print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))
train_df.head()
train_df['comment_text'].describe()
MAX_NUM_WORDS = 100000

TOXICITY_COLUMN = 'target'

TEXT_COLUMN = 'comment_text'



# Create a text tokenizer.

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(train_df[['comment_text']])



# All comments must be truncated or padded to be the same length.

MAX_SEQUENCE_LENGTH = 256

def pad_text(texts, tokenizer):

    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)
gc.collect()
EMBEDDINGS_DIMENSION = 300

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,EMBEDDINGS_DIMENSION))
ft_common_crawl = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

embeddings_index = KeyedVectors.load_word2vec_format(ft_common_crawl)
num_words_in_embedding = 0



for word, i in tokenizer.word_index.items():

    if word in embeddings_index.vocab:

        embedding_vector = embeddings_index[word]

        embedding_matrix[i] = embedding_vector        

        num_words_in_embedding += 1
train_text = pad_text(train_df[[TEXT_COLUMN]], tokenizer)

train_labels = train_df[[TOXICITY_COLUMN]]

validate_text = pad_text(validate_df[[TEXT_COLUMN]], tokenizer)

validate_labels = validate_df[[TOXICITY_COLUMN]]
gc.collect()
NODES = 64

vocab_size = len(tokenizer.word_index) + 1





model = Sequential()



model.add(Embedding(vocab_size,EMBEDDINGS_DIMENSION,input_length = MAX_SEQUENCE_LENGTH,weights = [embedding_matrix],trainable = False))



model.add(Bidirectional(CuDNNLSTM(100,return_sequences=True)))

model.add(Conv1D(64,7,padding='same'))

model.add(GlobalAveragePooling1D())



model.add(Dense(128))

model.add(LeakyReLU())



model.add(Dense(NODES,activation = 'relu'))



model.add(Dense(1,activation = 'sigmoid'))



model.summary()

model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics = ['accuracy'])
BATCH_SIZE = 1024

NUM_EPOCHS = 100
model.fit(

    train_text,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=NUM_EPOCHS,

    validation_data=(validate_text, validate_labels),

    callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)])

submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

submission['prediction'] = model.predict(pad_text(test[TEXT_COLUMN], tokenizer))

submission.reset_index(drop=False, inplace=True)

submission.head()


submission.to_csv('submission.csv', index=False)