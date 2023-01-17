# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_colwidth', 400)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

warnings.filterwarnings('ignore')



import os

os.environ['OMP_NUM_THREADS'] = '4'

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

import keras.backend as K
EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
train.head()
X_train = train['comment_text'].values

y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

X_test = test['comment_text'].values
unique_words = {word for sent in X_train for word in sent.split()}.union({word for sent in X_test for word in sent.split()})

print(f'number of unique words in the corpus {len(unique_words)}')
max(map(lambda x: len(x.split()), X_train))
max_words = 40000

max_len = 100

embedding_size = 300
tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(list(X_train) + list(X_test))

for token, idx in zip(list(tokenizer.word_index.keys())[:5], list(tokenizer.word_index.values())[:5]):

    print(f'tokenized {token} ------> {idx}')
X_train_tokenized = tokenizer.texts_to_sequences(X_train)

X_test_tokenized = tokenizer.texts_to_sequences(X_test)

for text, tokenized in zip(X_train[:5], X_train_tokenized[:5]):

    print(f'{text.split()[:5]}... ------> {tokenized[:5]}...')

X_train = X_train_tokenized

X_test = X_test_tokenized
print('lengths of first five examples: ', list(map(lambda x: len(x), X_train[:5])))
X_train_paddded = pad_sequences(X_train, maxlen=max_len, padding='post')

X_test_paddded = pad_sequences(X_test, maxlen=max_len, padding='post')

for train_unpadded, train_padded in zip(X_train[:2], X_train_paddded[:2]):

    print(f'{train_unpadded} ------> {len(train_unpadded)} values + {max_len - len(train_unpadded)} zeros')

X_train = X_train_paddded

X_test = X_test_paddded
word_index = tokenizer.word_index # dictionary of word -> idx

tokenized_words = set(list(word_index.keys())[:max_words]) # we only care about top `max_words` amount of words

embedding_matrix = np.zeros((len(tokenized_words)+1, embedding_size)) # we add a +1 because '0' actually isn't an embedding that we will use. However, our lookup table will still need it

found = 0

with open(EMBEDDING_FILE) as f:

    for line in f:

        word, coord = line.split(' ', 1)

        if word in tokenized_words:

            embedding_matrix[word_index[word]] = np.asarray(coord.split(), dtype='float32')

            found += 1

            

print(f'found {found} of {len(tokenized_words)} words')
class RocAucCallback(Callback):

    def __init__(self, validation_data=(), interval=1):

        super(Callback, self).__init__()

        self.interval = interval

        self.X_val, self.y_val = validation_data

        

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict(self.X_val, verbose=0)

            score - roc_auc_score(self.y_val, y_pred)

            print(f'\n ROC-AUC - epoch {epoch+1} - score: {score}')
class GlobalAveragePooling1DMasked(GlobalAveragePooling1D):

    def call(self, x, mask=None):

        if mask != None:

            # we basically only average over nonzero terms. 

            # Numerator does not change, but the denominator does

            return K.sum(x, axis=1) / K.clip(K.tf.cast(K.tf.count_nonzero(x, axis=1), dtype=K.tf.float32), 1, K.int_shape(x)[1])

        else:

            return super().call(x)
# No real change here, since max value won't be affected by masking.

# There will be a bug if all the values of a row of x are negative, but I do not count on that 

# happening

class GlobalMaxPooling1DMasked(GlobalAveragePooling1D):

    def call(self, x, mask=None):

        if mask != None:

            return K.max(x, axis=1)

        else:

            return super().call(x)
def get_model():

    inp = Input(shape=(max_len, ))

    x = Embedding(max_words+1, embedding_size, weights=[embedding_matrix], mask_zero=True)(inp)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(GRU(80, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1DMasked()(x)

    max_pool = GlobalMaxPooling1DMasked()(x)

    conc = concatenate([avg_pool, max_pool])

    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])



    return model



model = get_model()
batch_size = 32

epochs = 4



file_path="weights_base.best.hdf5"

checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)





callbacks_list = [checkpoint, early] #early



X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, train_size=0.95)

RocAuc = RocAucCallback(validation_data=(X_val, y_val), interval=1)



hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                 callbacks=callbacks_list)



model.load_weights(file_path)
y_pred = model.predict(X_test, batch_size=2048)

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred

submission.to_csv('submission.csv', index=False)