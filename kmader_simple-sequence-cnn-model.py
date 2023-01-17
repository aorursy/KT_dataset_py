from json import loads

import pandas as pd

from itertools import chain

from dask import bag

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
drink_df = pd.read_csv('../input/all_drinks.csv')

drink_df.sample(3)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical
str_vec = drink_df['strDrink'].str.lower()

MAX_NB_WORDS, MAX_SEQUENCE_LENGTH = 100, 40

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, char_level=True)

tokenizer.fit_on_texts(str_vec)

train_sequences = tokenizer.texts_to_sequences(str_vec)

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)

char_index = tokenizer.word_index

print('Found %s unique tokens.' % len(char_index))

train_x = np.stack([to_categorical(x, num_classes=len(char_index)+1) for x in train_data],0)

print(train_x.shape)
def isempty(x):

    try:

        if x is None: 

            return True

        elif len(x)<1:

            return True

        else:

            return False

    except:

        # floating point nans

        return True

all_ingred = drink_df[[x for x in drink_df.columns 

                       if 'Ingredient' in x]].apply(lambda c_row: [v.lower() for k,v in c_row.items() if not isempty(v)],1)

all_ingred[0:3]
from sklearn.preprocessing import LabelEncoder



ingred_label = LabelEncoder()

ingred_label.fit(list(chain(*all_ingred.values)))

print('Found', len(ingred_label.classes_), 'unique ingredients, ', ingred_label.classes_[0:3])
y_vec = np.stack(all_ingred.map(lambda x: np.sum(to_categorical(ingred_label.transform(x), 

                                        num_classes=len(ingred_label.classes_)),0)),0).clip(0,1)
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(range(y_vec.shape[0]), 

                                                    random_state = 12345,

                                                   train_size = 0.7)
from keras.models import Sequential

from keras.layers import Embedding, Dense, Dropout, Masking, Conv1D, GlobalMaxPooling1D

from keras.optimizers import Adam

simple_sequence_model = Sequential()

simple_sequence_model.add(Masking(0, input_shape = (None,)))

simple_sequence_model.add(Embedding(len(char_index)+1, 32))

simple_sequence_model.add(Conv1D(64, kernel_size = (3,), strides = 1, padding = 'same'))

simple_sequence_model.add(Conv1D(128, kernel_size = (3,), strides = 2, padding = 'same'))

simple_sequence_model.add(Conv1D(256, kernel_size = (3,), strides = 2, padding = 'same'))

simple_sequence_model.add(Conv1D(512, kernel_size = (3,), strides = 1, padding = 'same'))

simple_sequence_model.add(GlobalMaxPooling1D())

simple_sequence_model.add(Dropout(0.25))

simple_sequence_model.add(Dense(y_vec.shape[1], 

                                activation = 'sigmoid'))

simple_sequence_model.compile(loss = 'binary_crossentropy', # categorical and mae don't work well here

                              optimizer = Adam(lr = 5e-4, decay = 1e-6), 

                             metrics = ['mae'])

simple_sequence_model.summary()
simple_sequence_model.fit(train_data[train_idx], y_vec[train_idx], epochs=10,

                          batch_size = 32,

                         validation_data = (train_data[test_idx], y_vec[test_idx]), 

                          verbose = 1)

pred_vec = simple_sequence_model.predict(train_data[test_idx])



print('Mean Error %2.2f%%' % (100*mean_absolute_error(y_vec[test_idx], pred_vec)))
print('Input Name:', drink_df['strDrink'].values[test_idx[0]])

print('Real Ingredients', all_ingred.values[test_idx[0]])



proc_pred = lambda out_pred: sorted([(ingred_label.inverse_transform(idx), out_pred[idx])

                              for idx in np.where(out_pred>0)[0]], key = lambda x: -x[1])



print('Predicted Ingredients')

for _, (i,j) in zip(range(5), proc_pred(pred_vec[0])):

    print('%25s\t\t%2.2f%%' % (i,100*j))
rchar = lambda : np.random.choice(list(char_index.keys()))

SENTENCE_SWAP = 0.25

ADD_LETTERS = 0.95

DEL_LETTERS = 0.9

def tweak_sequence_gen(verbose = False):

    while True:

        c_train_idx = np.random.permutation(train_idx)

        s_str = str_vec.values[c_train_idx]

        if verbose: 

            print(s_str[0])

        # randomly reorder strings

        s_str = [(' '.join(np.random.permutation(x.split(' '))) if np.random.uniform(0,1)>SENTENCE_SWAP else x)

                      for x in s_str]

        if verbose: 

            print(s_str[0])

        # randomly add letters

        s_str = [''.join([(rchar() if np.random.uniform(0,1)>ADD_LETTERS else '') + y for y in x])

                      for x in s_str]

        if verbose: 

            print(s_str[0])

        # randomly delete letters

        s_str = [''.join([y for y in x if np.random.uniform(0,1)>(1-DEL_LETTERS)])

                      for x in s_str]

        if verbose: 

            print(s_str[0])

        t_seq = tokenizer.texts_to_sequences(s_str)

        t_data = pad_sequences(t_seq, maxlen=MAX_SEQUENCE_LENGTH)

        yield t_data, y_vec[c_train_idx]

for _, (x,y) in zip(range(1), tweak_sequence_gen(True)):

    print(x[0])
epochs = 60

# use a for loop so we can control batch size

for _, (c_bx, c_by) in zip(range(epochs), tweak_sequence_gen()):

    simple_sequence_model.fit(c_bx, c_by,

                                        batch_size = 32,

                                        epochs=1,

                             validation_data = (train_data[test_idx], y_vec[test_idx]), 

                              verbose = 0)



pred_vec = simple_sequence_model.predict(train_data[test_idx])

print('Mean Error %2.2f%%' % (100*mean_absolute_error(y_vec[test_idx], pred_vec)))
for rand_idx in np.random.choice(range(len(test_idx)), size = 3):

    print('Input Name:', drink_df['strDrink'].values[test_idx[rand_idx]])

    print('Real Ingredients', all_ingred.values[test_idx[rand_idx]])



    proc_pred = lambda out_pred: sorted([(ingred_label.inverse_transform(idx), out_pred[idx])

                                  for idx in np.where(out_pred>0)[0]], key = lambda x: -x[1])



    print('Predicted Ingredients')

    for _, (i,j) in zip(range(5), proc_pred(pred_vec[rand_idx])):

        print('%25s\t\t%2.2f%%' % (i,100*j))

    print('')
def predict_from_name(in_drink_name):

    seq_arr = np.array(tokenizer.texts_to_sequences([in_drink_name.lower()]))

    c_pred = simple_sequence_model.predict(seq_arr)

    for _, (i,j) in zip(range(5), proc_pred(c_pred[0])):

        print('%25s\t\t%2.2f%%' % (i,100*j))
predict_from_name('super fancy drink')
predict_from_name('hopping hippo')
predict_from_name('kevs special')