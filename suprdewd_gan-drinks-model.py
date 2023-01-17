from json import loads

import pandas as pd

from itertools import chain

from dask import bag

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#drink_df = pd.read_csv('../input/all_drinks.csv')

#drink_df.sample(3)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical
#str_vec = drink_df['strDrink'].str.lower()

#MAX_NB_WORDS, MAX_SEQUENCE_LENGTH = 100, 40

#tokenizer = Tokenizer(num_words=MAX_NB_WORDS, char_level=True)

#tokenizer.fit_on_texts(str_vec)

#train_sequences = tokenizer.texts_to_sequences(str_vec)

#train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)

#char_index = tokenizer.word_index

#print('Found %s unique tokens.' % len(char_index))

#train_x = np.stack([to_categorical(x, num_classes=len(char_index)+1) for x in train_data],0)

#print(train_x.shape)
#print(str_vec[0])

#print(train_sequences[0])

#print(train_data[0])

#print(train_x[0][-1])
train_data = pd.read_csv('../input/devresearch-xydata/X_train.txt', sep=' ', header=None).values
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

#all_ingred = drink_df[[x for x in drink_df.columns 

#                       if 'Ingredient' in x]].apply(lambda c_row: [v.lower() for k,v in c_row.items() if not isempty(v)],1)

#all_ingred[0:3]



all_ingred = pd.read_csv('../input/devresearch-xydata/y_train.txt', sep=' ', header=None)[0].apply(lambda x: [x])
from sklearn.preprocessing import LabelEncoder



ingred_label = LabelEncoder()

ingred_label.fit(list(chain(*all_ingred.values)))

print('Found', len(ingred_label.classes_), 'unique ingredients, ', ingred_label.classes_)
y_vec = np.stack(all_ingred.map(lambda x: np.sum(to_categorical(ingred_label.transform(x), 

                                        num_classes=len(ingred_label.classes_)),0)),0).clip(0,1)
#print(all_ingred[1])

#print(ingred_label.transform(all_ingred[1]))

#print(ingred_label.inverse_transform([147]))
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(range(y_vec.shape[0]), 

                                                    random_state = 12345,

                                                   train_size = 0.7)
from keras.models import Sequential

from keras.layers import Embedding, Dense, Dropout, Masking, Conv1D, GlobalMaxPooling1D

from keras.optimizers import Adam

#simple_embed_lay = Embedding(len(char_index)+1, len(char_index)+1, 

#                                    mask_zero = False, 

#                                    weights = [np.eye(len(char_index)+1)], # start with a 1-1 weighting

#                            name = '1_1_Mapping'       

#                            )





# XXX: I just changed len(char_index)+1 to 3, although I don't know if that makes any sense...

simple_embed_lay = Embedding(3, 3, 

                                    mask_zero = False, 

                                    weights = [np.eye(3)], # start with a 1-1 weighting

                            name = '1_1_Mapping'       

                            )



simple_embed_lay.trainable = False

simple_sequence_model = Sequential(name = 'Generator')

simple_sequence_model.add(Masking(0, input_shape = (None,)))

simple_sequence_model.add(simple_embed_lay)

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
from keras.models import Model

from keras.layers import Input, concatenate

seq_in = Input(shape = (None,), name = 'Name_Input')

ingred_in = Input(shape = (y_vec.shape[1],), name = 'Ingredients_Input')

seq_proc = simple_embed_lay(Masking(0, input_shape = (None,))(seq_in))

seq_feat = Conv1D(64, kernel_size = (3,), strides = 1, padding = 'same')(seq_proc)

seq_feat = Conv1D(128, kernel_size = (3,), strides = 2, padding = 'same')(seq_feat)

seq_feat = Conv1D(256, kernel_size = (3,), strides = 2, padding = 'same')(seq_feat)

seq_gap_feat = Dense(y_vec.shape[1], activation='relu')(GlobalMaxPooling1D()(seq_feat))

all_feat = Dropout(0.5)(concatenate([seq_gap_feat, ingred_in]))



out_layer = Dense(2, activation = 'softmax')(all_feat)



disc_model = Model(inputs = [seq_in, ingred_in],

                  outputs = [out_layer], name = 'Discriminator')

disc_model.compile(loss = 'categorical_crossentropy', 

                  optimizer = 'adam',

                  metrics = ['acc'])

disc_model.summary()
seq_in = Input(shape = (None,), name = 'Name_Input')

gen_output = simple_sequence_model(seq_in)

ingred_in = Input(shape = (y_vec.shape[1],), name = 'Ingredients_Input')

disc_output = disc_model([seq_in, gen_output])



comb_model = Model(inputs = [seq_in, ingred_in],

                  outputs = [disc_output])

comb_model.layers[-1].trainable = False

print('a')

print(comb_model.layers[-1].summary())

comb_model.layers[-1].compile(loss = 'categorical_crossentropy', 

                  optimizer = 'adam',

                  metrics = ['acc'])

print('b')

print(comb_model.layers[-1].summary())

comb_model.compile(loss = 'categorical_crossentropy', 

                  optimizer = 'adam',

                  metrics = ['acc'])

comb_model.summary()
gen_loss = []

disc_loss = []

def show_loss(loss_history, prefix):

    epich = np.cumsum(np.concatenate(

        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    _ = ax1.semilogy(epich,

                 np.concatenate([mh.history['loss'] for mh in loss_history]),

                 'b-',

                 epich, np.concatenate(

            [mh.history['val_loss'] for mh in loss_history]), 'r-')

    ax1.legend(['Training', 'Validation'])

    ax1.set_title('%s Loss' % prefix)



    _ = ax2.plot(epich, np.concatenate(

        [mh.history['acc'] for mh in loss_history]), 'b-',

                     epich, np.concatenate(

            [mh.history['val_acc'] for mh in loss_history]),

                     'r-')

    ax2.legend(['Training', 'Validation'])

    ax2.set_title('%s Accuracy' % prefix)
verbose = 0

from tqdm import tqdm

for i in tqdm(range(80)):

    # train the generator directly (optional step)

    simple_sequence_model.fit(train_data[train_idx], 

                              y_vec[train_idx], 

                              verbose = verbose, epochs = 2)

    # improve the discriminator

    gen_ing_train = simple_sequence_model.predict(train_data[train_idx])

    gen_ing_test = simple_sequence_model.predict(train_data[test_idx])

    disc_train_input = [

        np.concatenate([train_data[train_idx],train_data[train_idx]],0),

        np.concatenate([y_vec[train_idx],gen_ing_train],0)

    ]



    disc_test_input = [

        np.concatenate([train_data[test_idx],train_data[test_idx]],0),

        np.concatenate([y_vec[test_idx],gen_ing_test],0)

    ]



    disc_train_output = to_categorical(np.concatenate([np.ones((len(train_idx))), np.zeros((len(train_idx)))],0))

    disc_test_output = to_categorical(np.concatenate([np.ones((len(test_idx))), np.zeros((len(test_idx)))],0))



    disc_loss += [disc_model.fit(disc_train_input, disc_train_output,

                   validation_data = (disc_test_input, disc_test_output),

                   verbose = verbose, shuffle = True)]

    # improve the generator

    gen_loss += [comb_model.fit([train_data[train_idx], y_vec[train_idx]], 

                   to_categorical(np.ones(len(train_idx))),

                   validation_data = ([train_data[test_idx], y_vec[test_idx]], to_categorical(np.ones(len(test_idx)))),

                   verbose = verbose, epochs = 2,

                                shuffle = True)]
show_loss(gen_loss, 'Generator')

show_loss(disc_loss, 'Discriminator')
pred_vec = simple_sequence_model.predict(train_data[test_idx])

print('Mean Error %2.2f%%' % (100*mean_absolute_error(y_vec[test_idx][1:2], pred_vec[1:2])))

print('test', y_vec[test_idx][1:2])

print(train_data[0])

print(train_data[1])

print('out',simple_sequence_model.predict(train_data[0][0:1]))

print('out2',simple_sequence_model.predict(train_data[1][0:1]))
train_data[328]
print('Input Name:', drink_df['strDrink'].values[test_idx[0]])

print('Real Ingredients', all_ingred.values[test_idx[0]])



proc_pred = lambda out_pred: sorted([(ingred_label.inverse_transform([idx]), out_pred[idx])

                              for idx in np.where(out_pred>0)[0]], key = lambda x: -x[1])



print('Predicted Ingredients')

for _, (i,j) in zip(range(5), proc_pred(pred_vec[0])):

    print('%25s\t\t%2.2f%%' % (i,100*j))
print(pred_vec[0])

print(pred_vec[1])

print(pred_vec[100])
for rand_idx in np.random.choice(range(len(test_idx)), size = 3):

    print('Input Name:', drink_df['strDrink'].values[test_idx[rand_idx]])

    print('Real Ingredients', all_ingred.values[test_idx[rand_idx]])



    proc_pred = lambda out_pred: sorted([(ingred_label.inverse_transform([idx]), out_pred[idx])

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