import pandas as pd

import numpy as np

from IPython.display import display
# read in data

# train_raw_path = '\\\\SEAGATE-D4/Documents/My Hoang Nguyen/ML-SDrive/Kaggle/Mercari/train.tsv'

train_raw_path = '../input/mercari-price-suggestion-challenge/train.tsv'

train_raw = pd.read_table(train_raw_path)



# test_raw_path = '\\\\SEAGATE-D4/Documents/My Hoang Nguyen/ML-SDrive/Kaggle/Mercari/test.tsv'

test_raw_path = '../input/mercari-price-suggestion-challenge/test.tsv'

test_raw = pd.read_table(test_raw_path)
print('train_raw\n', train_raw.shape)

print(train_raw.dtypes)

display(train_raw.head())



print('test_raw\n', test_raw.shape)

print(test_raw.dtypes)

display(test_raw.head())
# load Google's pre-trained word2vec

import gensim

# pretrained_word2vec_path = '\\\\SEAGATE-D4/Documents/My Hoang Nguyen/ML-SDrive/Sentiment Analysis/data/GoogleNews-vectors-negative300.bin'

pretrained_word2vec_path = '../input/word2vecnegative300/GoogleNews-vectors-negative300.bin'

word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_word2vec_path, binary=True)
# tokenize text

from keras.preprocessing.text import Tokenizer



n_words = 20 # top most common words

text = train_raw['item_description'].astype(str).tolist()

tokenizer = Tokenizer(num_words=n_words)

tokenizer.fit_on_texts(text)



# pad_sequences so they are all of the same length

from keras.preprocessing.sequence import pad_sequences



sequences = tokenizer.texts_to_sequences(text) # list, same length as data. represent word as rank/index

padded_seq = pad_sequences(sequences)

print('padded_seq.shape', padded_seq.shape)
# kfold cv

from sklearn.model_selection import KFold



x = padded_seq

y = np.asarray(train_raw['price'])



n_splits = 2



kf = KFold(n_splits=n_splits)

kf.get_n_splits(x)
# create embedding_matrix to feed in as weights for embedding_layer

word_index = tokenizer.word_index

vocab_size = len(word_index)

EMBEDDING_DIM = 300 # this is from the pretrained vectors



embedding_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    if word in word2vec:

        embedding_vector = word2vec[word]

    else:

        embedding_vector = None

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
# custom loss function

import keras.backend as K



def rmsle(y_true, y_pred):

    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)

    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)

    

    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))



import keras

def truncated_normal(seed):

    return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=seed)
# create cnn model

from keras.layers import Embedding, Dense, Input, Flatten

from keras.layers import Conv1D, GlobalMaxPooling1D, Concatenate, Dropout

from keras.models import Model



# parameters

input_length = padded_seq.shape[1] # len (num words) of longest description

seed = 0

filter_sizes = [2,3]

n_filters = 2

dropout_prob = 0.5



def create_cnn(include_top=True, weights=None):    



    # input

    sequence_input = Input(shape=(input_length,), dtype='int32', name='input')

    # embedding_layer

    embedding_layer = Embedding(vocab_size + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=input_length

                                , name='embedding', trainable=False)(sequence_input)

    # conv layer

    features = []

    i = 0

    for filter_size in filter_sizes:

        i += 1

        # conv layer

        conv = Conv1D(n_filters, filter_size, activation='relu', kernel_initializer=truncated_normal(seed)

                      , name='conv'+str(i))(embedding_layer)

        # global max pooling

        conv = GlobalMaxPooling1D(name='pool'+str(i))(conv)

        # add features together

        features.append(conv)

    # penultimate layer

    nn = Concatenate(name='features')(features)

    if include_top:

        # dropout

        nn = Dropout(dropout_prob, seed=seed, name='dropout')(nn)

        # fully connected layer

        preds = Dense(1, kernel_initializer=truncated_normal(seed), name='output')(nn)



        model = Model(sequence_input, preds)

    else:

        model = Model(sequence_input, nn)

    

    

    if weights is not None:

        model.set_weights(weights)

        

    return model
model = create_cnn()

model.summary()
# train cnn

model.compile(loss='msle', optimizer='adadelta', metrics=[rmsle])



batch_size = 128

epochs = 1

for train_index, val_index in kf.split(x):

    x_train, x_val = x[train_index], x[val_index]

    y_train, y_val = y[train_index], y[val_index]

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)
# predict

# format test data

data = test_raw['item_description'].astype(str).tolist()

sequences = tokenizer.texts_to_sequences(data) # list, same length as data. represent word as rank/index

x_test = pad_sequences(sequences, maxlen=input_length)

print('x_test.shape', x_test.shape)



# predict

y_test = model.predict(x_test, batch_size=batch_size)

print('y_test.shape', y_test.shape)
# submission

submission = pd.concat((test_raw, pd.DataFrame(np.reshape(y_test,(-1,1)), columns=['price'])), axis=1)[['test_id', 'price']]

display(submission.head())



# export submission

submission.to_csv('first_submission.csv', index=False)
# export submission

submission.to_csv('first_submission.csv', index=False)