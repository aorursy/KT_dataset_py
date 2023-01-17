import numpy as np

from sklearn.model_selection import train_test_split

import pandas as pd





from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Embedding

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers.merge import concatenate

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.vis_utils import plot_model

from numpy import array



def get_data(train_file, test_file = None):

    if test_file == None:

        frame = pd.read_csv(train_file)

        data = frame.values

        np.random.shuffle(data)

        return data

    else:

        train_frame = pd.read_csv(train_file)

        test_frame = pd.read_csv(test_file)



        train_data = train_frame.values

        test_data = test_frame.values

        np.random.shuffle(train_data)

        np.random.shuffle(test_data)



        return train_data, test_data



def get_training_testing_sets(train_file, test_file = None):

    if test_file == None:

        data = get_data(train_file)

        train_data, test_data = train_test_split(data)

    else:



        train_data, test_data = get_data(train_file, test_file)



    X_train = train_data[:, 1]

    Y_train = train_data[:, 0]

    X_test = test_data[:, 1]

    Y_test = test_data[:, 0]



    print(X_train.shape, X_test.shape)

    

    return X_train, Y_train, X_test, Y_test



def get_tokenizer(lines):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer



def max_length(lines):

    return max([len(sentence.split()) for sentence in lines])



def encode_text(tokenizer, lines, length):

    encoded = tokenizer.texts_to_sequences(lines)

    padded = pad_sequences(encoded, maxlen=length, padding='post')

    return padded



def define_model(length, vocab_size, channels, kernel_size):

    inputs = {}

    embedding = {}

    conv = {}

    drop = {}

    pool = {}

    flat = {}

    for channel in range(1, channels + 1):

        inputs[channel] = Input(shape = (length,))

        embedding[channel] = Embedding(vocab_size, 100)(inputs[channel])

        conv[channel] = Conv1D(filters = 32, kernel_size = kernel_size[channel], activation = 'relu')(embedding[channel])

        drop[channel] = Dropout(0.5)(conv[channel])

        pool[channel] = MaxPooling1D(pool_size = 2)(drop[channel])

        flat[channel] = Flatten()(pool[channel])

    merged = concatenate(list(flat.values()))

    dense = Dense(10, activation = 'relu')(merged)

    outputs = Dense(1, activation = 'sigmoid')(dense)

    

    model = Model(list(inputs.values()), outputs=outputs)

    

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    

    print(model.summary())

    plot_model(model, show_shapes = True, to_file = 'multichannel.png')

    return model





X_train, Y_train, X_test, Y_test = get_training_testing_sets('../input/SPAM text message 20170820 - Data.csv')

for i in range(Y_train.shape[0]):

    Y_train[i] = (Y_train[i] == 'spam')



for i in range(Y_test.shape[0]):

    Y_test[i] = (Y_test[i] == 'spam')





tokenizer = get_tokenizer(X_train)

length = max_length(X_train)

vocab_size = len(tokenizer.word_index) + 1

X_train = encode_text(tokenizer, X_train, length)

model = define_model(length, vocab_size, 3, {1 : 8, 2 : 6, 3 : 4})

model.fit([X_train, X_train, X_train], array(Y_train), epochs = 20, batch_size = 16)



tokenizer = get_tokenizer(X_test)

vocab_size = len(tokenizer.word_index) + 1

X_test = encode_text(tokenizer, X_test, length)

loss, acc = model.evaluate([X_test,X_test,X_test],array(Y_test), verbose=0)



model.save('model.h5')

print(acc)