import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras import Model

from tensorflow.keras.layers import Input, Embedding, Dense, Activation, Flatten, GRU, Masking
# reading data

file_path = '/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv'

data = pd.read_csv(file_path).to_numpy()

file_length = len(data)
# parameters for training & testing

vocabulary_size = 10000

sequence_length = 200

train_test_split = 0.8

output_embeddings = 60

dense_layer_1_size = 70

dense_layer_2_size = 2

batch_size = 1000

steps_per_epoch = 40

epochs = 7
# separating data

np.random.shuffle(data)

separator_index = int(file_length * train_test_split)



train_data = data[:separator_index]

test_data = data[separator_index:]
# using keras preprocessing to get tokens 

tokenizer = Tokenizer(vocabulary_size)

tokens = tokenizer.fit_on_texts(train_data[:,0])
# method that prepares model input data

def get_model_input_data(text_data, labels = None):

    sequences = tokenizer.texts_to_sequences(text_data)

    model_input_array = np.zeros((len(sequences), sequence_length))

    for i, sequence in enumerate(sequences):

        model_input_array[i,:min(len(sequence), 200)] = sequence[:200]

    

    if labels is None:

        return model_input_array

    else:

        return (model_input_array, np.array([1. if label == 'positive' else 0. for label in labels]))



#get_model_input_data(train_data[:20,0], train_data[:20, 1])[1]
# method that returns batch

def get_train_data():

    while True:

        batch = np.random.permutation(train_data)[:batch_size]

        yield get_model_input_data(batch[:,0], batch[:,1])
# model

inputs = Input((sequence_length))

x = Embedding(vocabulary_size, output_embeddings)(inputs)

x = Flatten()(x)

x = Dense(dense_layer_1_size, activation="relu")(x)

x = Dense(dense_layer_2_size)(x)

outputs = Activation("softmax")(x)



model = Model(inputs=inputs, outputs=outputs)

model.summary()



# training definition

model.compile(

    loss = tf.keras.losses.SparseCategoricalCrossentropy(), #"categorical_crossentropy"

    optimizer = tf.keras.optimizers.Adam(), #"adam"

    metrics = ["accuracy"]

)



# training

results = model.fit_generator(get_train_data(), steps_per_epoch=steps_per_epoch, epochs=epochs)
# preparing testing data

def get_model_result(model):

    prepared_test_data = get_model_input_data(test_data[:,0])

    y_true = np.array([1. if label == 'positive' else 0. for label in test_data[:,1]])

    y_pred = np.argmax(model.predict(prepared_test_data), axis=1)

    return (y_true==y_pred).sum()/y_true.shape[0]



print("Test result - model: " + str(get_model_result(model)))
# RNN model

inputs = Input(shape=(sequence_length,))

x = Embedding(vocabulary_size, output_embeddings)(inputs)

x = GRU(units=40, return_sequences=True)(x)

x = Flatten()(x)

x = Dense(units=110, activation="relu")(x)

x = Dense(dense_layer_2_size)(x)

outputs = Activation("softmax")(x)



model_rnn = Model(inputs=inputs, outputs=outputs)

model_rnn.summary()



# training definition

model_rnn.compile(

    loss = tf.keras.losses.SparseCategoricalCrossentropy(), #"categorical_crossentropy"

    optimizer = tf.keras.optimizers.Adam(), #"adam"

    metrics = ["accuracy"]

)



# training

results = model_rnn.fit_generator(get_train_data(), steps_per_epoch=steps_per_epoch, epochs=epochs)
# results for GRU

print("Test result - RNN model: " + str(get_model_result(model_rnn)))