# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, SimpleRNN, RepeatVector

from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback



from termcolor import colored



print('Tested with tensorflow version 2.0.1')

print('Using tensorflow version:', tf.__version__)
all_chars = '0123456789+'
num_features = len(all_chars)



char_to_index = dict((c, i) for i, c in enumerate(all_chars))

index_to_char = dict((i, c) for i, c in enumerate(all_chars))



print('Number of features:', num_features)
def generate_data():

    first_num = np.random.randint(low=0,high=100)

    second_num = np.random.randint(low=0,high=100)

    example = str(first_num) + '+' + str(second_num)

    label = str(first_num+second_num)

    return example, label



generate_data()
hidden_units = 128

max_time_steps = 5



model = Sequential([

    SimpleRNN(hidden_units, input_shape=(None, num_features)),

    RepeatVector(max_time_steps),

    SimpleRNN(hidden_units, return_sequences=True),

    TimeDistributed(Dense(num_features, activation='softmax'))

])



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
def vectorize_example(example, label):

    

    x = np.zeros((max_time_steps, num_features))

    y = np.zeros((max_time_steps, num_features))

    

    diff_x = max_time_steps - len(example)

    diff_y = max_time_steps - len(label)

    

    for i, c in enumerate(example):

        x[diff_x+i, char_to_index[c]] = 1

    for i in range(diff_x):

        x[i, char_to_index['0']] = 1

    for i, c in enumerate(label):

        y[diff_y+i, char_to_index[c]] = 1

    for i in range(diff_y):

        y[i, char_to_index['0']] = 1

        

    return x, y



e, l = generate_data()

print('Text Example and Label:', e, l)

x, y = vectorize_example(e, l)

print('Vectorized Example and Label Shapes:', x.shape, y.shape)
def devectorize_example(example):

    result = [index_to_char[np.argmax(vec)] for i, vec in enumerate(example)]

    return ''.join(result)



devectorize_example(x)
def create_dataset(num_examples=2000):



    x_train = np.zeros((num_examples, max_time_steps, num_features))

    y_train = np.zeros((num_examples, max_time_steps, num_features))



    for i in range(num_examples):

        e, l = generate_data()

        x, y = vectorize_example(e, l)

        x_train[i] = x

        y_train[i] = y

    

    return x_train, y_train



x_train, y_train = create_dataset()

print(x_train.shape, y_train.shape)
devectorize_example(x_train[0])
devectorize_example(y_train[0])
simple_logger = LambdaCallback(

    on_epoch_end=lambda e, l: print('{:.2f}'.format(l['val_accuracy']), end=' _ ')

)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)



model.fit(x_train, y_train, epochs=500, validation_split=0.2, verbose=False,

         callbacks=[simple_logger, early_stopping])
x_test, y_test = create_dataset(num_examples=20)

preds = model.predict(x_test)

full_seq_acc = 0



for i, pred in enumerate(preds):

    pred_str = devectorize_example(pred)

    y_test_str = devectorize_example(y_test[i])

    x_test_str = devectorize_example(x_test[i])

    col = 'green' if pred_str == y_test_str else 'red'

    full_seq_acc += 1/len(preds) * int(pred_str == y_test_str)

    outstring = 'Input: {}, Out: {}, Pred: {}'.format(x_test_str, y_test_str, pred_str)

    print(colored(outstring, col))

print('\nFull sequence accuracy: {:.3f} %'.format(100 * full_seq_acc))