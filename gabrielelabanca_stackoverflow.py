import numpy as np

import tensorflow as tf

from tensorflow import keras



import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/60k-stack-overflow-questions-with-quality-rate/data.csv')

df = df.sample(frac=1).reset_index(drop=True)

df.loc[:, 'size_of_body'] = df.apply(lambda x: len(x.Body.replace('<p>', '').split()), axis=1)

df.loc[:, 'label'] = df.apply(lambda x: {

    'HQ': 0,

    'LQ_EDIT': 1,

    'LQ_CLOSE': 2

}[x.Y], axis=1)

df.head()
text_of_question = df.apply(lambda x: ' '.join([x.Title, x.Body]).replace('>', '> ').replace('<', ' <').replace('=', ' = ').lower(), axis=1)

text_of_question = np.array(text_of_question)

label_of_question = df.apply(lambda x: {

    'HQ': ([1.0, 0.0, 0.0]),

    'LQ_EDIT': ([0.0, 1.0, 0.0]),

    'LQ_CLOSE': ([0.0, 0.0, 1.0])

}[x.Y], axis=1)

label_of_question = list(label_of_question.values)
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from tensorflow.keras.layers.experimental.preprocessing import Normalization



max_length_dictionary = 1000

def preprocess_data(data, max_length_dictionary=max_length_dictionary, max_length_sentence=100):

    processed_data = data.reshape(-1,1)

    vectorizer = TextVectorization(output_mode="int", 

                                   max_tokens=max_length_dictionary, 

                                   output_sequence_length=max_length_sentence)

    vectorizer.adapt(processed_data)

    integer_data = vectorizer(processed_data)

    #normalizer = Normalization(axis=-1)

    #normalizer.adapt(integer_data)

    #normalized_data = normalizer(integer_data)

    normalized_data = integer_data

    return normalized_data
n_samples = {'train': 30000, 'dev': 1000, 'test': 1000}



X = preprocess_data(text_of_question[:int(np.sum(list(n_samples.values())))])

X_train = X[:n_samples['train']]

X_dev= X[n_samples['train']:n_samples['train']+n_samples['dev']]

X_test = X[n_samples['train']+n_samples['dev']:n_samples['train']+n_samples['dev']+n_samples['test']]



Y = tf.convert_to_tensor(label_of_question[:int(np.sum(list(n_samples.values())))])

Y_train = tf.convert_to_tensor(label_of_question[:n_samples['train']])

Y_dev = tf.convert_to_tensor(label_of_question[n_samples['train']:n_samples['train']+n_samples['dev']])

Y_test = tf.convert_to_tensor(label_of_question[n_samples['train']+n_samples['dev']:n_samples['train']+n_samples['dev']+n_samples['test']])



#import numpy.testing as npt

#for X in [X]:#X_train, X_dev, X_test]:

    #np.testing.assert_almost_equal(np.var(X), 1.0, decimal=5)

    #np.testing.assert_almost_equal(np.mean(X), 0.0, decimal=5)

print(X.shape, Y.shape)
from tensorflow.keras import layers



def create_model(n_tokens, max_sentence_length):

    inputs = keras.Input(shape=(max_sentence_length))



    x = layers.Embedding(n_tokens, 700, input_length=max_sentence_length)(inputs)

    

    #x = layers.Bidirectional(layers.SimpleRNN(100))(x)

    #x = layers.Conv2D(filters=1, kernel_size=(10,10), activation='relu')#, input_shape=(None, None, 20))

    #x = layers.MaxPooling2D(pool_size=(2,2))(x)

    

    x = layers.GlobalAveragePooling1D()(x)



    x = layers.Dense(700, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(600, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(500, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(400, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(300, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(200, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(100, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Dense(50, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Dense(25, activation="relu")(x)# kernel_regularizer='l2')(x)

    x = layers.Dense(20, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dense(15, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dense(10, activation="relu")(x)#, kernel_regularizer='l2')(x)

    x = layers.Dense(5, activation="relu")(x)#, kernel_regularizer='l2')(x)

    

    num_classes = 3

    outputs = layers.Dense(num_classes, activation="softmax")(x)



    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    model = create_model(n_tokens=1000, max_sentence_length=X.shape[1])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()


print(X_train.shape, Y_train.shape)

history = model.fit(X_train, Y_train,

          batch_size=512, epochs=100)
def list_errors(x_data, y_data):

    return [(np.argmax(x) != np.argmax(y)) for x,y in  zip(model(x_data), y_data)]



def n_errors(x_data, y_data):

    return np.sum(list_errors(x_data, y_data))



err1 = n_errors(X_train[:1000], Y_train[:1000])/n_samples['train']

err2 = n_errors(X_dev, Y_dev)/n_samples['dev']

err3 = n_errors(X_test, Y_test)/n_samples['test']

print(err2, err3)
le = list_errors(X_train[:1000], Y_train[:1000])

ie = 0
if err1 > 0:

    print("Sample error")

    ie = le.index(True, ie+1)

    df.iloc[ie].Title, df.iloc[ie].Body, df.iloc[ie].Y 
import matplotlib.pyplot as plt

plt.figure()

plt.plot(history.history['loss'], label='loss')

plt.plot(history.history['accuracy'], label='accuracy')

plt.legend()

plt.show()