import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

import keras

from keras import backend as K

from keras import models

from keras import layers

from keras.layers import Layer
def make_data(batch_size=10000, input_size=20, attention_index=5):

    '''

    batch_size : Total batch size

    input_size : Total input size, values are random number except at the attention_index

    attention_index : value is 0 or 1 at the attention index

    '''

    

    train_x = np.random.randn(batch_size, input_size)

    train_y = np.random.randint(2, size=(batch_size, 1))

    train_x[:, attention_index] = train_y[:, 0]

    return train_x, train_y
class AttentionLayer(Layer):

    def __init__(self, output_dim, **kwargs):

        self.output_dim = output_dim

        super(AttentionLayer, self).__init__(**kwargs)

        

    def build(self, input_shape):

        self.att_vec = self.add_weight(

            name = 'attention_vector', 

            shape = (input_shape[1], ),

            initializer = 'normal',

            trainable = True

        )

        super(AttentionLayer, self).build(input_shape)



    def call(self, x):

        y = tf.math.multiply(self.att_vec, x)

        y = keras.activations.softmax(y, axis=-1)

        return y



    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.output_dim)



# Make Training Set

train_x, train_y = make_data()

input_dims = train_x.shape[1]



# Make Net

input_layer = layers.Input(shape=(input_dims,))

attention_vector = AttentionLayer(input_dims)(input_layer)

y = layers.dot([attention_vector, input_layer], axes=1)

y = layers.Dense(1, activation='sigmoid', use_bias=False)(y)



# Train 

model = models.Model(input_layer, y)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(train_x, train_y, epochs=20, batch_size=32, validation_split=0.2, verbose=1)



# Evaluation

test_x, test_y = make_data()

true = test_y

pred = model.predict(test_x)

pred = np.around(pred)

score = (1-np.sum(np.abs(pred-true))/len(true))*100

print('score: %f %%'%(score))
# Calculate Attention vector

attention_layer = model.layers[1]

func = K.function([model.input], [attention_layer.output])

output = func([test_x])[0]

attention_vector = np.mean(output, axis=0)



# Show Attention vector

plt.title('Attention Vector')

plt.bar(np.arange(len(attention_vector))+0.3, attention_vector, width=0.6)

plt.xlabel('index')

plt.ylabel('Attention ratio')

plt.show()