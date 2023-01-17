import tensorflow as tf

import numpy as np

from tensorflow.python.keras import layers

from tensorflow.python.keras import models

tf.enable_eager_execution()
# input batch shape = (1, 2, 2, 1) -> (batch_size, height, width, channels) - 2x2x1 image in batch of 1

x = tf.constant(np.array([[

    [[1], [2]], 

    [[3], [4]]

]]), tf.float32)
inputs = layers.Input(shape=(2,2,1))    # 256

outputs = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.Ones())(inputs)

model = models.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer=tf.train.AdamOptimizer())

model.summary()

print('For one channel input filtered once:')

print('Given input:\n {}'.format(np.squeeze(x) ))

print('The output is:\n {}'.format(np.squeeze(model.predict(x)) ))
inputs = layers.Input(shape=(2,2,1))    # 256

outputs = layers.Conv2DTranspose(2, (2, 2), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.Ones())(inputs)

model = models.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer=tf.train.AdamOptimizer())

model.summary()

print('For one channel input filtered twice:')

print('output 1:\n {}'.format(np.squeeze(model.predict(x)[0,:,:,0]) ))

print('output 2:\n {}'.format(np.squeeze(model.predict(x)[0,:,:,1]) ))

x1 = tf.constant(np.array([[

    [[1, 1], [2, 2]], 

    [[3, 3], [4, 4]]

]]), tf.float32)

print('input with 2 channels:\nchannel 1:\n{}'.format(np.squeeze(x1.numpy()[0,:,:,0]) ))

print('channel 2:\n{}'.format(np.squeeze(x1.numpy()[0,:,:,1]) ))
inputs = layers.Input(shape=(2,2,2))    # 256

outputs = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.Ones())(inputs)

model = models.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer=tf.train.AdamOptimizer())

model.summary()

print('For 2-channel input filtered once:')

print('output:\n{}'.format(np.squeeze(model.predict(x1)[0,:,:,0]) ))

inputs = layers.Input(shape=(2,2,2))    # 256

outputs = layers.Conv2DTranspose(2, (2, 2), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.Ones())(inputs)

model = models.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer=tf.train.AdamOptimizer())

model.summary()



print('For 2-channel input filtered twice:')

print('output channel 1:\n{}'.format(np.squeeze(model.predict(x1)[0,:,:,0]) ))

print('output channel 2:\n{}'.format(np.squeeze(model.predict(x1)[0,:,:,1]) ))
x2 = tf.constant(np.array([[

    [[1, 5], [2, 6]], 

    [[3, 7], [4, 8]]

]]), tf.float32)

y2 = np.array([

    [6, 6, 8, 8], [6, 6, 8, 8], 

    [10, 10, 12, 12], [10, 10, 12, 12]

])

print('Assumption 1: If input is\n channel 1:\n{}'.format(np.squeeze(x2.numpy()[0,:,:,0]) ))

print('channel 2:\n{}'.format(np.squeeze(x2.numpy()[0,:,:,1]) ))

print('The output would be:\n{}'.format(y2))
inputs = layers.Input(shape=(2,2,2))    # 256

outputs = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.Ones())(inputs)

model = models.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer=tf.train.AdamOptimizer())

model.summary()

print('output:\n{}'.format(np.squeeze(model.predict(x2)[0,:,:,0]) ))

print('Assumption 1 is correct.')
print('Assumption 2: When output 2 channels, the second one would the same.' )

inputs = layers.Input(shape=(2,2,2))    # 256

outputs = layers.Conv2DTranspose(2, (2, 2), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.Ones())(inputs)

model = models.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer=tf.train.AdamOptimizer())

model.summary()

print('output channel 1:\n{}'.format(np.squeeze(model.predict(x2)[0,:,:,0]) ))

print('channel 2:\n{}'.format(np.squeeze(model.predict(x2)[0,:,:,1])) )

print('Assumption 2 is correct.')
print('Question: What is the output for input with 4 channels?')

x4 = tf.constant(np.array([[

    [[-1, 2, 3, 4], [2, 3, 4, 5]], 

    [[3, 4, 5, 6], [4, 5, 6, 7]]

]]), tf.float32)

inputs = layers.Input(shape=(2,2,4))    # 256

outputs = layers.Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.Ones())(inputs)

model = models.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer=tf.train.AdamOptimizer())

model.summary()

print('output channel 1:\n{}'.format(np.squeeze(model.predict(x4)[0,:,:,0]) ))

print('channel 2:\n{}'.format(np.squeeze(model.predict(x4)[0,:,:,1])) )

print('channel 3:\n{}'.format(np.squeeze(model.predict(x4)[0,:,:,2])) )

print('channel 4:\n{}'.format(np.squeeze(model.predict(x4)[0,:,:,3])) )

print('Conclusion: all channels convoluted with filter and sumed up.')
print('When filter size is (3, 3)')

inputs = layers.Input(shape=(2,2,1))    # 256

outputs = layers.Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.Ones())(inputs)

model = models.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer=tf.train.AdamOptimizer())

model.summary()

print('output channel 1:\n{}'.format(np.squeeze(model.predict(x)[0,:,:,0]) ))

print('channel 2:\n{}'.format(np.squeeze(model.predict(x)[0,:,:,1])) )

print('channel 3:\n{}'.format(np.squeeze(model.predict(x)[0,:,:,2])) )

print('channel 4:\n{}'.format(np.squeeze(model.predict(x)[0,:,:,3])) )
from IPython.display import Image

from IPython.core.display import HTML 

print("The calculation demostrated in the following picture is correct!")

print("We should go to this website to upvote this answer:\nhttps://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers/20176#20176")

Image(url="https://i.stack.imgur.com/GlqLM.png")
