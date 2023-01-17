import tensorflow as tf

import tensorflow_hub as hub

import os



model = hub.KerasLayer('https://tfhub.dev/google/bit/m-r152x4/1')

if not os.path.exists('bit'):

    os.makedirs('bit')

tf.saved_model.save(model,'bit/m-r152x4')