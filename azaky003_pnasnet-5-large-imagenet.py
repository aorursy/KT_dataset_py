!pip install -q -U tf-hub-nightly

!pip install -q tfds-nightly



import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import models, layers, optimizers, utils



IMAGE_SIZE = 331



model = models.Sequential([

    hub.KerasLayer('https://tfhub.dev/google/imagenet/pnasnet_large/classification/4', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

])



model.summary()



tf.saved_model.save(model, '.')