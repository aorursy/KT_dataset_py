#@title Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.
!pip install tensorflow_datasets
from __future__ import absolute_import, division, print_function, unicode_literals



import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf



tf.enable_eager_execution()



import tensorflow_hub as hub

import tensorflow_datasets as tfds



from tensorflow.keras import layers





tf.__version__
import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)
splits = tfds.Split.ALL.subsplit(weighted=(70,30))



(training_set, validation_set), dataset_info = tfds.load('tf_flowers', with_info=True, as_supervised=True, split=splits)
dataset_info
num_classes = dataset_info.features['label'].num_classes

num_training_examples = 0

num_validation_examples = 0



for _ in training_set:

    num_training_examples +=1



for _ in validation_set:

    num_validation_examples +=1



print('Total Number of Classes: {}'.format(num_classes))

print('Total Number of Training Images: {}'.format(num_training_examples))

print('Total Number of Validation Images: {} \n'.format(num_validation_examples))
for i, example in enumerate(training_set.take(5)):

  print('Image {} shape: {} label: {}'.format(i+1, example[0].shape, example[1]))
IMAGE_RES = 224



def format_image(image, label):

    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0



    return image, label



BATCH_SIZE = 32



train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)



validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)
URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
feature_extractor.trainable = False
model = tf.keras.Sequential([

    feature_extractor,

    layers.Dense(num_classes, activation='softmax')

])



model.summary()
model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



EPOCHS = 10



history = model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)
history.history.keys()
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(EPOCHS)



plt.figure(figsize=(16, 10))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
class_names = np.array(dataset_info.features['label'].names)



print(class_names)
image_batch, label_batch = next(iter(train_batches))



image_batch, label_batch = image_batch.numpy(), label_batch.numpy()





predicted_batch = model.predict(image_batch)

predicted_batch = tf.squeeze(predicted_batch).numpy()



predicted_ids = np.argmax(predicted_batch, axis=-1)

predicted_class_names = class_names[predicted_ids]



print(predicted_class_names)
print(label_batch)

print(predicted_ids)
plt.figure(figsize=(10,9))

for n in range(30):

  plt.subplot(6,5,n+1)

  plt.subplots_adjust(hspace = 0.3)

  plt.imshow(image_batch[n])

  color = "blue" if predicted_ids[n] == label_batch[n] else "red"

  plt.title(predicted_class_names[n].title(), color=color)

  plt.axis('off')

_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
!ls