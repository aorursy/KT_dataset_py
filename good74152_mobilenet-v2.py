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
!pip install tf-nightly-gpu

!pip install "tensorflow_hub==0.4.0"

!pip install -U tensorflow_datasets
from __future__ import absolute_import, division, print_function, unicode_literals



import matplotlib.pylab as plt



import tensorflow as tf

tf.enable_eager_execution()



import tensorflow_hub as hub

import tensorflow_datasets as tfds



from tensorflow.keras import layers
import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)
CLASSIFIER_URL ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #由tensorflow hub載入

IMAGE_RES = 224 #mobilenet當初訓練時圖片輸入大小是224*224, 因此之後也要一樣



model = tf.keras.Sequential([

    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))

])
import numpy as np

import PIL.Image as Image



grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')

grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))

grace_hopper 
grace_hopper = np.array(grace_hopper)/255.0

grace_hopper.shape
result = model.predict(grace_hopper[np.newaxis, ...])

result.shape
predicted_class = np.argmax(result[0], axis=-1)

predicted_class
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())



plt.imshow(grace_hopper)

plt.axis('off')

predicted_class_name = imagenet_labels[predicted_class]

_ = plt.title("Prediction: " + predicted_class_name.title())
splits = tfds.Split.ALL.subsplit(weighted=(80, 20))



splits, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split = splits)



(train_examples, validation_examples) = splits



num_examples = info.splits['train'].num_examples

num_classes = info.features['label'].num_classes
for i, example_image in enumerate(train_examples.take(3)):

  print("Image {} shape: {}".format(i+1, example_image[0].shape))
def format_image(image, label):

  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0

  return  image, label



BATCH_SIZE = 32



train_batches      = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
image_batch, label_batch = next(iter(train_batches.take(1)))

image_batch = image_batch.numpy()

label_batch = label_batch.numpy()



result_batch = model.predict(image_batch)



predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]

predicted_class_names
plt.figure(figsize=(10,9))

for n in range(30):

  plt.subplot(6,5,n+1)

  plt.imshow(image_batch[n])

  plt.title(predicted_class_names[n])

  plt.axis('off')

_ = plt.suptitle("ImageNet predictions")
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

feature_extractor = hub.KerasLayer(URL,

                                   input_shape=(IMAGE_RES, IMAGE_RES,3))
feature_batch = feature_extractor(image_batch)

print(feature_batch.shape)
feature_extractor.trainable = False
model = tf.keras.Sequential([

  feature_extractor,

  layers.Dense(2, activation='softmax')

])



model.summary()
model.compile(

  optimizer='adam', 

  loss='sparse_categorical_crossentropy',

  metrics=['accuracy'])



EPOCHS = 6

history = model.fit(train_batches,

                    epochs=EPOCHS,

                    validation_data=validation_batches)
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(EPOCHS)



plt.figure(figsize=(8, 8))

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
class_names = np.array(info.features['label'].names)

class_names
predicted_batch = model.predict(image_batch)

predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch, axis=-1)

predicted_class_names = class_names[predicted_ids]

predicted_class_names
print("Labels: ", label_batch)

print("Predicted labels: ", predicted_ids)
plt.figure(figsize=(10,9))

for n in range(30):

  plt.subplot(6,5,n+1)

  plt.imshow(image_batch[n])

  color = "blue" if predicted_ids[n] == label_batch[n] else "red"

  plt.title(predicted_class_names[n].title(), color=color)

  plt.axis('off')

_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")