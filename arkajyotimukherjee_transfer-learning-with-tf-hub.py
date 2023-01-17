%%capture

from __future__ import absolute_import, division, print_function, unicode_literals



!pip install -q tensorflow-gpu==2.0.0-beta1

!pip install -q tensorflow_hub
import matplotlib.pylab as plt

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
IMAGE_SHAPE = (224, 224)



classifier = tf.keras.Sequential([

    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))

])
import numpy as np

import PIL.Image as Image



grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')

grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)

grace_hopper
grace_hopper = np.array(grace_hopper)/255

grace_hopper.shape
results = classifier.predict(grace_hopper[np.newaxis, ...])

results
predicted_class = np.argmax(results[0], axis=-1)

predicted_class
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())

imagenet_labels
plt.imshow(grace_hopper)

plt.axis('off')

predicted_class_name = imagenet_labels[predicted_class]

_ = plt.title('Predition: '+ predicted_class_name, color='cyan')
data_root = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',

   untar=True)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)
for image_batch, label_batch in image_data:

  print("Image batch shape: ", image_batch.shape)

  print("Label batch shape: ", label_batch.shape)

  break
results_batch = classifier.predict(image_batch)

results_batch.shape
predicted_class_names = imagenet_labels[np.argmax(results_batch, axis=-1)]

predicted_class_names
plt.figure(figsize = (10,9))

plt.subplots_adjust(hspace=0.5)



for n in range(30):

  plt.subplot(6, 5, n+1)

  plt.imshow(image_batch[n])

  plt.title(predicted_class_names[n], color='cyan')

  plt.axis('off')

_ = plt.suptitle("Model predictions (green: correct, red: incorrect)", color='cyan')
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=IMAGE_SHAPE+(3,))
feature_batch = feature_extractor_layer(image_batch)

print(feature_batch.shape)
feature_extractor_layer.trainable = False
model = tf.keras.Sequential([

    feature_extractor_layer,

    layers.Dense(image_data.num_classes, activation='softmax')

])



model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(),

             loss = 'categorical_crossentropy',

             metrics = ['accuracy'])
class ConnectBatchStats(tf.keras.callbacks.Callback):

  def __init__(self):

    self.batch_losses = []

    self.batch_acc = []

    

  def on_train_batch_end(self, batch, logs=False):

    self.batch_losses.append(logs['loss'])

    self.batch_acc.append(logs['accuracy'])

    self.model.reset_metrics()
steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)



batch_stats_callback = ConnectBatchStats()



history = model.fit(image_data,

                   epochs = 15,

                   steps_per_epoch = steps_per_epoch,

                   callbacks = [batch_stats_callback])
plt.figure()

plt.xlabel('Training Steps')

plt.ylabel('Loss')

plt.ylim([0, 1])

plt.plot(batch_stats_callback.batch_losses, color='yellow')
plt.figure()

plt.xlabel('Training Steps')

plt.ylabel('Accuracy')

plt.ylim([0, 1])

plt.plot(batch_stats_callback.batch_acc, color='yellow')
class_names = sorted(image_data.class_indices.items(), key = lambda pair : pair[1])

class_names = np.array([key.title() for key, value in class_names])

class_names
predicted_batch = model.predict(image_batch)

predicted_id = np.argmax(predicted_batch, axis=-1)

predicted_label_batch = class_names[predicted_id]
label_id = np.argmax(label_batch, axis=-1)
plt.figure(figsize=(10,9))

plt.subplots_adjust(hspace=0.5)



for n in range(30):

  plt.subplot(6, 5, n+1)

  plt.imshow(image_batch[n])

  color = 'green' if predicted_id[n] == label_id[n] else 'red'

  plt.title(predicted_label_batch[n], color = color)

  plt.axis('off')



_=plt.suptitle('Model predictions (green: correct, red: incorrect)', color='cyan')
import time

t = time.time()



export_path = "/tmp/saved_models/{}".format(int(t))

tf.keras.experimental.export_saved_model(model, export_path)



export_path
reloaded = tf.keras.experimental.load_from_saved_model(export_path, custom_objects={'KerasLayer':hub.KerasLayer})



result_batch = model.predict(image_batch)

reloaded_result_batch = reloaded.predict(image_batch)



abs(reloaded_result_batch - result_batch).max()