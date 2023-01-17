# TODO: Make all necessary imports.

!pip install -q tqdm==4.28.1

!pip install -q -U "tensorflow-gpu==2.0.0b1"

!pip install -q -U tensorflow_datasets

!pip install -q -U tensorflow_hub

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import time

import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)
# TODO: Load the dataset with TensorFlow Datasets.

dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)

dataset_info

# TODO: Create a training set, a validation set and a test set.

test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']
# TODO: Get the number of examples in each set from the dataset info.

num_training_examples = 0

num_validation_examples = 0

num_test_examples = 0



for example in training_set:

  num_training_examples += 1



for example in validation_set:

  num_validation_examples += 1



for example in test_set:

  num_test_examples += 1



print('Total Number of Training Images: {}'.format(num_training_examples))

print('Total Number of Validation Images: {}'.format(num_validation_examples))

print('Total Number of Test Images: {} \n'.format(num_test_examples))

# TODO: Get the number of classes in the dataset from the dataset info.

num_classes = dataset_info.features['label'].num_classes

print('Total Number of Classes: {}'.format(num_classes))
# TODO: Print the shape and corresponding label of 3 images in the training set.

# The images in the Flowers dataset are not all the same size.

for i, example in enumerate(training_set.take(3)):

  print('Image {} shape: {} label: {}'.format(i+1, example[0].shape, example[1]))
# TODO: Plot 1 image from the training set. Set the title 

# of the plot to the corresponding image label. 

for image, label in training_set.take(1):

  break

image = image.numpy()

plt.figure()

plt.imshow(image, cmap=plt.cm.binary)

plt.title('corresponding image label {}'.format(label))

plt.colorbar()

plt.grid(False)

plt.show()
import json
with open('../input/labels/label_map.json', 'r') as f:

    class_names = json.load(f)
# TODO: Plot 1 image from the training set. Set the title 

# of the plot to the corresponding class name.

plt.figure()

plt.imshow(image, cmap=plt.cm.binary)

plt.title('corresponding image label: {}'.format(class_names[str(label.numpy())]))

plt.colorbar()

plt.grid(False)

plt.show()
# TODO: Create a pipeline for each set.

IMAGE_RES = 224



def format_image(image, label):

  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0

  return image, label



BATCH_SIZE = 32



train_batches = training_set.cache().shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)



validation_batches = validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)



test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)
# TODO: Build and train your network.

from tensorflow.keras import layers

# Create a Feature Extractor

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL,

                                   input_shape=(IMAGE_RES, IMAGE_RES, 3))

# Freeze the Pre-Trained Model

feature_extractor.trainable = False

# Attach a classification head

model = tf.keras.Sequential([

  feature_extractor,

  layers.Dense(num_classes, activation='softmax')

])



model.summary()
print('Is there a GPU Available:', tf.test.is_gpu_available())
model.compile(

  optimizer='adam',

  loss='sparse_categorical_crossentropy',

  metrics=['accuracy'])



EPOCHS = 20



# Stop training when there is no improvement in the validation loss for 5 consecutive epochs

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)



history = model.fit(train_batches,

                    epochs=EPOCHS,

                    validation_data=validation_batches,

                    callbacks=[early_stopping])
# TODO: Plot the loss and accuracy values achieved during training for the training and validation set.

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



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
# TODO: Print the loss and accuracy values achieved on the entire test set.

results = model.evaluate(test_batches)

print('test loss, test acc:', results)

# TODO: Save your trained model as a Keras model.

# The name of our HDF5 will correspond to the current time stamp.

import time

t = time.time()



export_path_keras = "./{}.h5".format(int(t))

print(export_path_keras)



model.save(export_path_keras)
!ls
# TODO: Load the Keras model

reloaded = tf.keras.models.load_model(

  export_path_keras, 

  # `custom_objects` tells keras how to load a `hub.KerasLayer`

  custom_objects={'KerasLayer': hub.KerasLayer})



reloaded.summary()
image_batch, label_batch = next(iter(train_batches.take(1)))

image_batch = image_batch.numpy()



result_batch = model.predict(image_batch)

reloaded_result_batch = reloaded.predict(image_batch)

reloaded_result_batch.shape
(abs(result_batch - reloaded_result_batch)).max()
# TODO: Create the process_image function

def process_image(img):

    image = np.squeeze(img)

    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0

    return image
from PIL import Image



image_path = '../input/test-images/hard-leaved_pocket_orchid.jpg'

im = Image.open(image_path)

test_image = np.asarray(im)



processed_test_image = process_image(test_image)



fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)

ax1.imshow(test_image)

ax1.set_title('Original Image')

ax2.imshow(processed_test_image)

ax2.set_title('Processed Image')

plt.tight_layout()

plt.show()
# TODO: Create the predict function

def predict(image_path, model, top_k):

    im = Image.open(image_path)

    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)

    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))

    top_values, top_indices = tf.math.top_k(prediction, top_k)

    print("These are the top propabilities",top_values.numpy()[0])

    top_classes = [class_names[str(value)] for value in top_indices.cpu().numpy()[0]]

    print('Of these top classes', top_classes)

    return top_values.numpy()[0], top_classes
# TODO: Plot the input image along with the top 5 classes

import glob

files = glob.glob('../input/test-images/*.jpg')

for image_path in files:

    im = Image.open(image_path)

    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)

    probs, classes = predict(image_path, reloaded, 5)

    fig, (ax1, ax2) = plt.subplots(figsize=(12,4), ncols=2)

    ax1.imshow(processed_test_image)

    ax2 = plt.barh(classes[::-1], probs[::-1])

    plt.tight_layout()

    plt.show()