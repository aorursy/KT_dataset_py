import os, sys, math

import numpy as np

from matplotlib import pyplot as plt

import tensorflow as tf

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware

try:

  tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection

except ValueError:

  tpu = None

#If TPU not found try with GPUs

  gpus = tf.config.experimental.list_logical_devices("GPU")

    

# Select appropriate distribution strategy for hardware

if tpu:

  tf.config.experimental_connect_to_cluster(tpu)

  tf.tpu.experimental.initialize_tpu_system(tpu)

  strategy = tf.distribute.experimental.TPUStrategy(tpu)

  print('Running on TPU ', tpu.master())  

elif len(gpus) > 0:

  strategy = tf.distribute.MirroredStrategy(gpus) # this works for 1 to multiple GPUs

  print('Running on ', len(gpus), ' GPU(s) ')

else:

  strategy = tf.distribute.get_strategy()

  print('Running on CPU')



# How many accelerators do we have ?

print("Number of accelerators: ", strategy.num_replicas_in_sync)
GCS_PATTERN = 'gs://flowers-public/tfrecords-jpeg-192x192-2/*.tfrec'

IMAGE_SIZE = [192, 192]



if tpu:

  BATCH_SIZE = 16*strategy.num_replicas_in_sync  # A TPU has 8 cores so this will be 128

else:

  BATCH_SIZE = 32  # On GPU, a higher batch size does not help and sometimes does not fit on the GPU (OOM)



VALIDATION_SPLIT = 0.20

CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'] # do not change, maps to the labels in the data (folder names)



# splitting data files between training and validation

filenames = tf.io.gfile.glob(GCS_PATTERN)

split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]

validation_filenames = filenames[:split]

print("Pattern matches {} data files. Splitting dataset into {} training files and {} validation files".format(len(filenames), len(training_filenames), len(validation_filenames)))

validation_steps = int(3670 // len(filenames) * len(validation_filenames)) // BATCH_SIZE

steps_per_epoch = int(3670 // len(filenames) * len(training_filenames)) // BATCH_SIZE

print("With a batch size of {}, there will be {} batches per training epoch and {} batch(es) per validation run.".format(BATCH_SIZE, steps_per_epoch, validation_steps))
def dataset_to_numpy_util(dataset, N):

  dataset = dataset.batch(N)

  

  # In eager mode, iterate in the Datset directly.

  for images, labels in dataset:

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    break;



  return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

  label = np.argmax(label, axis=-1)  # one-hot to class number

  correct_label = np.argmax(correct_label, axis=-1) # one-hot to class number

  correct = (label == correct_label)

  return "{} [{}{}{}]".format(CLASSES[label], str(correct), ', shoud be ' if not correct else '',

                              CLASSES[correct_label] if not correct else ''), correct



def display_one_flower(image, title, subplot, red=False):

    plt.subplot(subplot)

    plt.axis('off')

    plt.imshow(image)

    plt.title(title, fontsize=16, color='red' if red else 'black')

    return subplot+1

  

def display_chunk_images_from_dataset(dataset):

  subplot=331

  plt.figure(figsize=(15,15))

  images, labels = dataset_to_numpy_util(dataset, 9)

  for i, image in enumerate(images):

    title = CLASSES[np.argmax(labels[i], axis=-1)]

    subplot = display_one_flower(image, title, subplot)

    if i >= 8:

      break;

              

  plt.tight_layout()

  plt.subplots_adjust(wspace=0.1, hspace=0.1)

  plt.show()

  

def display_chunk_images_with_predictions(images, predictions, labels):

  subplot=331

  plt.figure(figsize=(15,15))

  for i, image in enumerate(images):

    title, correct = title_from_label_and_target(predictions[i], labels[i])

    subplot = display_one_flower(image, title, subplot, not correct)

    if i >= 8:

      break;

              

  plt.tight_layout()

  plt.subplots_adjust(wspace=0.1, hspace=0.1)

  plt.show()

  

def display_training_curves(training, validation, title, subplot):

  if subplot%10==1: # set up the subplots on the first call

    plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

    plt.tight_layout()

  ax = plt.subplot(subplot)

  ax.set_facecolor('#F8F8F8')

  ax.plot(training)

  ax.plot(validation)

  ax.set_title('model '+ title)

  ax.set_ylabel(title)

  ax.set_xlabel('epoch')

  ax.legend(['train', 'valid.'])
def read_tfrecord(example):

    features = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar

        "one_hot_class": tf.io.VarLenFeature(tf.float32),

    }

    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size will be needed for TPU

    one_hot_class = tf.sparse.to_dense(example['one_hot_class'])

    one_hot_class = tf.reshape(one_hot_class, [5])

    return image, one_hot_class



def load_dataset(filenames):

  # read from TFRecords. For optimal performance, read from multiple

  # TFRecord files at once and set the option experimental_deterministic = False

  # to allow order-altering optimizations.



  option_no_order = tf.data.Options()

  option_no_order.experimental_deterministic = False



  dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

  dataset = dataset.with_options(option_no_order)

  dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

  return dataset
display_chunk_images_from_dataset(load_dataset(training_filenames))
def data_augment(image, one_hot_class):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_saturation(image, 0, 2)

    return image, one_hot_class
def get_batched_dataset(filenames, train=False):

  dataset = load_dataset(filenames)

  dataset = dataset.cache() # This dataset fits in RAM

  if train:

    # Best practices for Keras:

    # Training dataset: repeat then batch

    # Evaluation dataset: do not repeat

    dataset = dataset.repeat()

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.shuffle(2000)

  dataset = dataset.batch(BATCH_SIZE)

  dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

  # should shuffle too but this dataset was well shuffled on disk already

  return dataset

  # source: Dataset performance guide: https://www.tensorflow.org/guide/performance/datasets



# instantiate the datasets

training_dataset = get_batched_dataset(training_filenames, train=True)

validation_dataset = get_batched_dataset(validation_filenames, train=False)



some_flowers, some_labels = dataset_to_numpy_util(load_dataset(validation_filenames), 160)
with strategy.scope(): # this line is all that is needed to run on TPU (or multi-GPU, ...)



  bnmomemtum=0.9

  def fire(x, squeeze, expand):

    y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)

    y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)

    y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, activation='relu', padding='same')(y)

    y1 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y1)

    y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)

    y3 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y3)

    return tf.keras.layers.concatenate([y1, y3])



  def fire_module(squeeze, expand):

    return lambda x: fire(x, squeeze, expand)



  x = tf.keras.layers.Input(shape=[*IMAGE_SIZE, 3]) # input is 192x192 pixels RGB



  y = tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', use_bias=True, activation='relu')(x)

  y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)

  y = fire_module(24, 48)(y)

  y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

  y = fire_module(48, 96)(y)

  y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

  y = fire_module(64, 128)(y)

  y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

  y = fire_module(48, 96)(y)

  y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

  y = fire_module(24, 48)(y)

  y = tf.keras.layers.GlobalAveragePooling2D()(y)

  y = tf.keras.layers.Dense(5, activation='softmax')(y)



  model = tf.keras.Model(x, y)



  model.compile(

    optimizer='adam',

    loss= 'categorical_crossentropy',

    metrics=['accuracy'])



  model.summary()
tf.keras.utils.plot_model(

    model, to_file='model.png', show_shapes=False, show_layer_names=True,

    rankdir='TB', expand_nested=False, dpi=96

)
EPOCHS = 35



history = model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,

                    validation_data=validation_dataset)
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 211)

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 212)
# randomize the input so that you can execute multiple times to change results

permutation = np.random.permutation(160)

some_flowers, some_labels = (some_flowers[permutation], some_labels[permutation])



predictions = model.predict(some_flowers, batch_size=16)

evaluations = model.evaluate(some_flowers, some_labels, batch_size=16)

  

print('[val_loss, val_acc]', evaluations)
display_chunk_images_with_predictions(some_flowers, predictions, some_labels)