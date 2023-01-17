#IPython display
import IPython.display as display
from PIL import Image

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__) 

# Helper libraries
import os
import pathlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# Set working directory
root_dir = Path.cwd()
PATH = str(root_dir / 'generated_dataset')
data_dir = pathlib.Path(PATH)
# Check image count
image_count = len(list(data_dir.glob('*/*.jpg')))
image_count
# Extract target labels
class_names = np.array([item.name for item in data_dir.glob('*')])
list_labels = list(class_names)
class_names
# View sample image
normal = list(data_dir.glob('normal/*'))
for image_path in normal[:1]:
  print(image_path)
  check_img = Image.open(str(image_path))
  display.display(check_img)
  w,h = check_img.size
  print("original dimensions:{}x{}".format(w,h))

# Use tf.data.Dataset to create a TF dataset in the file paths
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
type(list_ds)
# Break the path to components
def get_label(file_path):
  parts = tf.strings.split(file_path, sep = os.path.sep )
  #observe that the second last component is the name of the class
  return parts[-2] == class_names
# Decoding image to tensor and resize
IMG_HEIGHT = 224
IMG_WIDTH  = 224
BATCH_SIZE = 16

def decode_img(img):
  #convert the image to a 3D tensor
  img = tf.image.decode_jpeg(img, channels = 3)
  #Use `convert_image_dtype` to convert to floats in the [0, 1] range
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

# Finally club the desired operations
def process_path(file_path):
  label = get_label(file_path)
  #load the data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label
# TF map the above function to apply transformations to all samples in the dataset
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE) 
for image, label in labeled_ds.take(1):
    print(image.shape)
for image, label in labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label:", label.numpy())
    print(type(label))
  
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat(count=1)

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds
train_ds = prepare_for_training(labeled_ds)
# Plot a batch
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(16):
      ax = plt.subplot(4,4,n+1)
      plt.imshow(image_batch[n])
      title = plt.title(class_names[label_batch[n]==1][0].title())
      plt.setp(title, color = 'y')  
      plt.axis('off')

image_batch, label_batch = next(iter(train_ds))
show_batch(image_batch.numpy(), label_batch.numpy())
# check the number of elements in the dataset
# we have grouped the samples into batches so `cardinality` will consider the number of batches
num_elements = tf.data.experimental.cardinality(train_ds).numpy()
num_elements
# perform train/validation split using take/split
val_size = int(.1 * num_elements)
val_ds = train_ds.take(val_size)
train_ds = train_ds.skip(val_size)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
# create base (no-brainer) model
cd_model = Sequential([
    Conv2D(8, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Flatten(),
    Dense(2, activation='sigmoid')
])
# compile the model
cd_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# check the model summary
cd_model.summary()
# finally fit the model
epochs = 3
history = cd_model.fit_generator(
    train_ds,
    epochs=epochs,
    validation_data = val_ds,
    verbose = 1,
    initial_epoch = 0
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

acc_fig = plt.figure(figsize=(6, 6))
acc_ax = acc_fig.add_subplot()
acc_ax.tick_params(colors='y')
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy', color = 'yellow')
plt.show()

loss_fig = plt.figure(figsize= (6, 6))
loss_ax = loss_fig.add_subplot()
loss_ax.tick_params(color= 'y')
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', color = 'y')
plt.show()

