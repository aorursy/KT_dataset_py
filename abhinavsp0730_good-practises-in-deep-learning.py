! pip install -U tensorboard_plugin_profile
from datetime import datetime
from packaging import version

import os
import tensorflow as tf
import tensorflow_datasets as tfds
(train, test), dataset_info = tfds.load(
    'mnist',
    split =['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True )
def rescale(image, label):
    return tf.cast(image, tf.float32) / 255., label
# rescaling the image
train  = train.map(rescale)
test = test.map(rescale)
# Batching the datasets
train = train.batch(128)
test = test.batch(128)
# Creating not so cool model :)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(256,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

model.fit(train,
          epochs=2,
          validation_data=test,
          callbacks = [tboard_callback])
#Load the TensorBoard notebook extension.
%load_ext tensorboard
# Launch TensorBoard and navigate to the Profile tab to view performance profile
%tensorboard --logdir=logs
#again loading the datasets.
(train, test), dataset_info = tfds.load(
    'mnist',
    split =['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True )

# Creating the optimized input pipeline.
def rescale(image, label):
    return tf.cast(image, tf.float32) / 255., label
# rescaling the image
train  = train.map(rescale)
test = test.map(rescale)


train = train.batch(128)
# applying cache in training set
train = train.cache()
# applying prefetching in training sets
train = train.prefetch(tf.data.experimental.AUTOTUNE)


test = test.batch(128)
# applying cache in test set
test = test.cache()
# applying prefetching in test set
test = test.prefetch(tf.data.experimental.AUTOTUNE)
# again making and training the not so cool model but this time we'are using the optimised training

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(256,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(train,
          epochs=2,
          validation_data=test,
          callbacks = [tboard_callback])
# Launching the tensorboard.
%tensorboard --logdir=logs
# downloading tf_flowers it contains 5 classes of 5 differebt flowers species.

train, train_info = tfds.load('tf_flowers', split='train[:80%]', 
                              as_supervised=True, 
                              with_info=True)
val, val_info = tfds.load('tf_flowers', 
                          split='train[80%:]', 
                          as_supervised=True, 
                          with_info=True)
# preprocessing and making input pipeline
def resize(img, lbl):
  img_size = 224
  return (tf.image.resize(img, [img_size, img_size])/255.) , lbl

train = train.map(resize)
val = val.map(resize)

train = train.batch(32, drop_remainder=True)
val = val.batch(32, drop_remainder=True)
# function to create resnet model with random weights and imagenet weights
def resnet(imagenet_weights=False):
    num_classes = 5
    if imagenet_weights is False:
        return tf.keras.applications.ResNet50(include_top=True, 
                                        input_shape=(224, 224, 3), 
                                        weights=None, 
                                        classes=num_classes)
    if imagenet_weights is True:
        resnet = tf.keras.applications.ResNet50(include_top=False, 
                                        input_shape=(224, 224, 3), 
                                        weights='imagenet', 
                                        )
        resnet.trainable = True
        return  tf.keras.Sequential([resnet, 
                                     tf.keras.layers.GlobalAvgPool2D(), 
                                     tf.keras.layers.Dense(5, activation='softmax')])
# training the model in gpu if availiable
def try_gpu(i=0): 
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')
device_name = try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
  model = resnet(imagenet_weights=False)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback = callback = tf.keras.callbacks.EarlyStopping(patience=8)
history = model.fit(train, 
                    epochs=100, 
                    validation_data=val,
                    callbacks = [callback])
# Saving validation accuracy ib the variable val_acc_1
val_acc_1 = history.history['val_accuracy']

with strategy.scope():
  model = resnet(imagenet_weights=True)

# compiling and training.
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback = callback = tf.keras.callbacks.EarlyStopping(patience=8)
history = model.fit(train, 
                    epochs=100, 
                    validation_data=val,
                    callbacks = [callback])
# Saving validation accuracy of this model in the variable val_acc_2
val_acc_2 = history.history['val_accuracy']
print("Validation accuracy of first approach is {} VS Validation accuracy of secon approach is {}".format(max(val_acc_1), max(val_acc_2)))
# loading the tf_flowers dataset
train, train_info = tfds.load('tf_flowers', split='train[:80%]', 
                              as_supervised=True, 
                              with_info=True)
val, val_info = tfds.load('tf_flowers', 
                          split='train[80%:]', 
                          as_supervised=True, 
                          with_info=True)
#function to augment the images
def augment(image,label):
  image = tf.image.resize_with_crop_or_pad(image, 34, 34) # Add 6 pixels of padding
  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
  image = tf.image.random_flip_up_down(image)  # Randomily flips the image up and down 


  return image,label
train = train.map(augment).map(resize).batch(32, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
val = val.map(resize).batch(32, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
for x,y in train.take(1):
    print(x.shape)
def resnet_transfer_learning():
    # including_top=False means we're the last layer of resnet will not include fully connected dense layer
    resnet = tf.keras.applications.ResNet50(include_top=False, 
                                        input_shape=(224, 224, 3), 
                                        weights='imagenet', 
                                        )
    # freezing the layers of resnet.
    resnet.trainable = True
    # adding the dense layer to the resnet model
    return  tf.keras.Sequential([resnet, 
                                     tf.keras.layers.GlobalAvgPool2D(), 
                                     tf.keras.layers.Dense(5, activation='softmax')])
with strategy.scope():
  model = resnet_transfer_learning()

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback = callback = tf.keras.callbacks.EarlyStopping(patience=8)
history = model.fit(train, 
                    epochs=100, 
                    validation_data=val,
                    callbacks = [callback])
# Saving validation accuracy of this model in the variable val_acc_2
val_acc_3 = history.history['val_accuracy']
print("Validation accuracy of first approach is {} VS Validation accuracy of second approach is {} VS Validation accuracy of second approach is {}".format(max(val_acc_1), max(val_acc_2), max(val_acc_3)))