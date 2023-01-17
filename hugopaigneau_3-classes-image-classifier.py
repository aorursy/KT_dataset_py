import pathlib

import os

import numpy as np

import pandas as pb

import seaborn as sns

import tensorflow as tf

import keras

import IPython.display as display

from PIL import Image

import cv2

import matplotlib.pyplot as plt

%matplotlib inline
data_dir = pathlib.Path("../input/images_train")

image_count = len(list(data_dir.glob('*/*.jpg')))

image_count
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])

CLASS_NAMES
cat = list(data_dir.glob('cat/*'))

car = list(data_dir.glob('car/*'))

flower = list(data_dir.glob('flower/*'))



for image_path in cat[:1]:

    img = cv2.imread(str(image_path))
plt.imshow(img)
BATCH_TRAIN_SIZE = 64

IMG_HEIGHT = 224

IMG_WIDTH = 224

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_TRAIN_SIZE)

EPOCHS=12
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),

                                                     batch_size=BATCH_TRAIN_SIZE,

                                                     shuffle=True,

                                                     target_size=(224, 224),

                                                     class_mode="sparse",

                                                     classes = list(CLASS_NAMES))
def show_batch(image_batch, label_batch):

    plt.figure(figsize=(10,10))

    for n in range(25):

        ax = plt.subplot(5,5,n+1)

        plt.imshow(image_batch[n])

        plt.axis('off')
image_batch, label_batch = next(train_data_gen)

show_batch(image_batch, label_batch)
image_batch[2].shape
AUTOTUNE = tf.data.experimental.AUTOTUNE
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))



for f in list_ds.take(5):

    print(f.numpy())
def get_label(file_path):

    # convert the path to a list of path components

    parts = tf.strings.split(file_path, os.path.sep)

    # The second to last is the class-directory

    return parts[-2] == CLASS_NAMES



def decode_img(img):

    # convert the compressed string to a 3D uint8 tensor

    img = tf.image.decode_jpeg(img, channels=3)

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.

    img = tf.image.convert_image_dtype(img, tf.float32)

    # resize the image to the desired size.

    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])



def process_path(file_path):

    label = get_label(file_path)

    # load the raw data from the file as a string

    img = tf.io.read_file(file_path)

    img = decode_img(img)

    return img, label
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)



for image, label in labeled_ds.take(1):

    print("Image shape: ", image.numpy().shape)

    print("Label: ", label.numpy())
train_ds = labeled_ds.take(np.ceil(1596*0.7)) 

test_ds = labeled_ds.take(np.ceil(1596*0.7))
def prepare_for_training(ds, cache=True, shuffle_buffer_size=300):

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

    ds = ds.repeat()



    ds = ds.batch(BATCH_TRAIN_SIZE)



    # `prefetch` lets the dataset fetch batches in the background while the model

    # is training.

    ds = ds.prefetch(buffer_size=AUTOTUNE)



    return ds
train_dsfinal = prepare_for_training(train_ds)

test_dsfinal = prepare_for_training(test_ds)
image_batch, label_batch = next(iter(train_dsfinal))



show_batch(image_batch.numpy(), label_batch.numpy())
label_batch.shape
class CNNModel():

    def __init__(self):

        self.inputs = tf.keras.Input(shape=(224,224,3))

        self.x1= tf.keras.layers.Conv2D(32 , 3, activation='relu')(self.inputs)

        self.x1= tf.keras.layers.Conv2D(64, 3, activation='relu')(self.x1)

        self.x1= tf.keras.layers.MaxPooling2D(2,2)(self.x1)

        

        self.x2= tf.keras.layers.Conv2D(32, 3, activation='relu')(self.x1)

        self.x2= tf.keras.layers.Conv2D(64, 3, activation='relu')(self.x2)

        self.x2= tf.keras.layers.MaxPooling2D(3,3)(self.x2)

        

        self.x3= tf.keras.layers.Conv2D(32, 3, activation='relu')(self.x2)

        self.x3= tf.keras.layers.MaxPooling2D(2,2)(self.x3)

        self.x = tf.keras.layers.Dropout(0.2)(self.x3)

        

        self.output = tf.keras.layers.Flatten()(self.x)

        self.output = tf.keras.layers.Dense(224, activation='relu')(self.output)

        self.output = tf.keras.layers.Dense(3, activation='softmax')(self.output) 



        self.model = tf.keras.Model(self.inputs, self.output)

        

        """

        X_input = Input((480, 480, 3))



        X = Conv2D(6, (5, 5), kernel_initializer = glorot_uniform(seed=0))(X_input) #480 - 4 = 476

        X = BatchNormalization(axis = 3)(X)

        X = Activation('relu')(X)

        X = MaxPooling2D((2, 2), strides=(2, 2))(X) # 476 / 2 = 238



        X = Conv2D(16, (5, 5), kernel_initializer = glorot_uniform(seed=0))(X) #238 - 4 = 234

        X = BatchNormalization(axis = 3)(X)

        X = Activation('relu')(X)

        X = MaxPooling2D((2, 2), strides=(2, 2))(X) # 234 / 2 = 117



        X = Conv2D(32, (5, 5), kernel_initializer = glorot_uniform(seed=0))(X) #117 - 4 = 113

        X = BatchNormalization(axis = 3)(X)

        X = Activation('relu')(X)

        X = MaxPooling2D((2, 2), strides=(2, 2))(X) # 113 / 2 = 56



        X = Conv2D(16, (5, 5), kernel_initializer = glorot_uniform(seed=0))(X) #56 - 4 = 52

        X = BatchNormalization(axis = 3)(X)

        X = Activation('relu')(X)

        X = MaxPooling2D((2, 2), strides=(2, 2))(X) # 52 / 2 = 26



        X = Conv2D(5, (5, 5), kernel_initializer = glorot_uniform(seed=0))(X) #26 - 4 = 22

        X = BatchNormalization(axis = 3)(X)

        X = Activation('relu')(X)

        X = MaxPooling2D((2, 2), strides=(2, 2))(X) # 22 / 2 = 11



        model = Model(inputs = X_input, outputs = X, name='ResNet50')

        """





    def compile_cnn(self):

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

        

    def fit(self, dataset, n_epochs):

        self.model.fit(

            dataset,

            steps_per_epoch=STEPS_PER_EPOCH,

            epochs=n_epochs,

            validation_data=test_dsfinal,

            validation_steps=200

        )



# Create an instance of the model

model = CNNModel()
model.compile_cnn()
history = model.fit(dataset = train_dsfinal, n_epochs = EPOCHS)
acc = model.model.history.history['accuracy']

val_acc = model.model.history.history['val_accuracy']



loss = model.model.history.history['loss']

val_loss = model.model.history.history['val_loss']



epochs_range = range(EPOCHS)



plt.figure(figsize=(10, 6))

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
model.model.evaluate(test_dsfinal, verbose=2, steps=64)