import tensorflow as tf

import tensorflow_hub as hub

import os

from typing import Tuple
"""Creates the data generators to be used in training the network

Args

    dataset: location of the dataset

    image_dim: scale to this image dimension

    batch_size: batch size

Returns

    train_generator: ImageDataGenerator

tf.keras.preprocessing : slow in training

"""

def load_generators(dataset_dir: str, image_dim: Tuple[int, int], batch_size: int):

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

        rescale=1./255)



    train_generator = train_datagen.flow_from_directory(

        dataset_dir,

        shuffle=True,

        color_mode="rgb",

        class_mode="categorical",

        target_size=image_dim,

        batch_size=batch_size)



    print('created dataset for {}'.format(dataset_dir))



    return train_generator
BATCH_SIZE = 24

IMG_HEIGHT = 299

IMG_WIDTH = 299



# set the train and validation datasets

train_dir = '/kaggle/input/crowdai-plant-disease-dataset/Custom-Train-Test(color)/color'

validation_dir = '/kaggle/input/crowdai-plant-disease-dataset/Custom-Train-Test(color)/Test'



# load the data generators

train_datagen = load_generators(train_dir, (IMG_HEIGHT, IMG_WIDTH), BATCH_SIZE)



validation_datagen = load_generators(validation_dir, (IMG_HEIGHT, IMG_WIDTH), BATCH_SIZE)
INCEPTIONV3_TFHUB = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'
input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

num_classes = train_datagen.num_classes



# fetch the feature extractor from the tf_hub

feature_extractor = hub.KerasLayer(INCEPTIONV3_TFHUB, input_shape=input_shape)



# make the feature extractor trainable

feature_extractor.trainable = True



# create the sequential model

model = tf.keras.Sequential([

    feature_extractor,

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.0005))

])
# print the summary of the model

model.summary()
# compile the model

model.compile(

    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)
# train the model

model.fit(

    train_datagen,

    epochs=2,

    steps_per_epoch=train_datagen.samples//train_datagen.batch_size,

    validation_data=validation_datagen,

)
# evaluate the model

loss, accuracy = model.evaluate(validation_datagen)

# train accuracy

train_loss, train_accuracy = model.evaluate(train_datagen)
"Trained Model for {} epochs, train accuracy: {:5.2f}%, test accuracy: {:5.2f}%".format(2, 100*train_accuracy, 100*accuracy)