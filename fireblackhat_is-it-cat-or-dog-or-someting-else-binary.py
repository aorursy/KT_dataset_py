# Importing Required Packages



import numpy as np # linear algebra

import pathlib

from tensorflow import keras # Keras

import matplotlib.pyplot as plt # Plot - Visualization

# Image Test

from urllib import request

from io import BytesIO

import skimage.transform

import skimage.io
# Setup Directories

training_datadir = pathlib.Path("/kaggle/input/cat-and-dog/training_set/")

test_datadir = pathlib.Path("/kaggle/input/cat-and-dog/test_set/")



train_images = np.array([item.name for item in training_datadir.glob("*/*.jpg")])

train_image_count = len(train_images)

#train_classes = list(item.name for item in training_datadir.glob("*") if item.name != "LICENSE.txt") ##Binary



test_images = np.array([item.name for item in test_datadir.glob("*/*.jpg")])

test_image_count = len(test_images)

#test_classes = list(item.name for item in test_datadir.glob("*") if item.name != "LICENSE.txt") ##Binary
# Training Conf

BATCH_SIZE = 32

IMG_HEIGHT = 64

IMG_WIDTH = 64

TRAIN_STEPS_PER_EPOCH = np.ceil(train_image_count / BATCH_SIZE)

TEST_STEPS_PER_EPOCH = np.ceil(test_image_count / BATCH_SIZE)

EPOCHS = 2



# Image Generators (Training-Test)

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, 

                                   shear_range = 0.2, 

                                   zoom_range = 0.2, 

                                   horizontal_flip = True)



training_data = train_datagen.flow_from_directory(

    str(training_datadir),

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

  #  classes=train_classes, ##Binary

    class_mode="binary",

    shuffle=True,

)



test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

test_data = test_datagen.flow_from_directory(

    str(test_datadir),

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

 #   classes=test_classes, ##Binary

    class_mode="binary",

    shuffle=True,

)
# Keras Model



model = keras.Sequential(

    [

        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(32, (3, 3), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation="relu"),

        keras.layers.MaxPool2D((2, 2)),

        keras.layers.Flatten(),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(128, activation="relu"),

        keras.layers.Dense(training_data.num_classes, activation="sigmoid"),

    ]

)
# Compile and Train

model.compile(

    optimizer="Adam",

    loss="binary_crossentropy",

    metrics=["Accuracy"],

)



historical = model.fit(training_data, validation_data=test_data, steps_per_epoch=TRAIN_STEPS_PER_EPOCH, validation_steps=TEST_STEPS_PER_EPOCH, epochs=EPOCHS)
# Saving Time!

model.save_weights("model.h5")
# Visualizing - Epochs Against Train-Val Loss & Accuracy



# summarize history for accuracy

plt.plot(historical.history['Accuracy'])

plt.plot(historical.history['Accuracy'])



# Vis Title

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

# Show Vis

plt.show()



# Save Vis

plt.savefig('accuracy.png')



# summarize history for loss

plt.plot(historical.history['loss'])

plt.plot(historical.history['val_loss'])



# Vis Title

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

# Show Vis

plt.show()



# Save Vis

plt.savefig('loss.png')

# Testing an Image

image_url = 'https://s.abcnews.com/images/US/airbus-gty-er-190417_hpMain_16x9_992.jpg'

image_data = BytesIO(request.urlopen(image_url).read())

image = skimage.io.imread(image_data)



# Predicting

predicton = model.predict_classes(np.array(

        [

            skimage.transform.resize(item, (IMG_WIDTH, IMG_HEIGHT, 3))

            for item in [image]

        ]

    ))



print(predicton)