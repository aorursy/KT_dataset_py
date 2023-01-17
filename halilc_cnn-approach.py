!pip install split_folders



import split_folders

import os, random

import numpy as np

import matplotlib.pyplot as plt

import shutil



from PIL import Image

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping
split_folders.ratio('../input/flowers/flowers', output="output", ratio=(.8, .2))
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('./output/train',

                                                 target_size = (64, 64),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./output/val',

                                            target_size = (64, 64),

                                            batch_size = 1,

                                            class_mode = 'categorical')
classifier = Sequential()



# Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))



# Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Second convolutional layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Flattening

classifier.add(Flatten())



# Full connection

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 5, activation = 'sigmoid'))



# Compiling

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the CNN

es = EarlyStopping(monitor='val_loss', patience=2)



training_steps = training_set.n//training_set.batch_size

test_steps = test_set.n//test_set.batch_size



history = classifier.fit_generator(training_set,

                         steps_per_epoch = training_steps,

                         epochs = 30,

                         validation_data = test_set,

                         validation_steps = test_steps,

                         callbacks=[es]

                        )
def getLabel(value):

    for k, v in test_set.class_indices.items():

        if v == value:

            return k
path = "./output/val"

dirs = os.listdir(path)



plt.axis('off')

plt.figure(figsize=(10,10))



picsPerFlower = 5

for dirIdx, directory in enumerate(dirs):

    for i in range(picsPerFlower):

        file_path = path + "/" + directory + "/" + random.choice(os.listdir(path + "/" + directory))

        test_image = image.load_img(

            file_path,

            target_size=(64, 64)

        )

        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis = 0)

        predicted = getLabel(classifier.predict(test_image)[0].argmax(axis=-1))



        ax = plt.subplot(len(dirs), picsPerFlower, (dirIdx * picsPerFlower) + i + 1)

        ax.set_xticklabels([])

        ax.set_yticklabels([])

        plt.imshow(np.array(Image.open(file_path)), aspect='auto')

        backgroundcolor = "g" if directory == predicted else 'r'

        ax.text(0.5, 0.2, predicted, transform=ax.transAxes, ha="center", backgroundcolor=backgroundcolor)

        if i == 2:

            ax.text(0.5, 1.05, "Predictions for " + directory, transform=ax.transAxes, ha="center")
shutil.rmtree('./output')