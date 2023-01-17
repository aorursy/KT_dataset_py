# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

os.chdir('../input')



# Any results you write to the current directory are saved as output.
print(os.getcwd())

print(os.listdir())
print(os.listdir('fer2013-images/images/images_fer2013/Training'))
print(os.listdir('fer2013-images/images/images_fer2013'))
print(os.listdir('xception/'))
train_dir = 'fer2013-images/images/images_fer2013/Training/'

validation_dir = 'fer2013-images/images/images_fer2013/PublicTest/'
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

                            rotation_range=40,

                            width_shift_range=0.2,

                            height_shift_range=0.2,

                            shear_range=0.2,

                            zoom_range=0.2,

                            horizontal_flip=True,

                            fill_mode='nearest')
import matplotlib.pyplot as plt

from keras.preprocessing import image



fnames = [os.path.join(train_dir, 'Sad', fname) for fname in os.listdir(os.path.join(train_dir, 'Sad'))]



img_path = fnames[1]



img = image.load_img(img_path, target_size = (48, 48))



x = image.img_to_array(img)



x = x.reshape((1,) + x.shape)



i = 0

for batch in datagen.flow(x, batch_size = 1):

    plt.figure()

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break

plt.show()
from keras.applications.xception import Xception



conv_base = Xception(weights='xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',

                     include_top=False,

                     input_shape=(75, 75, 3))
train_datagen = ImageDataGenerator(

                rescale = 1./255,

                rotation_range=40,

                width_shift_range=0.2,

                height_shift_range=0.2,

                shear_range=0.2,

                zoom_range=0.2,

                horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

                                train_dir,

                                target_size = (75, 75),

                                batch_size = 32,

                                class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

                                validation_dir,

                                target_size = (75, 75),

                                batch_size = 32,

                                class_mode = 'categorical')
conv_base.summary()
from  keras import models

from keras import layers



model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation = 'relu'))

model.add(layers.Dense(7, activation = 'softmax'))
model.summary()

print('This is the number of trainable weights before freezing the conv base: ',len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))
import keras



callbacks_list = [

    keras.callbacks.ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.1,

    patience=10,

    )

]
from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers



train_datagen = ImageDataGenerator(

                                rescale = 1./255,

                                rotation_range = 40,

                                width_shift_range = 0.2,

                                height_shift_range = 0.2,

                                shear_range = 0.2,

                                zoom_range=0.2,

                                horizontal_flip=True,

                                fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(75, 75),

        color_mode = 'rgb',

        batch_size = 20,

        class_mode = 'categorical')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size = (75, 75),

        color_mode = 'rgb',

        batch_size = 20,

        class_mode = 'categorical')



model.compile(loss = 'categorical_crossentropy',

            optimizer = optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])



history = model.fit_generator(

        train_generator,

        steps_per_epoch=1436,

        callbacks = callbacks_list,

        epochs=30,

        validation_data=validation_generator,

        validation_steps = 180)
conv_base.trainable = True



set_trainable = False

for layer in conv_base.layers:

    if layer.name[:8] == 'block14_':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer

model.summary()
model.compile(loss = 'categorical_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-5),

             metrics=['acc'])



history = model.fit_generator(

    train_generator,

    steps_per_epoch=1436,

    epochs=60,

    validation_data = validation_generator,

    validation_steps=180)
model.save('fer2013_model.h5')
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()



plt.show()
test_dir = 'fer2013-images/images/images_fer2013/PrivateTest/'

test_generator = test_datagen.flow_from_directory(test_dir,

                                                 target_size=(75, 75),

                                                 batch_size=20,

                                                 color_mode = 'rgb',

                                                 class_mode='categorical')



test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

print('test acc:', test_acc)
model.predict_generator(test_generator, steps=len(test_generator))
import os



os.listdir('../')
os.chdir('../input/')
os.listdir('fer2013-images/images/images_fer2013/PrivateTest/')