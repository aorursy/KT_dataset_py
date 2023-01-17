import numpy as np

from keras import layers

from keras.layers import Input, Dense, Activation, Flatten, Conv2D

from keras.layers import MaxPooling2D, Dropout

from keras.models import Model, Sequential

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

import keras.backend as K

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
train_dir = '../input/cats-and-dogs/train/' 

val_dir = '../input/cats-and-dogs/validate/' 

test_dir = '../input/cats-and-dogs/test/'



input_shape = (200,200,3)

batch_size = 50

nb_train_samples = 20000

nb_val_samples = 2500

nb_test_samples = 2500

data_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = data_gen.flow_from_directory(

    train_dir,

    target_size = (200,200),

    batch_size = batch_size,

    class_mode = 'binary'

)



val_generator = data_gen.flow_from_directory(

    val_dir,

    target_size = (200,200),

    batch_size = batch_size,

    class_mode = 'binary'

)



test_generator = data_gen.flow_from_directory(

    test_dir,

    target_size = (200,200),

    batch_size = batch_size,

    class_mode = 'binary'

)
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
%%time

history = model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=30,

    validation_data=val_generator,

    validation_steps=nb_val_samples // batch_size)
%%time

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

print("Точность на тестовых данных: %.2f%%" % (scores[1]*100))
plt.plot(history.history['loss'], color='r', label= "validation loss")

plt.title("Test Loss")

plt.xlabel("Number of epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
plt.plot(history.history['val_acc'], color='b', label= "accuracy")

plt.title("Test Accuracy")

plt.xlabel("Number of epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
%%time

history = model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=15,

    validation_data=val_generator,

    validation_steps=nb_val_samples // batch_size)
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

print("Точность на тестовых данных: %.2f%%" % (scores[1]*100))
plt.plot(history.history['acc'], color='b', label= "accuracy")

plt.title("Test Accuracy")

plt.xlabel("Number of epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], color='r', label= "validation loss")

plt.title("Test Loss")

plt.xlabel("Number of epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
import h5py

from keras.models import load_model

import os



os.chdir(r'/kaggle/working')

model.save('my_model.h5')
%%time

from keras.applications.inception_v3 import InceptionV3

input_tensor = Input(shape = (200, 200, 3))





incept_model = InceptionV3(weights=None, include_top=False, input_tensor=input_tensor)

#Kaggle не закачивает с весами, тк падает с ошибкой времени выполнения, потэтому приходится добавлять веса вручную

incept_model.load_weights('/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
data_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = data_gen.flow_from_directory(

    train_dir,

    target_size = (200,200),

    batch_size = batch_size,

    class_mode = 'binary',

    shuffle=False

)



val_generator = data_gen.flow_from_directory(

    val_dir,

    target_size = (200,200),

    batch_size = batch_size,

    class_mode = 'binary',

    shuffle=False

)
%%time

features_train = incept_model.predict_generator(train_generator, nb_train_samples // batch_size)

np.save(open('features_train.npy', 'wb'), features_train)
%%time

features_validation = incept_model.predict_generator(val_generator, nb_val_samples // batch_size)

np.save(open('features_validation.npy', 'wb'), features_validation)
incept_train= np.load(open('/kaggle/working/features_train.npy','rb'))

# Благодаря запрету shuffle'a можно утверждать, что сначала идут nb_train_samples/2 котиков, потом собак

train_labels= np.array([0] * 10000 + [1] * 10000)



incept_validation = np.load(open('/kaggle/working/features_validation.npy','rb'))

validation_labels = np.array([0] * 1250 + [1] * 1250)
inc_model = Sequential()

inc_model.add(Flatten(input_shape=incept_train.shape[1:]))

inc_model.add(Dense(64, activation='relu'))

inc_model.add(Dropout(0.5))

inc_model.add(Dense(64, activation='relu'))

inc_model.add(Dropout(0.5))

inc_model.add(Dense(1, activation='sigmoid'))



inc_model.compile(optimizer='rmsprop', 

              loss='binary_crossentropy', 

              metrics=['accuracy'])
%%time

inc_model.fit(incept_train, train_labels,

              epochs=50, batch_size=50,

              validation_data=(incept_validation, validation_labels))
%%time

inc_model.evaluate(incept_validation, validation_labels)
os.chdir(r'/kaggle/working')

inc_model.save('inc_model.h5')
check_test_dir = '../input/check-dataset/check_ds/'



input_shape = (200,200,3)

nb_test_samples = 6
check_test_generator = data_gen.flow_from_directory(

    check_test_dir,

    target_size = (200,200),

    batch_size = 6,

    class_mode = 'binary'

)
%%time

scores = model.evaluate_generator(check_test_generator, 6)

print("Точность на тестовых данных: %.2f%%" % (scores[1]*100))
check_features_validation = incept_model.predict_generator(check_test_generator, 1)

np.save(open('check_features_validation.npy', 'wb'), check_features_validation)



check_incept_train= np.load(open('/kaggle/working/check_features_validation.npy','rb'))

check_labels= np.array([0] * 3 + [1] * 3)



inc_model.evaluate(check_incept_train, check_labels)