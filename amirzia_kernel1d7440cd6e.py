import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



import keras

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D



from sklearn.model_selection import train_test_split



from collections import Counter



import matplotlib.pyplot as plt
dataset_path = '/kaggle/input/planesnet/planesnet/planesnet/'

files = os.listdir(dataset_path)

files = list(filter(lambda x: x.startswith('0') or x.startswith('1'), files))
batch_size = 32

epochs = 100

data_augmentation = True

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'keras_plane_trained_model.h5'
def data_from_file(path):

    label = int(path.split('__')[0])

    img = plt.imread(os.path.join(dataset_path, path))

    return label, img
all_data = []

labels = []

for file in files:

    label, data = data_from_file(file)

    all_data.append(data)

    labels.append(label)



all_data = np.array(all_data)

labels = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(all_data, labels, train_size=0.8)



print(len(x_train), 'train samples')

print(len(x_test), 'test samples')
model = Sequential()

model.add(Conv2D(32, (2, 2), padding='same',

                 input_shape=x_train.shape[1:]))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Activation('relu'))

model.add(Conv2D(32, (2, 2)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Activation('relu'))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



opt = keras.optimizers.adam(learning_rate=0.001, decay=1e-6)

model.compile(loss='binary_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])
datagen = ImageDataGenerator(rotation_range=90)

datagen.fit(x_train)



model.fit_generator(datagen.flow(x_train, y_train,

                                 batch_size=batch_size),

                    epochs=epochs,

                    validation_data=(x_test, y_test))



if not os.path.isdir(save_dir):

    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)
scores = model.evaluate(x_test, y_test, verbose=1)

print('Test accuracy:', scores[1])
print("Number of positive istances in test set:", np.sum(y_test == 1))

print("Number of negative istances in test set:", np.sum(y_test == 0))