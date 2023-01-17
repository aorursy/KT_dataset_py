import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from PIL import Image

print(os.listdir("../input"))
path_dogs = r'../input/training_set/training_set/dogs/'

path_cats = r'../input/training_set/training_set/cats/'

path_test_cats = r'../input/test_set/test_set/cats/'

path_test_dogs = r'../input/test_set/test_set/dogs/'

training_dogs = os.listdir(path_dogs)

training_cats = os.listdir(path_cats)

testing_dogs = os.listdir(path_test_dogs)

testing_cats = os.listdir(path_test_cats)

if '.DS_Store' in training_dogs:

    training_dogs.remove('.DS_Store')

    

if '.DS_Store' in testing_dogs:

    testing_dogs.remove('.DS_Store')



if '.DS_Store' in training_cats:

    training_cats.remove('.DS_Store')

    

if '.DS_Store' in testing_cats:

    testing_cats.remove('.DS_Store')
dog_fig, dog_axes = plt.subplots(3,3)

dog_fig.set_figheight(24)

dog_fig.set_figwidth(24)



for i, axis in enumerate(dog_axes):

    for j, f in enumerate(axis):

        img = plt.imread(path_dogs+training_dogs[i*3+j])

        dog_axes[i,j].imshow(img)

        dog_axes[i,j].set_title("Shape.{}".format(img.shape))
cat_fig, cat_axes = plt.subplots(3,3)

cat_fig.set_figheight(24)

cat_fig.set_figwidth(24)



for i, axis in enumerate(cat_axes):

    for j, f in enumerate(axis):

        img = plt.imread(path_cats+training_cats[i*3+j])

        cat_axes[i,j].imshow(img)

        cat_axes[i,j].set_title("Shape.{}".format(img.shape))
input_shape = (224,224,3)

num_classes = 2
from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.optimizers import SGD

from keras.utils import np_utils

from keras.applications.mobilenet import MobileNet
mobilenet = MobileNet(input_shape, weights='imagenet', include_top=False)
model = Sequential()

model.add(mobilenet)

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
img_resize = (224,224)

dogs_train = []

cats_train = []

dogs_test = []

cats_test = []

for dog in training_dogs:

    img = Image.open(path_dogs + dog)

    img = img.resize(img_resize)

    img = np.asarray(img)

    dogs_train.append(img)

    

for cat in training_cats:

    img = Image.open(path_cats + cat)

    img = img.resize(img_resize)

    img = np.asarray(img)

    cats_train.append(img)

    

for dog in testing_dogs:

    img = Image.open(path_test_dogs + dog)

    img = img.resize(img_resize)

    img = np.asarray(img)

    dogs_test.append(img)

    

for cat in testing_cats:

    img = Image.open(path_test_cats + cat)

    img = img.resize(img_resize)

    img = np.asarray(img)

    cats_test.append(img)



dogs_train_label = [0 for _ in range(len(dogs_train))]

cats_train_label = [1 for _ in range(len(cats_train))]

dogs_test_label = [0 for _ in range(len(dogs_test))]

cats_test_label = [1 for _ in range(len(cats_test))]



dogs_train_label.extend(cats_train_label)

dogs_test_label.extend(cats_test_label)

label_train = np_utils.to_categorical(dogs_train_label, num_classes)

label_test = np_utils.to_categorical(dogs_test_label, num_classes)





#for train data

dogs_train.extend(cats_train)

data_train = np.array(dogs_train, dtype=np.float32)

data_train /= 255

index = np.arange(len(data_train))

np.random.shuffle(index)

data_train = data_train[index]

label_train = label_train[index]



#for test data

dogs_test.extend(cats_test)

data_test = np.array(dogs_test, dtype=np.float32)

data_test /= 255

index = np.arange(len(data_test))

np.random.shuffle(index)

data_test = data_test[index]

label_test = label_test[index]

data_train = np.squeeze(data_train)

data_test = np.squeeze(data_test)

label_train = np.squeeze(label_train)

label_test = np.squeeze(label_test)

print("Training label shape: ", label_train.shape)

print("Testing label shape: ", label_test.shape)

print("Training data shape: ", data_train.shape)

print("Testing data shape: ", data_test.shape)
history = model.fit(data_train, label_train, validation_split=0.2, batch_size=4, epochs=6)
print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
test_result = model.predict(data_test, batch_size=4)



_fig, _axes = plt.subplots(3,3)

_fig.set_figheight(24)

_fig.set_figwidth(24)



for i in range(3):

    for j in range(3):

        img = data_test[i*3+j]

        _axes[i,j].imshow(img)

        if np.argmax(test_result[i*3+j]) == 0:

            _axes[i,j].set_title("Prediction  {}".format('Dog'))

        else:

            _axes[i,j].set_title("Prediction  {}".format('Cat'))
_fig, _axes = plt.subplots(3,3)

_fig.set_figheight(24)

_fig.set_figwidth(24)



wrong_index = 0

for i in range(3):

    for j in range(3):

        while wrong_index < len(data_test):

            if np.argmax(test_result[wrong_index]) != np.argmax(label_test[wrong_index]):

                img = data_test[wrong_index]

                _axes[i,j].imshow(img)

                if np.argmax(test_result[wrong_index]) == 0:

                    _axes[i,j].set_title("Prediction  {}".format('Dog'))

                else:

                    _axes[i,j].set_title("Prediction  {}".format('Cat'))

                wrong_index += 1

                break

            wrong_index += 1

        if wrong_index ==len(data_test):

            break