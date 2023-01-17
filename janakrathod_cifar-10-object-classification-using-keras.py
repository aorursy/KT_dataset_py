import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, BatchNormalization
from keras.layers import Dense, Dropout

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras
def load_cifar10_data(batch_number):
    with open('../input/data_batch_'+ str(batch_number), 'rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data']
    labels = batch['labels']
    return features, labels
batch_1, labels_1 = load_cifar10_data(1)
batch_2, labels_2 = load_cifar10_data(2)
batch_3, labels_3 = load_cifar10_data(3)
batch_4, labels_4 = load_cifar10_data(4)
batch_5, labels_5 = load_cifar10_data(5)
all_images = np.append(batch_1, batch_2, axis=0)
all_images = all_images.reshape((len(all_images), 3, 32, 32)).transpose(0,2,3,1)
all_labels = np.append(labels_1, labels_2, axis=0)
dict = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer', 5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}
def plot_image(number):
    fig = plt.figure(figsize = (15,8))
    plt.imshow(all_images[number])
    plt.title(dict[all_labels[number]])
plot_image(321)
plot_image(2490)
plot_image(4201)
plot_image(3430)
X_train = np.append(batch_1[0:8000], batch_2[0:8000], axis=0)
X_train = np.append(X_train, batch_3[0:8000], axis=0)
X_train = np.append(X_train, batch_4[0:8000], axis=0)
X_train = np.append(X_train, batch_5[0:8000], axis=0)
Y_train = np.append(labels_1[0:8000], labels_2[0:8000], axis=0)
Y_train = np.append(Y_train, labels_3[0:8000], axis=0)
Y_train = np.append(Y_train, labels_4[0:8000], axis=0)
Y_train = np.append(Y_train, labels_5[0:8000], axis=0)
X_validation = np.append(batch_1[8001:9000], batch_2[8001:9000], axis=0)
X_validation = np.append(X_validation, batch_3[8001:9000], axis=0)
X_validation = np.append(X_validation, batch_4[8001:9000], axis=0)
X_validation = np.append(X_validation, batch_5[8001:9000], axis=0)
Y_validation = np.append(labels_1[8001:9000], labels_2[8001:9000], axis=0)
Y_validation = np.append(Y_validation, labels_3[8001:9000], axis=0)
Y_validation = np.append(Y_validation, labels_4[8001:9000], axis=0)
Y_validation = np.append(Y_validation, labels_5[8001:9000], axis=0)
X_test = np.append(batch_1[9001:10000], batch_2[9001:10000], axis=0)
X_test = np.append(X_test, batch_3[9001:10000], axis=0)
X_test = np.append(X_test, batch_4[9001:10000], axis=0)
X_test = np.append(X_test, batch_5[9001:10000], axis=0)
Y_test = np.append(labels_1[9001:10000], labels_2[9001:10000], axis=0)
Y_test = np.append(Y_test, labels_3[9001:10000], axis=0)
Y_test = np.append(Y_test, labels_4[9001:10000], axis=0)
Y_test = np.append(Y_test, labels_5[9001:10000], axis=0)
print("Length of X_train:", len(X_train), "Length of Y_train:", len(Y_train))
print("Length of X_validation:",len(X_validation), "Length of Y_validation:", len(Y_validation))
print("Length of X_test:",len(X_test), "Length of Y_test:", len(Y_test))
Y_train_one_hot = np_utils.to_categorical(Y_train, 10)
Y_validation_one_hot = np_utils.to_categorical(Y_validation, 10)
Y_test_one_hot = np_utils.to_categorical(Y_test, 10)
X_train = X_train.reshape((len(X_train), 3, 32, 32)).transpose(0,2,3,1)
X_validation = X_validation.reshape((len(X_validation), 3, 32, 32)).transpose(0,2,3,1)
X_test = X_test.reshape((len(X_test), 3, 32, 32)).transpose(0,2,3,1)
classifier = Sequential()

# Convolution layer 1
classifier.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu', border_mode='same', bias=True))

# Pooling layer 1
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))

# Convolution and pooling layer 2
classifier.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', bias=True))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.3))

# Classifier and pooling layer 3.
classifier.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', bias=True))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.3))

# Classifier and pooling layer 4.
classifier.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', bias=True))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.3))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(BatchNormalization())
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 10, activation='softmax'))

classifier.summary()

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_set = train_datagen.flow(X_train, Y_train_one_hot, batch_size=32)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_set = validation_datagen.flow(X_validation, Y_validation_one_hot, batch_size=32)


classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit_generator(train_set,
                    steps_per_epoch=40000,epochs=10,
                    validation_data=(validation_set), validation_steps=4995, shuffle=True)
X_test = (X_test)*1./255 
scores = classifier.evaluate(X_test, Y_test_one_hot, batch_size=32)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
def show_test(number):
    fig = plt.figure(figsize = (15,8))
    test_image = np.expand_dims(X_test[number], axis=0)
    test_result = classifier.predict_classes(test_image)
    plt.imshow(X_test[number])
    dict_key = test_result[0]
    plt.title("Predicted: {}, Actual: {}".format(labels[dict_key], labels[Y_test[number]]))
show_test(123)
show_test(9)
show_test(490)
show_test(558)