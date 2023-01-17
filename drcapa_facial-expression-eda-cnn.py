import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt
from keras import models

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.utils import to_categorical
path = '/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/'

os.listdir(path)
data = pd.read_csv(path+'icml_face_data.csv')
data.head()
def prepare_data(data):

    """ Prepare data for modeling 

        input: data frame with labels und pixel data

        output: image and label array """

    

    image_array = np.zeros(shape=(len(data), 48, 48))

    image_label = np.array(list(map(int, data['emotion'])))

    

    for i, row in enumerate(data.index):

        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')

        image = np.reshape(image, (48, 48))

        image_array[i] = image

        

    return image_array, image_label
data[' Usage'].value_counts()
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
train_image_array, train_image_label = prepare_data(data[data[' Usage']=='Training'])

val_image_array, val_image_label = prepare_data(data[data[' Usage']=='PrivateTest'])

test_image_array, test_image_label = prepare_data(data[data[' Usage']=='PublicTest'])
train_images = train_image_array.reshape((train_image_array.shape[0], 48, 48, 1))

train_images = train_images.astype('float32')/255

val_images = val_image_array.reshape((val_image_array.shape[0], 48, 48, 1))

val_images = val_images.astype('float32')/255

test_images = test_image_array.reshape((test_image_array.shape[0], 48, 48, 1))

test_images = test_images.astype('float32')/255
train_labels = to_categorical(train_image_label)

val_labels = to_categorical(val_image_label)

test_labels = to_categorical(test_image_label)
class_weight = dict(zip(range(0, 7), (((data[data[' Usage']=='Training']['emotion'].value_counts()).sort_index())/len(data[data[' Usage']=='Training']['emotion'])).tolist()))
class_weight
model = models.Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))

model.add(MaxPool2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPool2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(7, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_images, train_labels,

                    validation_data=(val_images, val_labels),

                    class_weight = class_weight,

                    epochs=15,

                    batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test caccuracy:', test_acc)
pred_test_labels = model.predict(test_images)
loss = history.history['loss']

loss_val = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='loss_train')

plt.plot(epochs, loss_val, 'b', label='loss_val')

plt.title('value of the loss function')

plt.xlabel('epochs')

plt.ylabel('value of the loss function')

plt.legend()

plt.grid()

plt.show()
acc = history.history['accuracy']

acc_val = history.history['val_accuracy']

epochs = range(1, len(loss)+1)

plt.plot(epochs, acc, 'bo', label='accuracy_train')

plt.plot(epochs, acc_val, 'b', label='accuracy_val')

plt.title('accuracy')

plt.xlabel('epochs')

plt.ylabel('value of accuracy')

plt.legend()

plt.grid()

plt.show()
def plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, image_number):

    """ Function to plot the image and compare the prediction results with the label """

    

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    

    bar_label = emotions.values()

    

    axs[0].imshow(test_image_array[image_number], 'gray')

    axs[0].set_title(emotions[test_image_label[image_number]])

    

    axs[1].bar(bar_label, pred_test_labels[image_number])

    axs[1].grid()

    

    plt.show()
plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, 26)
plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, 10)