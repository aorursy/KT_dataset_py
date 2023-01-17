import numpy as np

import os

import pandas

import random

import matplotlib.pyplot as plt

from keras import layers, models, optimizers

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

%matplotlib inline
os.chdir('../input/digit-recognizer')



def create_images(filename, mode='train'):

    dataset = pandas.read_csv(filename)

    images = []

    for i in range(0, dataset.shape[0]):

        if mode == 'train':

            pixels = dataset.iloc[i].tolist()[1:]

        else:

            pixels = dataset.iloc[i].tolist()

        img = np.zeros((28, 28))

        for j in range(0, len(pixels)):

            img[j // 28, j % 28] = pixels[j]

        images.append(img)

    if mode == 'train':

        return (np.asarray(dataset['label'].tolist()), np.asarray(images))

    else:

        return np.asarray(images)
train_labels, train_images = create_images('train.csv')
plt.figure(figsize=(20, 15))

for i in range(50):

    plt.subplot(5, 10, i + 1)

    plt.imshow(train_images[random.randrange(train_images.shape[0])])
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
img_train, img_test, label_train, label_test = train_test_split(train_images, train_labels, test_size = 0.1)
train_images.shape[0] == img_train.shape[0] + img_test.shape[0]
train_labels.shape[0] == label_train.shape[0] + label_test.shape[0]
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))  

model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.4))

model.add(layers.Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(layers.Dense(10, activation='softmax'))
model.summary()
image_gen = ImageDataGenerator(rotation_range=10,

                               width_shift_range=0.1,

                               height_shift_range=0.1,

                               rescale=1./255,

                               shear_range=0.2,

                               zoom_range=0.2,

                               fill_mode='nearest')
train_datagen = image_gen.flow(img_train, label_train, batch_size=64)
model.compile(loss='sparse_categorical_crossentropy', 

              optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),

              metrics=['accuracy'])
history = model.fit_generator(train_datagen, steps_per_epoch=img_train.shape[0] // 64, 

                    epochs=20, verbose=1, callbacks=None, validation_data=(img_test, label_test))
acc = history.history['acc']

val_acc = history.history['val_acc']

plt.plot(range(20), acc, 'r', label='Training accuracy')

plt.plot(range(20), val_acc, 'b', label='Validation accuracy')

plt.legend()

plt.show()
loss = history.history['loss']

val_loss = history.history['val_loss']

plt.plot(range(20), loss, 'r', label='Training loss')

plt.plot(range(20), val_loss, 'b', label='Validation loss')

plt.legend()
test_images = create_images('test.csv', 'test')
answer = model.predict(test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
def classes(answer):

    numbers_classes = []

    for item in answer:

        numbers_classes.append(item.argmax())

    return numbers_classes
result = pandas.DataFrame({'ImageId': range(1, answer.shape[0] + 1), "Label": classes(answer)})
os.chdir('/kaggle/working')
result.to_csv('result.csv', index=False)
model.save('mnist.h5')