from keras.datasets import mnist

from matplotlib import pyplot as plt



(x_train, y_train), (x_test, y_test) = mnist.load_data()



for i in range(0, 9):

    plt.subplot(330 + 1 + i)

    plt.imshow(x_train[i], cmap=plt.get_cmap("gray"))

    

plt.show()
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K



x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)



x_train = x_train.astype("float32")

x_test = x_test.astype("float32")



train_datagen = ImageDataGenerator(rotation_range=60)

train_datagen.fit(x_train)



for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):

    for i in range(9):

        plt.subplot(330 + 1 + i)

        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap("gray"))

    plt.show()

    break
train_datagen = ImageDataGenerator(shear_range=0.5, zoom_range=0.5)

train_datagen.fit(x_train)



for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):

    for i in range(9):

        plt.subplot(330 + 1 + i)

        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap("gray"))

    plt.show()

    break
train_datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True)

train_datagen.fit(x_train)



for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):

    for i in range(9):

        plt.subplot(330 + 1 + i)

        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap("gray"))

    plt.show()

    break
train_datagen = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3)

train_datagen.fit(x_train)



for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):

    for i in range(9):

        plt.subplot(330 + 1 + i)

        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap("gray"))

    plt.show()

    break
train_datagen = ImageDataGenerator(rotation_range=45,

                                   width_shift_range=0.3,

                                   height_shift_range=0.3,

                                   shear_range=0.5,

                                   zoom_range=0.5,

                                   horizontal_flip=True,

                                   fill_mode="nearest")

train_datagen.fit(x_train)



for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):

    for i in range(9):

        plt.subplot(330 + 1 + i)

        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap("gray"))

    plt.show()

    break