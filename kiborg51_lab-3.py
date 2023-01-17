import keras

from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



import matplotlib.pyplot as plt # В коде из примера эта библиотека не подключена
batch_size = 128

num_classes = 10

epochs = 10

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape)

print(train_labels.shape)

print(test_images.shape)

print(test_labels.shape)
train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)

test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)



train_labels = keras.utils.to_categorical(train_labels, num_classes)

test_labels = keras.utils.to_categorical(test_labels, num_classes)



train_images = train_images / 255.0

test_images = test_images / 255.0
print(train_images.shape)

print(train_labels.shape)

print(test_images.shape)

print(test_labels.shape)
plt.figure(figsize = (10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(test_images[i])

plt.show()
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation ='relu',

                input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes,activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

             optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])



model.fit(train_images, train_labels, batch_size=batch_size,epochs = epochs,verbose=1)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dropout

from keras.layers import BatchNormalization

from keras import backend as K



import matplotlib.pyplot as plt
from keras.datasets import cifar10



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
def ShowData(data):

    plt.figure(figsize = (10,10))

    for i in range(25):

        plt.subplot(5,5,i+1)

        plt.imshow(data[i])

    plt.show()
ShowData(x_train)
y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



x_train = x_train / 255.0

x_test = x_test / 255.0



print(y_train.shape)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation ='relu',

                input_shape=(32,32,3)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10,activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

             optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=128,epochs = epochs,verbose=1)
def TestModel(model, test_image, test_labels):

    from sklearn.metrics import confusion_matrix

    import numpy as np

    import seaborn as sn

    import pandas as pd

    rounded_predictions = model.predict_classes(test_image, batch_size=128, verbose=0)

    rounded_labels=np.argmax(test_labels, axis=1)



    conf = confusion_matrix(rounded_predictions,rounded_labels) # Матрица ошибок

    conf = conf / 1000.0



    test_loss, test_acc = model.evaluate(test_image, test_labels, verbose=2) # Проверка точности



    print("Loss: ", test_loss)

    print("Accuracy: ", test_acc)



    df_cm = pd.DataFrame(conf)

    plt.figure(figsize = (10,10))

    sn.heatmap(df_cm, annot=True)
TestModel(model, x_test, y_test)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

             optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=128,epochs = 10,verbose=1)
TestModel(model, x_test, y_test)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model.add(Dense(10, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

             optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=128,epochs = 20,verbose=1)
TestModel(model, x_test, y_test)