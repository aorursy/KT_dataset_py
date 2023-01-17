import os # file paths

import numpy as np # tensor (> 2D matrix) operations

import pandas as pd # 

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import ResNet50

import matplotlib.pyplot as plt

# import pandas as pd



train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



print(train_data.shape)

print(train_data.head(10))
# from keras.utils import to_categorical



# seperate features and labels

X = train_data.drop(['label'], axis=1).to_numpy()

y = train_data['label'].to_numpy()



# one-hot encode labels

y = to_categorical(y, 10)



# scale pixel values

X = X.astype('float32') / 255.0

test_data = test_data.astype('float32') / 255.0
# from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .2)
# from keras.models import Sequential

# from keras.layers import Dense



mnist_model = Sequential()

mnist_model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))

mnist_model.add(Dense(10, activation='softmax'))



mnist_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
mnist_model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_val, y_val))
X = train_data.drop(['label'], axis=1).to_numpy()

y = train_data['label'].to_numpy()



X = X.reshape((42000, 28, 28, 1))

X = X.astype('float32') / 255



y = to_categorical(y, 10)



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)
# from keras.layers import Conv2D, MaxPooling2D, Flatten



mnist_cnn_model = Sequential()

mnist_cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

mnist_cnn_model.add(MaxPooling2D((2, 2)))

mnist_cnn_model.add(Conv2D(64, (3, 3), activation='relu'))

mnist_cnn_model.add(MaxPooling2D((2, 2)))

mnist_cnn_model.add(Conv2D(64, (3, 3), activation='relu'))

mnist_cnn_model.add(Flatten())

mnist_cnn_model.add(Dense(64, activation='relu'))

mnist_cnn_model.add(Dense(10, activation='softmax'))

mnist_cnn_model.summary()
mnist_cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

mnist_cnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val))
cats_dogs_model = Sequential()

cats_dogs_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

cats_dogs_model.add(MaxPooling2D((2, 2)))

cats_dogs_model.add(Conv2D(64, (3, 3), activation='relu'))

cats_dogs_model.add(MaxPooling2D(2, 2))

cats_dogs_model.add(Conv2D(128, (3, 3), activation='relu'))

cats_dogs_model.add(MaxPooling2D(2, 2))

cats_dogs_model.add(Conv2D(128, (3, 3), activation='relu'))

cats_dogs_model.add(MaxPooling2D(2, 2))

cats_dogs_model.add(Flatten())

cats_dogs_model.add(Dense(512, activation='relu'))

cats_dogs_model.add(Dense(1, activation='sigmoid'))



cats_dogs_model.summary()
cats_dogs_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
# from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory('/kaggle/input/cats-and-dogs-small/cats_and_dogs_small/train', target_size=(150, 150), batch_size=20, class_mode='binary')

validation_generator = train_datagen.flow_from_directory('/kaggle/input/cats-and-dogs-small/cats_and_dogs_small/validation', target_size=(150, 150), batch_size=20, class_mode='binary')
cats_dogs_model_history = cats_dogs_model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50)
# import matplotlib.pyplot as plt



def display_model_performance(model_history):

    acc = model_history.history['acc']

    val_acc = model_history.history['val_acc']

    loss = model_history.history['loss']

    val_loss = model_history.history['val_loss']



    epochs = range(1, len(acc) + 1)



    plt.plot(epochs, acc, 'bo', label='Training acc')

    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()



    plt.figure()



    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()



    plt.show()

    

display_model_performance(cats_dogs_model_history)
train_datagen = ImageDataGenerator(

    rescale = 1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True

)
# from keras.preprocessing import image



img = image.load_img('/kaggle/input/cats-and-dogs-small/cats_and_dogs_small/train/cats/cat.3.jpg', target_size=(150, 150))



x = image.img_to_array(img)



plt.figure(0)

imgplot = plt.imshow(image.array_to_img(x))

plt.title('Original Image')



x = x.reshape((1,) + x.shape)



i = 1

for batch in train_datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i == 5:

        break



plt.show()
test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory('/kaggle/input/cats-and-dogs-small/cats_and_dogs_small/train', target_size=(150, 150), batch_size=32, class_mode='binary')

validation_generator = test_datagen.flow_from_directory('/kaggle/input/cats-and-dogs-small/cats_and_dogs_small/validation', target_size=(150, 150), batch_size=32, class_mode='binary')
# from keras.layers import Dropout

# 

cats_dogs_dropout_model = Sequential()

cats_dogs_dropout_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

cats_dogs_dropout_model.add(MaxPooling2D((2, 2)))

cats_dogs_dropout_model.add(Conv2D(64, (3, 3), activation='relu'))

cats_dogs_dropout_model.add(MaxPooling2D((2, 2)))

cats_dogs_dropout_model.add(Conv2D(128, (3, 3), activation='relu'))

cats_dogs_dropout_model.add(MaxPooling2D((2, 2)))

cats_dogs_dropout_model.add(Conv2D(128, (3, 3), activation='relu'))

cats_dogs_dropout_model.add(MaxPooling2D((2, 2)))

cats_dogs_dropout_model.add(Flatten())

cats_dogs_dropout_model.add(Dropout(0.5))

cats_dogs_dropout_model.add(Dense(512, activation='relu'))

cats_dogs_dropout_model.add(Dense(1, activation='sigmoid'))



cats_dogs_dropout_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])



cats_dogs_dropout_model.summary()
cats_dogs_dropout_model_history = cats_dogs_dropout_model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)
display_model_performance(cats_dogs_dropout_model_history)