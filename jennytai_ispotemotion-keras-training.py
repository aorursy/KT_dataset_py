import os

import numpy as np

import pandas as pd

import keras

import matplotlib.pyplot as plt

import tensorflow as tf

from keras import models

from tensorflow.keras import layers
num_expressions = 6

img_size = 48

epochs = 50

batch_size = 64

num_features = 64

data = pd.read_csv('../input/facial-expression/fer2013.csv')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(data.head(10))

print(data.shape)

print(data.Usage.value_counts())
def add_one(x):

    if x>1:

        x=x-1

    return x



data = data[data['emotion']!=1]

data['emotion'] = data['emotion'].apply(add_one)



print(data.head(10))
emotion_map = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprise', 5: 'Neutral'}

emotion_counts = data['emotion'].value_counts(sort=False).reset_index()

emotion_counts.columns = ['emotion', 'number']

emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)

print(emotion_counts)
def row2image(row):

    pixels, emotion = row['pixels'], emotion_map[row['emotion']]

    img = np.array(pixels.split())

    img = img.reshape(img_size, img_size)

    image = np.zeros((img_size, img_size, 3))

    image[:, :, 0] = img

    image[:, :, 1] = img

    image[:, :, 2] = img

    return np.array([image.astype(np.uint8), emotion])





# show sample emotion expressions from dataset

def showFace(index):

    plt.figure(0, figsize=(16, 10))

    for i in range(1, 7):

        face = data[data['emotion'] == i - 1].iloc[index]

        img = row2image(face)

        plt.subplot(2, 3, i)

        plt.imshow(img[0])

        plt.title(img[1])

    plt.show()
showFace(19)
data_train = data[data['Usage'] == 'Training'].copy()

data_val = data[data['Usage'] == 'PublicTest'].copy()

data_test = data[data['Usage'] == 'PrivateTest'].copy()



# use all the data to train the final model

data_train = data.copy()
def CRNO(df, name):

    # convert pixels strings to integer lists

    df['pixels'] = df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])

    # to image, reshape and normalize grayscale

    x = np.array(df['pixels'].tolist()).reshape(-1, img_size, img_size, 1) / 255.0

    y = keras.utils.to_categorical(df['emotion'], num_expressions)

    print(name, "_X shape: {}, ", name, "_Y shape: {}".format(x.shape, y.shape))

    return x, y
train_X, train_Y = CRNO(data_train, "train")  # training data

val_X, val_Y = CRNO(data_val, "val")  # validation data

test_X, test_Y = CRNO(data_test, "test")  # test data
model = models.Sequential()

model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(num_expressions))



# model.summary()



model.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
# data generator

data_generator = keras.preprocessing.image.ImageDataGenerator(

                        featurewise_center=False,

                        featurewise_std_normalization=False,

                        rotation_range=10,

                        width_shift_range=0.1,

                        height_shift_range=0.1,

                        zoom_range=.1,

                        horizontal_flip=True)



es = keras.callbacks.ModelCheckpoint('/kaggle/working/my_model', monitor='val_loss', savebest_only=True)



history = model.fit(data_generator.flow(train_X, train_Y, 128), batch_size=128, epochs=200, shuffle=True, callbacks = [es], validation_data=(val_X, val_Y)) 

# history = model.fit(train_X, train_Y, epochs=50, batch_size=128, validation_data=(val_X, val_Y), callbacks = [es], shuffle=True)
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.4, 0.7])

plt.legend(loc='lower right')



test_loss, test_acc = model.evaluate(test_X,  test_Y, verbose=2)

print(test_acc)

model.save('saved_model/my_model')  
new_model = tf.keras.models.load_model('saved_model/my_model')

new_model.summary()



test_loss, test_acc = new_model.evaluate(test_X,  test_Y, verbose=2)

print(test_acc)
