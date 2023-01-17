! pip install imutils
import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

import pickle

import random

import time

import cv2

import os



from sklearn.preprocessing import LabelBinarizer

from imutils import paths
# get hold of the train images directory

data_path_train = '../input/seg_train/seg_train'

# get hold of the test images directory

data_path_test = '../input/seg_test/seg_test'



def get_images_and_labels(data_path):

    images = []

    image_labels = []



    # get the image paths 

    image_paths = sorted(list(paths.list_images(data_path)))

    random.seed(42)

    # shuffle the images

    random.shuffle(image_paths)

    

    for image_path in image_paths:

        # load and resize the images

        image = cv2.imread(image_path)

        image = cv2.resize(image, (128, 128))

        images.append(image)



        # get the train labels

        image_label = image_path.split(os.path.sep)[-2]

        image_labels.append(image_label)



    # rescale the image pixels

    images = np.array(images, dtype='float') / 255.0

    # make the `image_labels` as array

    image_labels = np.array(image_labels)

    

    return images, image_labels
train_X, train_y = get_images_and_labels(data_path_train)

test_X, test_y = get_images_and_labels(data_path_test)
print(len(train_X))

print(len(test_X))
# one-hot encode the labels

lb = LabelBinarizer()

train_y = lb.fit_transform(train_y)

test_y = lb.fit_transform(test_y)
# generator for image augmentation

image_aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30, 

                                                            shear_range=0.2, 

                                                            zoom_range=0.2, 

                                                            height_shift_range=0.2, 

                                                            width_shift_range=0.2, 

                                                            horizontal_flip=True, 

                                                            vertical_flip=True,

                                                            fill_mode='nearest')
# build the model

model = tf.keras.models.Sequential()

input_shape = (128, 128, 3)



model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', 

            activation='relu', input_shape=input_shape))

model.add(tf.keras.layers.BatchNormalization(axis=-1))      

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.2))

        

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', 

                                 activation='relu'))

model.add(tf.keras.layers.BatchNormalization(axis=-1))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', 

                                 activation='relu'))

model.add(tf.keras.layers.BatchNormalization(axis=-1))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', 

                                 activation='relu'))

model.add(tf.keras.layers.BatchNormalization(axis=-1))

model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', 

                                 activation='relu'))

model.add(tf.keras.layers.BatchNormalization(axis=-1))

model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', 

                                 activation='relu'))

model.add(tf.keras.layers.BatchNormalization(axis=-1))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Dense(len(lb.classes_), activation='softmax'))
optimizer = tf.keras.optimizers.Adam(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, 

              metrics=['accuracy'])



history = model.fit_generator(image_aug.flow(train_X, train_y, 

                                             batch_size=64), 

                                             validation_data=(test_X, test_y), 

                                             steps_per_epoch=len(train_X)//64, 

                                             epochs=30)
model.save('scene_classification.model')

f = open('scene_classification_lb.pickle', 'wb')

f.write(pickle.dumps(lb))

f.close()
num_epochs = np.arange(0, 30)

plt.figure(dpi=300)

plt.plot(num_epochs, history.history['loss'], label='train_loss', c='red')

plt.plot(num_epochs, history.history['val_loss'], 

    label='val_loss', c='orange')

plt.plot(num_epochs, history.history['acc'], label='train_acc', c='green')

plt.plot(num_epochs, history.history['val_acc'], 

    label='val_acc', c='blue')

plt.title('Training Loss and Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Loss/Accuracy')

plt.legend()

plt.savefig('plot.png')
# load the test image

image = cv2.imread('../input/seg_pred/seg_pred/350.jpg')

output = image.copy()

image = cv2.resize(image, (128, 128))



# scale the pixels

image = image.astype('float') / 255.0



image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
model = tf.keras.models.load_model('scene_classification.model')

lb = pickle.loads(open('scene_classification_lb.pickle', 'rb').read())
# predict

preds = model.predict(image)



# get the class label

max_label = preds.argmax(axis=1)[0]

print('PREDICTIONS: \n', preds)

print('PREDICTION ARGMAX: ', max_label)

label = lb.classes_[max_label]

print(label)
# class label along with the probability

text = '{}: {:.2f}%'.format(label, preds[0][max_label] * 100)

plt.figure(figsize=(3, 3))

plt.text(0, -3, text, fontsize=12)

plt.imshow(output[:, :, :])