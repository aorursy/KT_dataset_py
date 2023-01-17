# Import Libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import cv2 

import os



import keras.backend as k

import tensorflow as tf



from sklearn.model_selection import train_test_split

from tqdm import tqdm
# Getting the training images and labels

train = pd.read_csv('../input/train.csv')



train_labels = train['has_cactus']

train_images = []



for img in tqdm(train['id']):

    img_path = '../input/train/train/'+img;

    train_images.append(cv2.resize(cv2.imread(img_path), (70, 70)))

train_X = np.asarray(train_images)

train_Y = pd.DataFrame(train_labels)
plt.title(train_Y['has_cactus'][0])

_ = plt.imshow(train_X[0])
plt.title(train_Y['has_cactus'][1000])

_ = plt.imshow(train_X[1000])
x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)
import keras

from keras import layers

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator


input_shape = (70, 70, 3)

dropout_dense_layer = 0.6



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(Flatten())

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dropout(dropout_dense_layer))



model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(dropout_dense_layer))



model.add(Dense(1))

model.add(Activation('sigmoid'))



opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator()
datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=50), steps_per_epoch=x_train.shape[0], epochs=2, validation_data=(x_test, y_test), verbose=1)
[loss, accuracy] = model.evaluate(x_test, y_test)
print('Test Set Accuracy: '+str(accuracy*100)+"%");
# Getting the test set images

test_path = '../input/test/test/'

test_images_names = []



for filename in tqdm(os.listdir(test_path)):

    test_images_names.append(filename)

    

test_images_names.sort()



images_test = []



for image_id in tqdm(test_images_names):

    images_test.append(np.array(cv2.resize(cv2.imread(test_path + image_id), (70, 70))))

    

images_test = np.asarray(images_test)

images_test = images_test.astype('float32')

images_test /= 255
# making predictions

prediction = model.predict(images_test)
predict = []

for i in range(len(prediction)):

    if prediction[i][0]>0.5:

        answer = prediction[i][0]

    else:

        answer = prediction[i][0]

    predict.append(answer)
submission = pd.read_csv('../input/sample_submission.csv')

submission['has_cactus'] = predict
# Creating the final submission file

submission.to_csv('sample_submission.csv',index = False)