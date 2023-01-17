from __future__ import division, print_function, absolute_import

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
# read training & testing data

trainImg = pd.read_csv("../input/ahdd1/csvTrainImages 60k x 784.csv",header=None)
trainLabel = pd.read_csv("../input/ahdd1/csvTrainLabel 60k x 1.csv",header=None)

testImg = pd.read_csv("../input/ahdd1/csvTestImages 10k x 784.csv",header=None)
testLabel = pd.read_csv("../input/ahdd1/csvTestLabel 10k x 1.csv",header=None)
trainImg.head()
testImg.head()
# Split data into training set and validation set
#training images
trainImg = trainImg.values.astype('float32') /255.0
#training labels
trainLabel = trainLabel.values.astype('int32') 

#testing images
testImg = testImg.values.astype('float32') /255.0
#testing labels
testLabel = testLabel.values.astype('int32')
testImg
#One Hot encoding of train labels.
trainLabel = to_categorical(trainLabel,10)

#One Hot encoding of test labels.
testLabel = to_categorical(testLabel,10)
trainLabel[0]
print(trainImg.shape, trainLabel.shape, testImg.shape, testLabel.shape)
# reshape input images to 28x28x1
trainImg = trainImg.reshape([-1, 28, 28, 1])
testImg = testImg.reshape([-1, 28, 28, 1])
print(trainImg.shape, trainLabel.shape, testImg.shape, testLabel.shape)
trainImg[0]
# Building convolutional network
model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1), padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=10, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(trainImg, trainLabel, 
          batch_size=100, epochs=1, verbose=1)
print('Predict the classes: ')
prediction = model.predict_classes(trainImg)
print('Predicted classes: ', prediction)
# Evaluate model
score = model.evaluate(testImg, testLabel,verbose=0)
print(model.metrics_names)
print('Loss accuracy: %2f%%' % (score[0] * 100))
print('Test accuarcy: %2f%%' % (score[1] * 100))
model.save('LeNet-5')
from keras.models import load_model

import cv2
from PIL import Image,ImageEnhance
import PIL.ImageOps
from matplotlib import pyplot as plt
import numpy as np

model = load_model('LeNet-5')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

cleanImage = Image.open('../input/ontwth/123.jpg')
crop1=(55.2034,235.397,227.397,411.966);
crop2=(446.179,235.397,601.728,411.966);
crop3=(820.338,235.397,988.499,411.966);
crop4=(1190.29,235.397,1337.43,411.966);
crop5=(1539.23,235.397,1690.57,411.966);
crop6=(1888.16,235.397,2018.49,411.966);

images = [0,0,0,0,0,0]
final = [0,0,0,0,0,0]

crops = [crop1, crop2, crop3, crop4, crop5, crop6]
for i in range(6):
    images[i] = cleanImage.crop(crops[i])
for i in range(6):
    images[i] = images[i].resize((32,32))
    images[i] = ImageEnhance.Sharpness(images[i])
    images[i] = images[i].enhance(10.0)
    images[i] = PIL.ImageOps.invert(images[i])
    final[i] = np.asarray(images[i])[:,:,::-1].copy()
cleanImage.show();

for i in range(6):
    plt.subplot(2,6,i+1),plt.imshow(images[i],'gray')
    plt.xticks([]),plt.yticks([])
plt.show()

for i in range(6):
    img = cv2.fastNlMeansDenoisingColored(final[i], None, 10, 10, 7, 21)
    img = cv2.resize(img,(28,28))
    img = img.reshape((-1, 28, 28,1))
    classes = model.predict_classes(img)
    print (classes)
    rediction = classes.argmax()
    print (rediction)

