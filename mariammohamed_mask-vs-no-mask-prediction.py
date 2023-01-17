import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
import keras
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras import backend as K
train_csv = pd.read_csv('../input/train.csv')
train_csv.head()
ids = train_csv.id.tolist()
images = np.array(list(map(lambda x: cv2.imread('../input/train/images/'+ x + '.png'), ids)))
rle_mask = train_csv.rle_mask.isna()
images.shape
images = np.array(list(map(lambda x: cv2.resize(x, (299, 299)), images)))
images.shape
X_train, X_test, y_train, y_test = train_test_split(images, rle_mask, test_size=0.2, random_state=42)
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
vgg.summary()
# input = Input(shape=(299, 299, 3))


layers = dict([(layer.name, layer) for layer in vgg.layers])

vgg_top = layers['block5_conv2'].output

x = Flatten(name='flatten')(vgg_top)
x = Dense(2048, activation='relu', name='fc1')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = Dense(512, activation='relu', name='fc3')(x)
x = Dense(1, activation='sigmoid', name='predictions')(x)

my_model = Model(input=vgg.input, output=x)
my_model.summary()
for layer in vgg.layers:
    layer.trainable = False
sgd = keras.optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=True)
my_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=45,
    horizontal_flip=True,
    vertical_flip=True)


datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
my_model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=50)
my_model.save_weights('model_50.h5')
my_model.evaluate(X_test, y_test)
K.eval(my_model.optimizer.lr.assign(0.0001))
# fits the model on batches with real-time data augmentation:
my_model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=20)
my_model.evaluate(X_test, y_test)
