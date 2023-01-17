import warnings

warnings.filterwarnings('ignore')



import tensorflow.keras as keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications.xception import Xception



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import PIL.Image



daisy_path = "../input/flower/flower_classification/train/daisy/"

dandelion_path = "../input/flower/flower_classification/train/dandelion/"

rose_path = "../input/flower/flower_classification/train/rose/"

sunflower_path = "../input/flower/flower_classification/train/sunflower/"

tulip_path = "../input/flower/flower_classification/train/tulip/"

test_path="../input/flower/flower_classification/test/"

submission = pd.read_csv('../input/submission.csv')
from os import listdir

import cv2







img_data = []

labels = []



size = 224,224

def iter_images(images,directory,size,label):

    try:

        for i in range(len(images)):

            img = cv2.imread(directory + images[i])

            img = cv2.resize(img,size,PIL.Image.ANTIALIAS)

            img_data.append(img)

            labels.append(label)

    except:

        pass



iter_images(listdir(daisy_path),daisy_path,size,0)

iter_images(listdir(dandelion_path),dandelion_path,size,1)

iter_images(listdir(rose_path),rose_path,size,2)

iter_images(listdir(sunflower_path),sunflower_path,size,3)

iter_images(listdir(tulip_path),tulip_path,size,4)
len(img_data),len(labels)
test_data = []



size = 224,224

def test_images(images,directory,size):

    try:

        for i in range(len(images)):

            img = cv2.imread(directory + submission['id'][i]+".jpg")

            img = cv2.resize(img,size,PIL.Image.ANTIALIAS)

            test_data.append(img)

    except:

        pass





test_images(listdir(test_path),test_path,size)
len(test_data)
train_X = np.asarray(img_data)

train_Y = np.asarray(labels)



idx = np.arange(train_X.shape[0])

np.random.shuffle(idx)



train_X = train_X[idx]

train_Y = train_Y[idx]



testData=np.asarray(test_data)



print(train_X.shape)

print(train_Y.shape)
model_name = 'Xception-Fine-Tune'



img_rows, img_cols, img_channel = 224, 224, 3

base_model = Xception(weights='imagenet', include_top=False,

                         input_shape=(img_rows, img_cols, img_channel))



x = base_model.output

x = GlobalAveragePooling2D(data_format='channels_last')(x)

x = Dropout(0.5)(x)

predictions = Dense(5, activation='softmax')(x)



model = Model(inputs=base_model.input, outputs=predictions)



model.summary()
datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.1,

    zoom_range=0.1,

    horizontal_flip=True,

    fill_mode='nearest')



optimizer = keras.optimizers.Adam(lr=10e-6)



model_path = './{}.h5'.format(model_name)



# checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)

# earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)



model.compile(loss='sparse_categorical_crossentropy',

              optimizer=optimizer, metrics=['accuracy'])



batch_size = 64

History = model.fit_generator(datagen.flow(train_X, train_Y, batch_size = batch_size),

                                    steps_per_epoch=len(train_X) / 64,

                        epochs=15)
plt.plot(History.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train'])

plt.show()
plt.plot(History.history['acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train'])

plt.show()
pred =  np.argmax(model.predict(testData), axis=1)

newsSbmission=submission

newsSbmission["class"]=pred

newsSbmission.to_csv("submission.csv", index=False)
pred