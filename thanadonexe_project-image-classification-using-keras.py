# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
from keras.models import Sequential   

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import SeparableConv2D

from keras import regularizers

from keras import optimizers as Optimizer

from keras.layers import Dropout
model = Sequential()

model.add(Conv2D(32, (3,3),padding="same",activation="relu",input_shape=(150,150,3)))

model.add(MaxPooling2D(2,2))



model.add(SeparableConv2D(64 , (3,3),padding="same",activation="relu"))

model.add(MaxPooling2D(2,2))



model.add(SeparableConv2D(128,(3,3),padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))



model.add(SeparableConv2D(128,(3,3),padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))



model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.8))

model.add(Dense(6,activation="softmax"))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
train_path = '/kaggle/input/intel-image-classification/seg_train/seg_train'

test_path = '/kaggle/input/intel-image-classification/seg_test/seg_test'

pred_path = '/kaggle/input/intel-image-classification/seg_pred'
from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(

        train_path,

        target_size=(150, 150),

        batch_size=100,

        class_mode='categorical')
test_generator = test_datagen.flow_from_directory(

        test_path,

        target_size=(150, 150),

        batch_size=100,

        class_mode='categorical')
model.fit_generator(

        train_generator,

        steps_per_epoch=len(train_generator),

        epochs=30,

        validation_data=test_generator,

        validation_steps=len(test_generator))
model.save('model.hdf5')
evaluation = model.evaluate_generator(test_generator, steps=len(test_generator))

print(f'Best model loss: {round(evaluation[0], 2)}')

print(f'Best model accuracy: {round(evaluation[1] * 100, 2)}%')
pred_generator = test_datagen.flow_from_directory(

    pred_path,

    target_size=(150, 150),

    batch_size=100,

    class_mode=None,

    shuffle=False

)
from keras.models import load_model

model = load_model('model.hdf5')

predictions = model.predict_generator(pred_generator, steps=len(pred_generator), verbose=1)
import cv2

 

filenames = pred_generator.filenames

rows, cols = (4, 4)



predicted_class_indices = np.argmax(predictions, axis = 1)

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions_label = [labels[k] for k in predicted_class_indices]



fig = plt.figure(figsize = (12, 12))

for i in range(rows * cols):

    r = np.random.randint(0, len(filenames)-1)

    image_path = pred_path + '/' + filenames[r]

    image = cv2.imread(image_path)

    fig.add_subplot(rows, cols, i+1)

    plt.imshow(image[:, :, ::-1])

    plt.title(f'\nPrediction: {predictions_label[r]}')

    plt.axis('off')



plt.tight_layout()