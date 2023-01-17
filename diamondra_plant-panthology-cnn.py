# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

train_csv_path = '/kaggle/input/plant-pathology-2020-fgvc7/train.csv'

test_csv_path = '/kaggle/input/plant-pathology-2020-fgvc7/test.csv'

folder_images = '/kaggle/input/plant-pathology-2020-fgvc7/images'
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os
def add_filename(file_path):

    data = pd.read_csv(file_path)

    data['filename'] = '/kaggle/input/plant-pathology-2020-fgvc7/images/'+ data['image_id'] + '.jpg'

    data.drop(['image_id'], axis = 1)

    return data



def prepare_data(file_path):

    data = pd.read_csv(file_path)

    y = data[['healthy', 'multiple_diseases', 'scab', 'rust']]

    df = pd.DataFrame({

        'filename': '/kaggle/input/plant-pathology-2020-fgvc7/images/'+ data['image_id'] + '.jpg',

        'category': np.where(y==1)[1]

    })

    

    return df
data = prepare_data(train_csv_path)

test = add_filename(test_csv_path)
data["category"] = data["category"].replace({0: 'healthy', 1: 'multiple_diseases', 2: 'scab', 3: 'rust'}) 
data
train, validate = train_test_split(data, test_size=0.20, random_state=1)

train = train.reset_index(drop=True)

validate = validate.reset_index(drop=True)
IMAGE_WIDTH=1024

IMAGE_HEIGHT=1024

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3

OUTPUT = 4

batch_size = 5
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train, 

    x_col='filename',

    y_col= 'category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate, 

    x_col='filename',

    y_col= 'category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
example = train.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example,  

    x_col='filename',

    y_col= 'category',

    target_size=IMAGE_SIZE,

    class_mode='categorical'

)
plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



model = Sequential()



model.add(Conv2D(256, (3, 3),strides = 2, activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(256, (3, 3),strides = 2, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))







model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(OUTPUT, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]
total_train = train.shape[0]

total_validate = validate.shape[0]
epochs= 20



history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test,  

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
model.save_weights("model.h5")
nb_samples = test.shape[0]
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
submission = test.copy()

submission['healthy'] = [row[0] for row in predict]

submission['multiple_diseases'] =  [row[1] for row in predict]

submission['rust'] =  [row[2] for row in predict]

submission['scab'] =  [row[3] for row in predict]
submission = submission.drop(['filename'], axis = 1)
submission.to_csv('submission.csv', index=False)