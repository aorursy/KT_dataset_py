# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/vgg-weights"))



# Any results you write to the current directory are saved as output.
!wget https://github.com/bogdanluncasu/kmnist_generators/blob/master/generator_0.h5?raw=true -O generators_0.h5

!wget https://github.com/bogdanluncasu/kmnist_generators/blob/master/generator_1.h5?raw=true -O generators_1.h5

!wget https://github.com/bogdanluncasu/kmnist_generators/blob/master/generator_2.h5?raw=true -O generators_2.h5

!wget https://github.com/bogdanluncasu/kmnist_generators/blob/master/generator_3.h5?raw=true -O generators_3.h5

!wget https://github.com/bogdanluncasu/kmnist_generators/blob/master/generator_4.h5?raw=true -O generators_4.h5

!wget https://github.com/bogdanluncasu/kmnist_generators/blob/master/generator_5.h5?raw=true -O generators_5.h5

!wget https://github.com/bogdanluncasu/kmnist_generators/blob/master/generator_6.h5?raw=true -O generators_6.h5

!wget https://github.com/bogdanluncasu/kmnist_generators/blob/master/generator_7.h5?raw=true -O generators_7.h5

!wget https://github.com/bogdanluncasu/kmnist_generators/blob/master/generator_8.h5?raw=true -O generators_8.h5

!wget https://github.com/bogdanluncasu/kmnist_generators/blob/master/generator_9.h5?raw=true -O generators_9.h5
generators = []



from keras.models import load_model

for i in range(10):

    generators.append(load_model(f'generators_{i}.h5'))
def use_generators(_tmp):

    global X_train, y_train

    import keras

    r, c = _tmp,_tmp

    noise = np.random.normal(0, 1, (r * c, 100))

    gen_imgs = []

    gen_labels = []

    i=0

    for gen in generators:

        gen_imgs.append(gen.predict(noise))

        gen_labels.append(np.full((r*c,),i))

        i+=1

    

    y_gen = [keras.utils.to_categorical(labels,10) for labels in gen_labels]

    

    for i in range(0,10):

        X_train = np.concatenate((X_train,gen_imgs[i]), axis=0)

        y_train = np.concatenate((y_train,y_gen[i]), axis=0)

    print(X_train.shape)

    print(y_train.shape)
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, Input, MaxPooling2D

from keras.layers.merge import add

from keras.activations import relu, softmax

from keras.models import Model

from keras import regularizers

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import keras
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import os



from keras import layers



def VGG16(include_top=True):

    

    img_input = layers.Input(shape=(28,28,1))



    x = layers.Conv2D(64, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block1_conv1')(img_input)

    x = layers.Conv2D(64, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block1_conv2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)



    # Block 2

    x = layers.Conv2D(128, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block2_conv1')(x)

    x = layers.Conv2D(128, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block2_conv2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)



    # Block 3

    x = layers.Conv2D(256, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv1')(x)

    x = layers.Conv2D(256, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv2')(x)

    x = layers.Conv2D(256, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv3')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)



    # Block 4

    x = layers.Conv2D(512, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block4_conv1')(x)

    x = layers.Conv2D(512, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block4_conv2')(x)

    x = layers.Conv2D(1028, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block4_conv3')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    

    

    if include_top:

        # Classification block

        x = layers.Flatten(name='flatten')(x)

        x = layers.Dense(4096, activation='relu', name='fc1')(x)

        x = layers.Dense(4096, activation='relu', name='fc2')(x)

        x = layers.Dense(10, activation='softmax', name='predictions')(x)



    # Create model.

    model = Model(img_input, x, name='vgg16')



    return model
model = VGG16(include_top=True)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])

model.load_weights("../input/vgg-weights/weights.best (1).keras")
import numpy as np

import os

train_images = np.load(os.path.join('../input/cursive-hiragana-classification','train-imgs.npz'))['arr_0']

test_images = np.load(os.path.join('../input/cursive-hiragana-classification','test-imgs.npz'))['arr_0']

train_labels = np.load(os.path.join('../input/cursive-hiragana-classification','train-labels.npz'))['arr_0']
def data_preprocessing(images):

    num_images = images.shape[0]

    x_shaped_array = images.reshape(num_images, 28, 28, 1)

    out_x = x_shaped_array / np.std(x_shaped_array, axis = 0)

    return out_x
X = data_preprocessing(train_images)

y = keras.utils.to_categorical(train_labels, 10)

X_test = data_preprocessing(test_images)
X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

mc = ModelCheckpoint('weights.best.keras', monitor='val_acc', save_best_only=True)

from keras.preprocessing.image import ImageDataGenerator

EPOCHS = 310

datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.1, width_shift_range=0.09, shear_range=0.28, height_shift_range=0.09 )

datagen.fit(X_train)
# use_generators(70)
hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),

                    epochs=EPOCHS, validation_data=(x_val, y_val),steps_per_epoch=85000/32, callbacks=[mc])
model.load_weights('weights.best.keras')
predicted_classes = model.predict(X_test)
import numpy as np

submission = pd.read_csv(os.path.join("../input/cursive-hiragana-classification","sample_submission.csv"))

submission['Class'] = np.argmax(predicted_classes, axis=1)

submission.to_csv(os.path.join(".","submission.csv"), index=False)



new_cols = ["p0","p1","p2","p3","p4","p5","p6","p7","p8","p9"]

new_vals = predicted_classes

submission = submission.reindex(columns=submission.columns.tolist() + new_cols)

submission[new_cols] = new_vals



submission.to_csv(os.path.join(".","submission_arr.csv"), index=False)
#get the predictions for the test data

predicted_classes = model.predict(x_val)

#get the indices to be plotted

y_true = np.argmax(y_val,axis=1)
correct = np.nonzero(np.argmax(predicted_classes,axis=1)==y_true)[0]

incorrect = np.nonzero(np.argmax(predicted_classes,axis=1)!=y_true)[0]
print("Correct predicted classes:",correct.shape[0])

print("Incorrect predicted classes:",incorrect.shape[0])