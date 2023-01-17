# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.applications.xception import Xception 

from keras.layers import *

from keras.optimizers import *

from keras.applications import *

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.xception import preprocess_input

from keras.applications.xception import decode_predictions

# hyper parameters for model

nb_classes = 4  # number of classes

img_width, img_height = 299, 299  # change based on the shape/structure of your images

batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).

nb_epoch = 200  # number of iteration the algorithm gets trained.

learn_rate = 1e-3  # sgd learning rate

momentum = .9  # sgd momentum to avoid local minimum







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.
import os

print(os.listdir("../input/dokurek/data/validation"))
def load_model(weights_path):

    base_model = Xception(input_shape=(img_width, img_height, 3), weights=None, include_top=False)



    # Top Model Block

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    predictions = Dense(nb_classes, activation='softmax')(x)



    # add your top layer block to your base model

    model = Model(base_model.input, predictions)

    model.load_weights(weights_path)

    model.compile(optimizer='nadam',

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
frame = pd.read_csv('../input/data-undistorted/data_undistorted/data.csv', sep=';')

model = load_model('../input/model-weights2/model_weights2.h5')

validation_data_dir = '../input/data-undistorted/data_undistorted/validation'

train_data_dir = '../input/data-undistorted/data_undistorted/train'

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(train_data_dir,

                                                        target_size=(img_width, img_height),

                                                        batch_size=batch_size,

                                                        class_mode='categorical')

validation_generator = datagen.flow_from_directory(validation_data_dir,

                                                        target_size=(img_width, img_height),

                                                        batch_size=batch_size,

                                                        class_mode='categorical')



score = model.evaluate_generator(validation_generator, 1, verbose=1)

score

# filenames = test_generator.filenames

# nb_samples = len(filenames)



# predictions = model.predict_generator(test_generator,steps=1)

# predictions = np.argmax(predictions, axis=-1) #multiple categories



# label_map = (train_generator.class_indices)

# label_map = dict((v,k) for k,v in label_map.items()) #flip k,v

# predictions = [label_map[k] for k in predictions]

# predictions
score_train = model.evaluate_generator(train_generator, 1, verbose=1)

score_train
import os

d = '../input/model-weights2/model_weights2.h5'

print(os.listdir(d))