# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_dir = '/kaggle/input/aerial-cactus-identification/train/train/'

test_dir = '/kaggle/input/aerial-cactus-identification/test/'



train = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
train.head()
train.has_cactus = train.has_cactus.astype(str)
train[train['has_cactus']=='1']['id'].count()/train['id'].count()
class_weight = {

    '0.0':0.25, 

    '1.0':0.75

}
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, Dropout, Input, Flatten
IMG_SHAPE = (32,32)
train_datagen = ImageDataGenerator( rescale = 1./255,

                                   rotation_range=40,

                                    zoom_range = 0.2, 

                                   shear_range = 0.2,

                                   horizontal_flip=True, 

                                   validation_split=0.2

                                  )

training_data = train_datagen.flow_from_dataframe(train, 

                                                 train_dir,

                                                  x_col='id', y_col='has_cactus',

                                                 target_size=IMG_SHAPE,

                                                  batch_size = 150,

                                                  shuffle=True,

                                                  class_mode = 'binary', 

                                                  subset='training'

                                                 )



validation_data = train_datagen.flow_from_dataframe(train, 

                                                 train_dir,

                                                  x_col='id', y_col='has_cactus',

                                                 target_size=IMG_SHAPE,

                                                  batch_size = 150,

                                                  shuffle=True,

                                                  class_mode = 'binary', 

                                                  subset='validation'

                                                 )



# test datagen = ImageDataGenerator()



# testing_data = test_datagen.flow_from_dataframe()
input_img = Input(shape = (32,32,3))

x = Conv2D(128, (3,3), padding = 'same', activation='relu')(input_img)

x = MaxPool2D((2,2))(x)

x = Dropout(0.25)(x)



x = Conv2D(128, (3,3), padding = 'same', activation='relu')(x)

x = MaxPool2D((2,2))(x)

x = Dropout(0.25)(x)



# x = Conv2D(64, (3,3), padding = 'same', activation='relu')(x)

# x = MaxPool2D((2,2))(x)

# x = Dropout(0.25)(x)





# x = Conv2D(32, (3,3), padding = 'same', activation='relu')(x)

# x = MaxPool2D((2,2))(x)

# x = Dropout(0.25)(x)



# x = BatchNormalization()(x)



x = Flatten()(x)

x = Dense(1296, activation='relu')(x)

# x = Dense(256, activation='relu')(x)

x = Dense(64, activation='relu')(x)

x = Dense(1, activation='sigmoid')(x)





model = Model(input_img, x)

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(training_data, 

                              epochs=20, 

                              steps_per_epoch=14000//150,

#                               class_weight=class_weight,

                              validation_data= validation_data, 

                              validation_steps= 3500//150

                              

                             )
def plot_history(history):

    fig = plt.figure(figsize = (20, 6))

    plt.subplot(1, 2, 1)

    plt.plot(history.history['acc'], label='Train Acc')

    plt.plot(history.history['val_acc'], label='Validation Acc')

    plt.title("Accuracy")

    plt.legend()

    plt.grid()

    

    plt.subplot(1, 2, 2)

    plt.plot(history.history['loss'], label='Train loss')

    plt.plot(history.history['val_loss'], label='Validation loss')

    plt.title("Loss")

    plt.legend()

    plt.grid()

    

    plt.show()

    
plot_history(history)

to_submit = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')

to_submit.head()
submission_datagen = ImageDataGenerator(rescale=1.0/255)



submission_gen = submission_datagen.flow_from_directory(test_dir,

#                                                        x_col = 'id',

#                                                        y_col=None,

                                                       batch_size = 1,

                                                        class_mode=None,

                                                        shuffle=False,

                                                        target_size = IMG_SHAPE

                                                       )

classes = model.predict_generator(submission_gen, steps = len(submission_gen.filenames))
classes.shape
to_submit.count()
to_submit['has_cactus'] = classes.flatten()

to_submit.head(10)
to_submit.to_csv('samplesubmission.csv',index=False)