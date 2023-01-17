# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras import regularizers, optimizers

from keras.models import Model

from keras.layers import Input

from keras.applications import inception_v3

from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.optimizers import Adam

from keras.layers import GlobalAveragePooling2D

from keras.layers import Dropout

from keras.regularizers import l1,l2

import os

from keras.models       import Model

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Reading data

def data_reader(data): 

    read_data = pd.read_csv(data)

    return(read_data)
def Build_model():

    base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(200, 200, 3))

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.2)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.2)(x)

    x = Dense(2048, activation='relu')(x)

    predictions = Dense(5, activation='softmax')(x)

    # The model we will train

    model = Model(inputs = base_model.input, outputs = predictions)

    # first: train only the top layers i.e. freeze all convolutional InceptionV3 layers

    for layer in base_model.layers:

        layer.trainable = False

    # Compile model

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

train_df = data_reader('../input/train.csv')

train_df['diagnosis'] = train_df['diagnosis'].astype('str')

train_df['Image_name'] = train_df['id_code'].astype(str)+'.png'

train_df = train_df.drop(columns = ['id_code'])



from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(

    rescale=1./255,

    rotation_range = 10,

    shear_range = 10,

    zoom_range = 0.2,

    horizontal_flip = True,

    vertical_flip = True,

    validation_split=0.1)





batch_size = 32



training_set=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="Image_name",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    target_size=(200,200),

    subset='training')



testing_set=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="Image_name",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical", 

    target_size=(200,200),

    subset='validation')

classifier = Build_model()

classifier.fit_generator(training_set,

                         steps_per_epoch = 100,

                         epochs = 50,

                         validation_data = testing_set,

                         validation_steps = 10)
classifier.save_weights("classifier.h5")
print(classifier)
submission = data_reader('../input/sample_submission.csv')

submission['Images'] = submission['id_code'].astype(str)+'.png'
submission.head(5)
submission_datagen=ImageDataGenerator(rescale=1./255)

submission_gen=submission_datagen.flow_from_dataframe(

    dataframe=submission,

    directory="../input/test_images",

    x_col="Images",    

    batch_size=batch_size,

    shuffle=False,

    class_mode=None, 

    target_size=(256,256)

)
predictions=classifier.predict_generator(submission_gen, steps = len(submission_gen))

max_probability = np.argmax(predictions,axis=1) 

print(max_probability)
submission.drop(columns=['Images'], inplace= True)

submission['diagnosis'] = max_probability

submission.to_csv('submission.csv', index=False)
