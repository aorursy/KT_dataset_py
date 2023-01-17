# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

print(train_df.info())
train_df['diagnosis'] = train_df['diagnosis'].astype('str')

train_df['id_code'] = train_df['id_code'].astype(str)+'.png'

from keras.preprocessing.image import ImageDataGenerator



datagen=ImageDataGenerator(

    rescale=1./255, 

    validation_split=0.2)



batch_size = 32



train_gen=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    target_size=(224,224),

    subset='training')



test_gen=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical", 

    target_size=(224,224),

    subset='validation')
y_train = train_df['diagnosis']

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]
from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K
model = Sequential()

model.add(Conv2D(64, (3, 3), padding="same",input_shape=(224,224,3)))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))



model.add(MaxPooling2D(pool_size=(3, 3),strides=2))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(MaxPooling2D(pool_size=(3, 3),strides=2))

model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(256, (3, 3), padding="valid"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))



model.add(Conv2D(256, (3, 3), padding="valid"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(256, (3, 3), padding="valid"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(MaxPooling2D(pool_size=(3, 3),strides=2))

model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(512, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(512, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(512, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(MaxPooling2D(pool_size=(3,3),strides=2))

model.add(Dropout(0.25))



model.add(Conv2D(512, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(512, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(512, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(512, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=-1))

model.add(MaxPooling2D(pool_size=(3,3),strides=2))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(1024))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint

es= EarlyStopping(monitor='val_loss', mode ='min', verbose = 0, patience = 5)

mc = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only = True, mode ='min', verbose = 0)
model.fit_generator(generator=train_gen,              

                                    steps_per_epoch=len(train_gen),

                                    validation_data=test_gen,                    

                                    validation_steps=len(test_gen),

                                    epochs=10,

                                    callbacks = [es, mc], 

                                    use_multiprocessing = True,

                                    verbose=1)
submission_df = pd.read_csv('../input/sample_submission.csv')

#submission_df['diagnosis'] = submission_df['diagnosis'].astype('str')

submission_df['id_code'] = submission_df['id_code'].astype(str)+'.png'

submission_datagen=ImageDataGenerator(rescale=1./255)

submission_gen=submission_datagen.flow_from_dataframe(

    dataframe=submission_df,

    directory="../input/test_images",

    x_col="id_code",    

    batch_size=batch_size,

    shuffle=False,

    class_mode=None, 

    target_size=(224,224)

)
predictions=model.predict_generator(submission_gen, steps = len(submission_gen))
max_probability = np.argmax(predictions,axis=1) 

submission_df1 = pd.read_csv('../input/sample_submission.csv')
submission_df['diagnosis'] = max_probability

submission_df.drop('id_code',axis=1)

submission_df['id_code']=submission_df1['id_code']

submission_df.to_csv('submission.csv', index=False)