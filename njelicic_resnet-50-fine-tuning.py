# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.applications import ResNet50

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Dropout, BatchNormalization

from keras.models import Model 

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import regularizers

from keras import optimizers

from keras import backend as K 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(len(os.listdir("../input/celeba-dataset/img_align_celeba/img_align_celeba")))



# Any results you write to the current directory are saved as output.

df_path = "../input/celeba-dataset/list_attr_celeba.csv"

df_train = pd.read_csv(df_path)
df_train = df_train.rename(columns={'image_id':'filename','Male':'class'})

df_train = df_train.replace(1,'Male')

df_train = df_train.replace(-1,'Female')

df_train.head()
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=False

)





train_generator = train_datagen.flow_from_dataframe(

    directory ='../input/celeba-dataset/img_align_celeba/img_align_celeba/',

    dataframe=df_train ,

    X_col ='filename' ,

    Y_col = 'class',

    batch_size=64,

    class_mode='binary'

)



step_train = train_generator.n//train_generator.batch_size




K.clear_session()



base_model = ResNet50(weights="../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False)



for layer in base_model.layers:

    layer.trainable = True



x = base_model.output

x = GlobalAveragePooling2D()(x)

predictions = Dense(1, activation = 'sigmoid',activity_regularizer=regularizers.l1(0.01)) (x)







model = Model(inputs=base_model.input, outputs=predictions)





optim = optimizers.SGD(lr=0.001)

model.compile(optimizer=optim, loss='binary_crossentropy',metrics=['accuracy'])


model.fit_generator(

    train_generator,

    steps_per_epoch=step_train,

    epochs=1,

)
model.save('resnet_face.hdf5')
print(os.listdir("../input"))