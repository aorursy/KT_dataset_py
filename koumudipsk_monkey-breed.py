# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.applications import MobileNet

img_rows=224

img_cols=224

MobileNet=MobileNet(include_top=False, weights='imagenet',input_shape=(img_rows,img_cols,3))

for layer in MobileNet.layers:

    layer.trainable=False

for (i,layer) in enumerate(MobileNet.layers):

    print(str(i)+" "+layer.__class__.__name__,layer.trainable)
def addModel(my_model,num_classes):

    top_model = my_model.output

    top_model = GlobalAveragePooling2D()(top_model)

    top_model = Dense(1024,activation='relu')(top_model)

    top_model = Dense(1024,activation='relu')(top_model)

    top_model = Dense(512,activation='relu')(top_model)

    top_model = Dense(num_classes,activation='softmax')(top_model)

    return top_model
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D,GlobalAveragePooling2D,Dense,Flatten,Dropout,Activation

from keras.models import Sequential, Model

from keras.optimizers import SGD, RMSprop, Adam

from keras import backend as K
num_classes=10

layers_bottom=addModel(MobileNet,num_classes)

model=Model(inputs=MobileNet.input, outputs=layers_bottom)

train_data='../input/10-monkey-species/training/training'

test_data='../input/10-monkey-species/validation/validation'
train_datagen=ImageDataGenerator(rescale=1./255,width_shift_range=0.3,height_shift_range=0.3,

                                horizontal_flip=True,rotation_range=45, fill_mode='nearest')
train=train_datagen.flow_from_directory(train_data,target_size=(224, 224), batch_size=32,

        class_mode='categorical')
test_datagen=ImageDataGenerator(rescale=1./255)
test=test_datagen.flow_from_directory(test_data,target_size=(224, 224), batch_size=32,

        class_mode='categorical')
from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, EarlyStopping

earlystopping=EarlyStopping(monitor='val_loss',min_delta=0,patience=3,verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
callbacks=[earlystopping]

n_samples=1098

nv_samples=272

batch_size=16

epochs=5

model.fit_generator(train,

        steps_per_epoch=n_samples // batch_size,

        epochs=epochs,

        callbacks=callbacks,

        validation_data=test,

        validation_steps=nv_samples // batch_size)



Y_pred = model.predict_generator(test, nv_samples // batch_size+1)

y_pred = np.argmax(Y_pred, axis=1)

y_pred