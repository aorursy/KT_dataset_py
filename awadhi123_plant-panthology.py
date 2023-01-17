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
train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
train.head()
train["image_id"] = train["image_id"] + ".jpg"
test["image_id"] = test["image_id"] + ".jpg"
train_label = train[["healthy","multiple_diseases","rust","scab"]]
train_label
train =  train.drop(columns={"healthy","multiple_diseases","rust","scab"})
train.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,train_label,test_size=0.2,random_state = 2)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.layers import GlobalAveragePooling2D
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( 
            rotation_range=45,
            brightness_range=[0.5, 2],
            shear_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=.1,
            fill_mode='nearest',
            rescale=1/255.,
          
    )
val_test_datagen = ImageDataGenerator( 
           
            rescale=1/255.,
         
    )

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import BatchNormalization

# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
baseModel = InceptionV3(include_top = False , weights = "imagenet" , input_shape = (300,300,3))
baseModel.summary()
for layer in baseModel.layers:
    layer.trainable = False

middle_layer = baseModel.get_layer('mixed7')
print('middle pretrained layer output shape: ', middle_layer.output_shape)
middle_output = middle_layer.output


x = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu')(middle_output)
x = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',   activation ='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)


x= Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu')(x)
x = Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',  activation ='relu')(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
x = Dropout(0.25)(x)


x = Flatten()(x)
x = Dense(256, activation = "relu")(x)
x = Dropout(0.5)(x)
x = Dense(4, activation = "softmax")(x)

      
model = Model( inputs = baseModel.input, outputs = x) 
from keras.optimizers import RMSprop
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import ReduceLROnPlateau

train = pd.concat([X_train,y_train],axis=1)
train
val = pd.concat([X_test,y_test],axis=1)
val
train_generator=train_datagen.flow_from_dataframe(
    
            dataframe=train,
            directory="/kaggle/input/plant-pathology-2020-fgvc7/images/",
            x_col="image_id",
            y_col=["healthy","multiple_diseases","rust","scab"],
            batch_size=104,   #large batch size can help effieciently on large dataset
            shuffle=True,
            class_mode="raw", 
            target_size=(300,300)
            
    )
val_generator=val_test_datagen.flow_from_dataframe(
    
            dataframe=val,
            directory="/kaggle/input/plant-pathology-2020-fgvc7/images/",
            x_col="image_id",
            y_col=["healthy","multiple_diseases","rust","scab"],   
            batch_size=73,
            shuffle=True,
            class_mode="raw",
            target_size=(300,300)
    
    )
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = val_generator.n//val_generator.batch_size
STEP_SIZE_TRAIN,STEP_SIZE_VALID
fitted = model.fit_generator(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs = 100,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VALID,
        callbacks=[reduce_lr]
)
test_generator=val_test_datagen.flow_from_dataframe(
    
            dataframe=test,
            directory="/kaggle/input/plant-pathology-2020-fgvc7/images/",
            x_col="image_id",
            y_col=None,
           # batch_size=,
           # seed=42,
            shuffle=False,
            class_mode=None,
            target_size=(300,300)
        
    )
a = model.predict(test_generator)
sub = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")
sub.loc[:, 'healthy':] =a

sub.to_csv('submissiond.csv', index=False)
sub.head()


