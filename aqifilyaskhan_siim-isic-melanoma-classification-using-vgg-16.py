# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option("display.max_columns",None)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#heavily inspired by: https://www.kaggle.com/ibtesama/siim-baseline-keras-vgg16

base_tile_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
df['path']=base_tile_dir+df.image_name+".jpg"
df.head()
# df['path'].head()
# pd.DataFrame.to_string(df)
df_0=df[df['target']==0].sample(2000)
df_1=df[df['target']==1]
train=pd.concat([df_0,df_1])
train=train.reset_index()
train.head()

df=train
import cv2
img=cv2.imread("/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_0015719.jpg")
cv2.imshow("Output here",img)
# y = l_encoded
# y=df['target']
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df['path'],df['target'], test_size=0.2, random_state=1234)
# train_x, test_x, train_y, test_y = train_test_split(df['path'],y, test_size=0.15, shuffle=True)
test_x.head()
train_x=pd.DataFrame(train_x)
train_y=pd.DataFrame(train_y)
train_x.reset_index(drop=True, inplace=True)
train_y.reset_index(drop=True, inplace=True)

train=pd.concat([train_x,train_y],axis=1)
train.head()

test_x=pd.DataFrame(test_x)
test_y=pd.DataFrame(test_y)
test_x.reset_index(drop=True, inplace=True)
test_y.reset_index(drop=True, inplace=True)

test=pd.concat([test_x,test_y],axis=1)
test.head()
train['target']=train['target'].astype('str')
test['target']=test['target'].astype('str')
from keras.preprocessing.image import ImageDataGenerator
batch_size=64
colormode = 'rgb'
classmode = 'binary'
shuffle=True
seed=666
target_size=(224,224)
# target_size=(299,299)

datagen = ImageDataGenerator(rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True)
# train_set_path ="/content/train_images/"
validation=test
columns = [0,1]

training_generator = datagen.flow_from_dataframe(train,
                                           x_col='path',
                                          #  y_col=0,
                                           y_col='target',
                                           target_size = target_size,
                                           batch_size = batch_size,
                                           class_mode = classmode,
                                           color_mode=colormode,
                                           shuffle = shuffle,
                                           seed=seed)

validation_generator = datagen.flow_from_dataframe(validation,
                                             x_col='path',
                                             y_col='target',
                                             target_size = target_size,
                                             batch_size = batch_size,
                                             class_mode = classmode,
                                             color_mode=colormode,
                                             shuffle = shuffle,
                                             seed=seed)
from keras import layers
from keras.preprocessing import image
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import vgg19
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg16 import VGG16,preprocess_input

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
vgg = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)

for layer in vgg.layers[0:7]:
  layer.trainable = False
for layer in vgg.layers[8:]:
  layer.trainable=True  

# useful for getting number of classes
# folders = glob('/content/train_images/*')
  

# our layers - you can add more if you want
x = Flatten()(vgg.output)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(1, activation='sigmoid')(x)
# prediction = Dense(5005, activation='softmax')(x)
# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',
              metrics=['accuracy'])

# print(model.summary())
nb_epochs = 2
batch_size=16
# nb_train_steps = train.shape[0]//batch_size
# nb_val_steps=validation.shape[0]//batch_size

nb_train_steps=training_generator.n//training_generator.batch_size
nb_val_steps=validation_generator.n//validation_generator.batch_size

print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
history = model.fit_generator(
    training_generator,
    steps_per_epoch=nb_train_steps,
    epochs=nb_epochs,
    validation_data=validation_generator,
    # callbacks=cb,
    verbose=1,
    validation_steps=nb_val_steps)
