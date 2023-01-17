# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# '/kaggle/input'
from keras.layers import Dense,Dropout,Input,MaxPooling2D,ZeroPadding2D,Conv2D,Flatten,BatchNormalization
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam,SGD
from keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator
from keras.utils import to_categorical

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_colwidth=150
df1=pd.read_csv('../input/dog-breed-identification/labels.csv')
df1.head()
# path of the dogs images
img_file='../input/dog-breed-identification/train/'
df=df1.assign(img_path=lambda x: img_file + x['id'] +'.jpg')
df.head()
img_pixel=np.array([img_to_array(dtype='int8',img=load_img(img, target_size=(128, 128))) for img in df['img_path'].values.tolist()])
sys.getsizeof(img_pixel)
img_label=df.breed
img_label=pd.get_dummies(df.breed).astype('int8')

X=img_pixel
y=img_label.values
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

del df
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow(X_train,y=y_train,batch_size=32)
testing_set=test_datagen.flow(X_test,y=y_test,batch_size=32)
model=Sequential()

model.add(Conv2D(32,input_shape=(128,128,3),kernel_size=(3,3),activation='relu',padding='same'))

model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(120,activation='softmax'))

model.summary()

model.compile(optimizer='Adam',
          loss='categorical_crossentropy', 
           metrics=['accuracy'])
history=model.fit_generator(training_set,
                      steps_per_epoch = X_train.shape[0]/32,
                      validation_data = testing_set,
                      validation_steps = 4,
                      epochs = 500,
                      verbose = 1)