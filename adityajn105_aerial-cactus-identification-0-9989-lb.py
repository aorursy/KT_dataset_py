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
import glob

import pandas as pd

import numpy as np

data = pd.read_csv('../input/train.csv')

data['path'] = '../input/train/train/'

data['path'] = data[['path','id']].apply(lambda x: "".join(x), axis=1)

data.drop(['id'],axis=1,inplace=True)

data.head()
all_images = glob.glob('../input/train/train/*')

print(len(all_images), data.shape)



IMG_DIM = (30,30,3)



from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator

imgs = np.array([ img_to_array( load_img(file, target_size = IMG_DIM) ) for file in data.path.values ])

labels = data.has_cactus.values





print(imgs.shape,labels.shape)
from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val = train_test_split(imgs,labels,test_size=0.3, stratify=labels)



train_datagen = ImageDataGenerator(rescale=1./255, 

                                   zoom_range=0.3, 

                                   rotation_range=50,

                                   width_shift_range=0.2, 

                                   height_shift_range=0.2, 

                                   shear_range=0.2, 

                                   horizontal_flip=True, 

                                   fill_mode='nearest')



val_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)

val_generator = val_datagen.flow(X_val, Y_val, batch_size=32)
from keras.layers import Input,Conv2D,Dense,Dropout, MaxPooling2D, Flatten

from keras import optimizers

from keras.models import Model



#Input Layer

inp = Input(IMG_DIM)



#1st Conv

conv_1  = Conv2D( 64, kernel_size=(2,2), activation='relu')(inp)

pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)



#2nd Conv

conv_2 = Conv2D( 32, kernel_size=(2,2), activation='relu')(pool_1)

pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)



#flatten

flatten = Flatten()(pool_2)

dropout_1 = Dropout(0.3)(flatten)



#1st Dense

dense_1 = Dense(512, activation='relu')(dropout_1)

dropout_2 = Dropout(0.3)(dense_1)



#2nd Dense

dense_2 = Dense(64,activation='relu')(dropout_2)

dropout_3 = Dropout(0.2)(dense_2)



#output

output = Dense(1, activation='sigmoid')(dropout_3)



model = Model(inp,output)



model.compile( loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'] )



model.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping



model_checkpoint  = ModelCheckpoint('model_best_checkpoint.h5', save_best_only=True, monitor='val_acc', mode='max', verbose=2)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min')



callback_list = [model_checkpoint]



history = model.fit_generator(train_generator, steps_per_epoch=150, epochs=400,

                              validation_data=val_generator, validation_steps=50, 

                              verbose=1,callbacks=callback_list)
tests =  glob.glob('../input/test/test/*')

imgs = np.array([ img_to_array( load_img(file, target_size = IMG_DIM) ) for file in tests ])/255

submission = pd.DataFrame({'id':tests})

submission.id = submission.id.apply(lambda x: x.split('/')[-1])

submission.head()
submission['has_cactus'] = np.squeeze(model.predict(imgs))

submission.to_csv('samplesubmission.csv',index=False)

submission.head()