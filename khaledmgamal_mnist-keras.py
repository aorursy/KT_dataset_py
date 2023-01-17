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
!pip install tensorflow==1.13.1
import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten,Conv2D,MaxPooling2D,Activation,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam ,RMSprop

from keras.models import load_model



def read_csv(file_path):

    data=pd.read_csv(file_path)

    return data



train_data=read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data=read_csv('/kaggle/input/digit-recognizer/test.csv')



def pre_process(data,is_test):

    if is_test==False:

       y=data['label']

       y_one_hot=pd.get_dummies(y,prefix='label')

       X_train=data.drop('label',axis=1)

       X_train_np=np.array(X_train)

       X_train_np=X_train_np.reshape([X_train_np.shape[0],28,28,1])

       X_train_np_normalized=X_train_np/255

    

       return X_train_np_normalized,y_one_hot

    if is_test==True:

       X_train=data

       y_one_hot=0

       X_train_np=np.array(X_train)

       X_train_np=X_train_np.reshape([X_train_np.shape[0],28,28,1])

       X_train_np_normalized=X_train_np/255

    

       return X_train_np_normalized,y_one_hot



def create_CNN_model():

    model=Sequential()

    model.add(Conv2D(filters=64, kernel_size=2, strides=(1, 1), padding='same',input_shape=(28, 28,1)))

    model.add(BatchNormalization(axis=1))

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(filters=32, kernel_size=2, strides=(1, 1), padding='same'))

    model.add(BatchNormalization(axis=1))

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(filters=16, kernel_size=2, strides=(1, 1), padding='same'))

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Flatten())

    model.add(Dense(512))

    model.add(Activation('relu'))

    model.add(Dense(10))

    model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



    return model



X_train,y_train=pre_process(train_data,False)



epochs=10

gen =ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,

                               height_shift_range=0.08, zoom_range=0.08)

gen.fit(X_train)

model=create_CNN_model()

for e in range(epochs):

    print('Epoch', e)

    batches = 0

    for x_batch, y_batch in gen.flow(X_train, y_train, batch_size=32):

        model.fit(x_batch, y_batch)

        batches += 1

        if batches >= len(X_train) / 32:

            # we need to break the loop by hand because

            # the generator loops indefinitely

            break



#model.fit(X_train, y_train, epochs=10, batch_size=32)





X_test,y_test=pre_process(test_data,True)



y_predict = model.predict(X_test)



minst=pd.read_csv('digit-recognizer/sample_submission.csv')

minst=minst.drop('Label',axis=1)

minst['Label']=np.argmax(y_predict,axis=1)

minst.to_csv('sample_submission02.csv',index=None)