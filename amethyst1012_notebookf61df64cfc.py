# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import tensorflow as tf

from keras.utils.np_utils import to_categorical

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train.head()



test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test.head()



y=train['label'].values

x=train.drop('label',axis=1).values

print(x.shape)

print(y.shape)



import matplotlib.pyplot as plt

x=x.reshape(x.shape[0],28,28)

fig=plt.figure()

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(x[i],cmap='gray')



x = x.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



x.shape



x = x/255

test = test/255

y = to_categorical(y)



from sklearn.model_selection import train_test_split



train_x,valid_x,train_y,valid_y = train_test_split(x,y,random_state=1,test_size=0.2)



from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D,Dropout,GlobalAveragePooling2D,MaxPool2D,Flatten



model = Sequential()

model.add(Conv2D(64,kernel_size=(3,3,),padding='same',strides=(2,2),activation=tf.nn.relu6,input_shape=(28,28,1)))

model.add(tf.keras.layers.BatchNormalization(axis=-1))

model.add(MaxPool2D(strides=(2,2),padding='same'))

model.add(Conv2D(128,kernel_size=(3,3),strides=(2,2),padding='same',activation=tf.nn.relu6))

model.add(MaxPool2D())

model.add(Dropout(0.2))



model.add(GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu6))

model.add(tf.keras.layers.Dense(10,activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),metrics = ['accuracy'])



from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen=ImageDataGenerator(

    rescale=1/.255,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    fill_mode='nearest'

)

train_generator=train_datagen.flow(

    train_x,train_y,

    batch_size=128

    )



valid_datagen=ImageDataGenerator(

    rescale=1/.255)

valid_generator=valid_datagen.flow(

    valid_x,valid_y,

    batch_size=128)



import tensorflow as tf

callback=tf.keras.callbacks.EarlyStopping(monitor='accuracy',min_delta=0,patience=5,verbose=2,mode='auto',restore_best_weights=True)

history=model.fit(train_generator,

                  epochs=100,

                  steps_per_epoch=train_x.shape[0]//128,

                  verbose=2,

                  validation_data=valid_generator,

                  validation_steps=valid_x.shape[0]//128,

                  callbacks=[callback])

import matplotlib.pyplot as plt

def plot_model(history):

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))

    fig.suptitle('Model Accuracy and Loss')



    ax1.plot(history.history['accuracy'])

    ax1.plot(history.history['val_accuracy'])

    ax1.title.set_text('Accuracy')

    ax1.set_ylabel('Accuracy')

    ax1.set_xlabel('Epoch')

    ax1.legend(['Train','Valid'],loc=4)



    ax2.plot(history.history['loss'])

    ax2.plot(history.history['val_loss'])

    ax2.title.set_text('Loss')

    ax2.set_ylabel('Loss')

    ax2.set_xlabel('Epoch')

    ax2.legend(['Train','Valid'],loc=1)



    fig.show()



plot_model(history)
prediction=model.predict(test)

prediction=np.argmax(prediction,axis = 1)

prediction=pd.Series(prediction,name="Label")
zz