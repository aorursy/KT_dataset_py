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
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
X_train = train.drop("label",axis=1)

X_train = X_train.values.reshape(-1,28,28,1)

X_test = test.values.reshape(-1,28,28,1)

print(X_train.shape)

print(X_test.shape)
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



X_train /= 255

X_test /= 255
from tensorflow.keras.utils import to_categorical

y_train = train.pop("label")

y_train = y_train.values

print(y_train.shape)

y_train = to_categorical(y_train, num_classes=10)

print(y_train.shape)

print(y_train[0])
import tensorflow as tf
def squeeze_excite_block(filters,input):                      

    se = tf.keras.layers.GlobalAveragePooling2D()(input)

    se = tf.keras.layers.Reshape((1, filters))(se) 

    se = tf.keras.layers.Dense(filters//16, activation='relu')(se)

    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)

    se = tf.keras.layers.multiply([input, se])

    return se
def make_model():

        s = tf.keras.Input(shape=X_train.shape[1:]) 

        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(s)

        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = squeeze_excite_block(32,x)



        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = squeeze_excite_block(32,x)

        x = tf.keras.layers.AveragePooling2D(2)(x)



        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = squeeze_excite_block(32,x)

        x = tf.keras.layers.AveragePooling2D(2)(x)        





        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = squeeze_excite_block(32,x)

        x = tf.keras.layers.AveragePooling2D(2)(x)





        x = tf.keras.layers.concatenate([tf.keras.layers.GlobalMaxPooling2D()(x),

                                         tf.keras.layers.GlobalAveragePooling2D()(x)])



        x = tf.keras.layers.Dense(10,activation='softmax',use_bias=False,

                                  kernel_regularizer=tf.keras.regularizers.l1(0.00025))(x)

        return tf.keras.Model(inputs=s, outputs=x)
from tensorflow.keras import optimizers

model=make_model()                

model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
model.fit(x=X_train, y=y_train, batch_size=32, epochs=15)
pred = model.predict(X_test,verbose=1)

predictions = pred.argmax(axis=-1)
sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission = sub['ImageId']
pred = pd.DataFrame(data=predictions ,columns=["Label"])

DT = pd.merge(submission , pred, on=None, left_index= True,

    right_index=True)

DT.head()
DT.to_csv('submission.csv',index = False)