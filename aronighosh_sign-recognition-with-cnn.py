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
import csv

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

df_train=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

df_train.head()
df_test=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')

df_test.head()
Y=df_train['label']

X=df_train.drop(['label'],axis=1)

df_test=df_test.drop(['label'],axis=1)

X.head()
X=X/255

df_test=df_test/255

X.head()
X.shape,df_test.shape
X=X.values.reshape(-1,28,28,1)

df_test=df_test.values.reshape(-1,28,28,1)

X.shape,df_test.shape
from keras.utils.np_utils import to_categorical

Y = to_categorical(Y, num_classes = 26)

Y = Y.astype("int8")
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

model = tf.keras.models.Sequential([



    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.Dense(26, activation='softmax')

    ])



# Compile Model. 

model.compile(loss = 'categorical_crossentropy',

              optimizer='rmsprop', metrics=['acc'])
train_datagen = ImageDataGenerator(

    rescale=1/255,

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    rotation_range=40,

    horizontal_flip=True,

    fill_mode='nearest'

    )





train_datagen.fit(X)
model.fit(x_train, y_train, batch_size=26, epochs=10)
test_predictions = model.predict(df_test)





results = np.argmax(test_predictions,axis = 1) 



results = pd.Series(results,name="Label")





submission = pd.concat([pd.Series(range(1,7173),name = "ImageId"),results],axis = 1)



submission.to_csv("sign_recognition.csv",index=False)