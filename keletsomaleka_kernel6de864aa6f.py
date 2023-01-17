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
from sklearn.model_selection import train_test_split

from tensorflow.python import keras

import tensorflow as tf

from tensorflow.python.keras import Sequential

from tensorflow.python.keras.layers import Conv2D,Dense,Flatten,Dropout
raw_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

raw_data.shape
raw_data['label'].unique()
img_rows, img_cols = 28,28

num_classes = 10
def data_prep(data):

    

    out_y = tf.keras.utils.to_categorical(data.label,num_classes)

    num_imgs = data.shape[0]

    x_as_array = data.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_imgs,img_rows, img_cols,1)

    out_x = x_shaped_array/255

    return out_x,out_y
X,y = data_prep(raw_data)



model = Sequential()

model.add(Conv2D(20,

                activation = 'relu',

                 kernel_size=(3,3),

                input_shape=(img_rows, img_cols,1)))

model.add(Conv2D(20,

                activation='relu',

                 kernel_size=(3,3)))

model.add(Conv2D(20,

                activation='relu',strides=2,

                 kernel_size=(3,3)))

model.add(Flatten())

model.add(Dense(128, activation= 'relu'))

model.add(Dense(num_classes, activation = 'softmax'))



model.compile(loss = keras.losses.categorical_crossentropy,

             optimizer= 'adam',

             metrics= ['accuracy'])



model.fit(X,y, batch_size=128,

         epochs=4,

         validation_split=0.2)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test.shape
x_test = test.values

num_test_imgs = test.shape[0]

test_arrays = x_test.reshape(num_test_imgs,img_rows,img_cols,1)

test_x = test_arrays/255
pred = model.predict(test_x)
results = np.argmax(pred,axis = 1) 
results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission_best.csv",index=False)