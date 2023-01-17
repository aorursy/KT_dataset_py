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
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.layers import Conv2D,Input,MaxPooling2D,Flatten,Dense
from keras.models import Sequential
import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Before: \n",x_train.shape,y_train.shape)
#scaling
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

#adding channel dimension
x_train = tf.expand_dims(x_train,-1)
x_test = tf.expand_dims(x_test,-1)

#one hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#
print("After: \ n",x_train.shape,y_train.shape,x_test.shape,y_test.shape)
model = Sequential()
model.add(Input(shape = (28,28,1)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(10,activation='softmax'))
batch_size = 128
epochs = 20
model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))
pdf = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
npdf = pdf.to_numpy()
s = npdf.shape
x_pred = tf.expand_dims(npdf.reshape((s[0],28,28)),-1)
print(x_pred.shape)
y_pred = model.predict(x_pred)
y_label = tf.argmax(y_pred,axis=1)
print(y_label.shape)
out_df = pd.DataFrame(y_label)
out_df['ImageId'] = out_df.index + 1
out_df['Label'] = out_df[0]
out_df.drop(columns = [0],inplace = True)
out_df.head()
out_df.to_csv("/kaggle/working/out.csv")
