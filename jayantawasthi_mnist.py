# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization

from tensorflow.keras.optimizers import RMSprop,Adam

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
label=train["label"].values
train.drop("label",axis=1,inplace=True)
train
t=train.values

ttest=test.values
t
t=t.astype('float32')

ttest=ttest.astype('float32')
t/=255

ttest/=255
from tensorflow import keras

tl = keras.utils.to_categorical(label, 10)

t= t.reshape(t.shape[0], 28, 28, 1)

ttest= ttest.reshape(ttest.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

BatchNormalization(),

model.add(MaxPooling2D(pool_size=(2, 2)))

Conv2D(64, (3, 3), activation='relu'),

Conv2D(64, (3, 3), activation='relu'),

BatchNormalization(),

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))





model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
np.random.seed(1234)



(x_train,x_test,y_train,y_test) = train_test_split(t,tl, train_size=0.75, random_state=1)
model.fit(x_train, y_train,

                    batch_size=100,

                    epochs=400,

                    verbose=2,

                    validation_data=(x_test, y_test))
y_pred=model.predict(ttest,verbose=0)
sample=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sample.head()
y_pred
pred = np.argmax(y_pred, axis = 1)
r = pd.Series(pred,name="Label")
r
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),r],axis = 1)
submission
submission.to_csv("mnist4.csv",index=False)