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
Train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

Test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

x_t=Train[Train.columns[Train.columns!='label']]

x_t=x_t/255

y_t=Train['label']

x_test=Test

x_test=x_test/255
import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten

from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid=train_test_split(x_t,y_t,test_size=0.1,shuffle=True)
x_test=np.array(x_test)

x_train=np.array(x_train)

x_valid=np.array(x_valid)
x_test=x_test.reshape((x_test.shape[0], 28, 28, 1))

x_train=x_train.reshape((x_train.shape[0], 28, 28, 1))

x_valid=x_valid.reshape((x_valid.shape[0], 28, 28, 1))
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

model.add(Dense(10, activation='softmax'))



model.compile(

    loss='sparse_categorical_crossentropy',

    optimizer=tf.keras.optimizers.Adam(0.001),

    metrics=['accuracy'],

)



model.fit(

    x_train,y_train,

    epochs=3,

    validation_data=(x_valid,y_valid),

)
# X_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

a=np.argmax(model.predict(x_test),axis=1)
x_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

x_test['Label']=a


x_test['ImageId']=x_test.index+1

x_test[['ImageId','Label']].to_csv('submission.csv',index=False)
pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')