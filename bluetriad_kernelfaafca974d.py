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
import pandas as ps

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

train_data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

X=train_data.drop('label',axis=1)

y=train_data['label']

X=X/255.0

test_data=test_data/255.0

X=X.values.reshape(-1,28,28,1)

test_data=test_data.values.reshape(-1,28,28,1)

from tensorflow.keras.utils import to_categorical

y_train=to_categorical(y,num_classes=10)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.1, random_state=42)

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D





model = Sequential()



model.add(Conv2D(filters = 32,

                 kernel_size = (5,5),

                 padding = 'Same', 

                 activation ='relu',

                 input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.5))





model.add(Conv2D(filters = 64,

                 kernel_size = (3,3),

                 padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64,

                 kernel_size = (3,3),

                 padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.5))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))





model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping

early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=2)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen=ImageDataGenerator(rotation_range=10,

                            width_shift_range=0.1,

                            height_shift_range=0.1,

                            shear_range=0.1,

                            zoom_range=0.15,

                            horizontal_flip=False,

                            vertical_flip=False, 

                            fill_mode='nearest')

image_gen.fit(X_train)

history = model.fit_generator(image_gen.flow(X_train,y_train, batch_size=16),

                              epochs = 8, validation_data = (X_test,y_test),

                              callbacks=[early_stop])

results = model.predict(test_data)

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("digit_recognizer.csv",index=False)
