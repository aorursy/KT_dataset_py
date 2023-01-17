# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load







import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, CSVLogger



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img_size = 28

img_width, img_ht = img_size, img_size

verbosity = 1

batch_size = 512

classes = 10

channels = 1

validation_ratio = 0.1



train_set = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_set = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



y_train = train_set["label"].values

x_train = train_set.drop(labels=["label"], axis=1).values



x_train = x_train.reshape(-1, 28, 28, 1)

test_set = test_set.values.reshape(-1, 28, 28, 1)







# x_train = x_train.values.reshape(-1, 28, 28, 1)

# test_set = test_set.values.reshape(-1, 28, 28, 1)



# # One-Hot encoding

y_train = to_categorical(y_train, num_classes=10)


from sklearn.model_selection import train_test_split

x_training, x_validation, y_training, y_validation = train_test_split(x_train,

                                                                      y_train,

                                                                      test_size=validation_ratio,

                                                                      shuffle = True)
input_shape=(28,28,1)

model=Sequential()

model.add(Conv2D(filters=32,

                kernel_size=(5,5),

               padding='Same',

               activation='relu',

               input_shape=input_shape))



model.add(Conv2D(filters=32,

                kernel_size=(5,5),

               padding='Same',

               activation='relu',

               input_shape=input_shape))



model.add(Conv2D(filters=32,

                kernel_size=(5,5),

               padding='Same',

               activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.5))





model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 

                 activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 

                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(8192, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(2048, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(10, activation="softmax"))



model.summary()
model.compile(optimizer=RMSprop(lr=0.0001,

                                rho=0.9,

                                epsilon=1e-08,

                                decay=0.00001),

              loss="categorical_crossentropy",

              metrics=["accuracy"])



data_generator = ImageDataGenerator(rescale=1./255,

                                    rotation_range=1,

                                    zoom_range=0.1, 

                                    width_shift_range=0.05,

                                    height_shift_range=0.05)

data_generator.fit(x_training)



history = model.fit_generator(data_generator.flow(x_training,

                                                  y_training,

                                                  batch_size=512),

                              epochs=10,

                              validation_data=(x_validation, y_validation),

                              verbose=1,

                              steps_per_epoch=x_training.shape[0]// 32,

                             )





test_x=test_set[:,:]

test_y=model.predict(test_x)

# print(test_y)

# y_test_y=test_y.argmax(axis=1)

# print(y_test_y)

ids=[]

for x in range(len(test_set)):

    ids.append(x)

    

# print(ids)

mysubmission=pd.DataFrame({"ImageId": ids,"Label":test_y.argmax(axis=-1)})

mysubmission.head()
mysubmission.to_csv('submission.csv', index=False)

# kaggle competitions submit -c digit-recognizer -f submission.csv -m "Message"