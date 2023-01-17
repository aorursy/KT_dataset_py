import numpy as np 

import pandas as pd 

from keras import optimizers

from keras import models

from keras import layers

from keras import losses

from keras.utils import to_categorical



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train_data.head(2)
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_data.head(2)
train_label = train_data['label']
train_data = train_data.drop('label',axis=1)
train_data /= 255

test_data /= 255
train_label_to_cat= to_categorical(train_label)

train_label_to_cat.shape
train_data.shape
train_data = train_data.values.reshape(-1,28,28,1)

test_data = test_data.values.reshape(-1,28,28,1)
train_data.shape
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPool2D((2,2)))
model.summary()
model.add(layers.Flatten())

model.add(layers.Dense(1024,activation='relu'))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer=optimizers.RMSprop(lr=0.001),

             loss= losses.categorical_crossentropy,

             metrics=['accuracy'])
model.fit(train_data,train_label_to_cat,epochs=60)
predict = model.predict_classes(test_data)
test_data_DF = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
output = pd.DataFrame({'ImageId': test_data_DF.index+1,

                       'Label': predict})

output.to_csv('submission.csv', index=False)