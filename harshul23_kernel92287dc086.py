# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten,Conv2D,MaxPool2D
from keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
df = pd.read_csv("../input/train.csv")
print(df.shape)
x_train=df.iloc[:,1:].values.astype('float32')
y_train=df.iloc[:,0].values.astype('int32')
print(y_train.shape)
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df = pd.read_csv("../input/test.csv")
print(df.shape)
x_train=x_train/255.0

x_train = x_train.reshape(-1,28,28,1)

from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes
print(x_train.shape) # 2

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        )  # randomly flip images


datagen.fit(x_train)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization

model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])




model.fit_generator(datagen.flow(x_train,y_train, batch_size=2048),
                              epochs = 10, validation_data = (x_val,y_val),
                               steps_per_epoch=100
                              )


x_test=df.iloc[:,:].values.astype('float32')

x_test=x_test/255.0
x_test = x_test.reshape(-1,28,28,1)

y_pred=model.predict(x_test)




predicted_classes = model.predict_classes(x_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),
                         "Label": predicted_classes})
submissions.to_csv("asd2.csv", index=False, header=True)


