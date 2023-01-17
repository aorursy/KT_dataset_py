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
from tensorflow.keras import  utils

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

x_train = train.drop(["label"], axis=1)
y_train = train["label"]
x_test=test
x_train = np.array(x_train)

x_test = np.array(x_test)

x_train=x_train.reshape(x_train.shape[0],28,28,1)

x_test=x_test.reshape(x_test.shape[0],28,28,1)
y_train = utils.to_categorical(y_train,10)
x_train = x_train / 255

x_test = x_test / 255
x_train, x_val, y_train , y_val=train_test_split(x_train,y_train,test_size=0.1)
datagen= ImageDataGenerator(

            zoom_range=0.3,

            rotation_range=0.2,

            width_shift_range=0.5,

            height_shift_range=0.5

            )
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape=(x_train.shape[1:]),padding="Same",activation="relu"))

model.add(Conv2D(filters=32,kernel_size=(5,5),padding="Same",activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(5,5),padding="Same",activation="relu"))

model.add(Conv2D(filters=64,kernel_size=(5,5),padding="Same",activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),))

model.add(Dropout(0.25))

model.add(Conv2D(filters=128,kernel_size=(5,5),padding="Same",activation="relu"))

model.add(Conv2D(filters=128,kernel_size=(5,5),padding="Same",activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10,activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())
checkpoint=ModelCheckpoint("mnistCNN",

                           monitor="val_acc",

                          save_best_only=True,

                          verbose=1)
lr_reduce=ReduceLROnPlateau(monitor="val_acc",

                            patience=4,

                            verbose=1,

                            factor=0.5,

                            min_lr=0.00000001

    )
model.fit(datagen.flow(x_train,y_train,batch_size=100),

         batch_size=1,

         epochs=40,

         validation_data=(x_val,y_val),

         verbose=1,

         callbacks=[checkpoint,lr_reduce])
model.load_weights("mnistCNN")
predictions = model.predict(x_test)
print(predictions.shape)
predictions=np.argmax(predictions, axis=1)
print(predictions.shape)
submission=pd.DataFrame({"ImageId":range(1,x_test.shape[0]+1),"Label": predictions})

submission = submission.to_csv("submission.csv",index=False)