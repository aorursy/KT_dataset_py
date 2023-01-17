import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import Adam ,RMSprop



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/digit-recognizer/train.csv")

test= pd.read_csv("../input/digit-recognizer/test.csv")

test.head()
y_train = train['label']

x_train = train.drop("label",axis = 1)
x_train = x_train / 255.0

test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
y_train = to_categorical(y_train, num_classes = 10)
#spliting data 90% training 10% testing

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)
g = plt.imshow(x_train[2][:,:,0],cmap=plt.cm.binary)
model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

# Compile the model

model.compile(optimizer = 'adam' , 

              loss = "categorical_crossentropy", 

              metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
model.fit(x_train, y_train, batch_size = 40, epochs = 10, 

          validation_data = (x_val, y_val), verbose = 1)
val_loss, val_acc = model.evaluate(x_val, y_val)

print("Loss=", val_loss)  # model's loss (error)

print("Accuracy", val_acc)  # model's accuracy
pred = model.predict(test)

print(pred)
# predict results

reslt = model.predict(test)



# select the index with the maximum probability

reslt = np.argmax(reslt,axis = 1)



reslt = pd.Series(reslt,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),reslt],axis = 1)



submission.to_csv("cnn_mnist.csv",index=False)