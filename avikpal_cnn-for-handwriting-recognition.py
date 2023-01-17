# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
X_train = np.asarray(train.ix[:,1:].values,dtype=np.float32)

Y_train = np.asarray(train.ix[:,0].values,dtype=np.int32)

X_test = np.asarray(test.values,dtype=np.float32)
X_train
ins = np.shape(X_train)[0]

print(np.shape(X_train))
Y_train
import matplotlib.pyplot as plt

%matplotlib inline
X_train = np.reshape(X_train,(ins,28,28))



for i in range(1,10):

    plt.subplot(330+i)

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(Y_train[i])
X_train = np.reshape(X_train,(ins,28,28,1))
X_test = np.reshape(X_test,(np.shape(X_test)[0],28,28,1))
from keras.utils.np_utils import to_categorical,normalize
Y_train = to_categorical(Y_train)

Y_train
seed = 124

np.random.seed = seed
from keras.models import Sequential

from keras.layers.core import Dropout,Lambda,Flatten,Activation,Dense

from keras.layers import Conv2D,MaxPool2D,AveragePooling2D,BatchNormalization

from keras import optimizers
model = Sequential()

model.add(Conv2D(32,(5,5),strides=(2,2),padding="same",input_shape=(28,28,1)))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2),padding="same"))

model.add(Conv2D(64,(5,5),strides=(2,2),padding="same"))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2),padding="same"))

model.add(Conv2D(128,(5,5),strides=(2,2),padding="same"))

model.add(Activation('relu'))

model.add(AveragePooling2D(pool_size=(2,2),padding="same"))
model.add(Flatten())

#model.add(Dense(1500,activation="relu"))

#model.add(BatchNormalization())

#model.add(Dropout(0.5))

model.add(Dense(256,activation="relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(128,activation="relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10,activation="softmax"))
print(model.input_shape)

print(model.output_shape)
model.compile(optimizer=optimizers.Nadam(),loss="categorical_crossentropy",metrics=['accuracy'])
from keras.preprocessing import image

gen = image.ImageDataGenerator()
from sklearn.model_selection import train_test_split

X_train1, X_val, Y_train1, Y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=124)

batches = gen.flow(X_train1, Y_train1, batch_size=50)

val_batches=gen.flow(X_val, Y_val, batch_size=50)
history=model.fit_generator(batches, batches.n, nb_epoch=1,validation_data=val_batches, nb_val_samples=val_batches.n)
hist = history.history

print(hist.keys)
print(hist['acc'])

print(hist['val_acc'])
predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)