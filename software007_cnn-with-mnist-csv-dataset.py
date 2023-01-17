# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense

from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sbs





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)



# Any results you write to the current directory are saved as output.
trainData=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")

testData=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")
trainData.head()
testData.head()
print("Train Data Shape: ",trainData.shape)

print("Test Data Shape: ",testData.shape)
Y_train=trainData["label"]

X_train=trainData.drop(labels=["label"],axis=1)



print(Y_train.shape)
Y_test=testData["label"]

X_test=testData.drop(labels=["label"],axis=1)
plt.figure(figsize=(15,7))

sbs.countplot(Y_train,palette="icefire")

plt.show()
plt.figure(figsize=(15,7))

sbs.countplot(Y_test,palette="icefire")

plt.show()
img=X_train.iloc[0].as_matrix()

img=img.reshape((28,28)) #28*28=784

plt.imshow(img,cmap="gray")

plt.axis("off")

plt.title(Y_train.iloc[0])

plt.show()
X_train=X_train/255.0

X_test=X_test/255.0



X_train=X_train.values.reshape(-1,28,28,1)

X_test=X_test.values.reshape(-1,28,28,1)



print("X Train Shape: ",X_train.shape)

print("X Test Shape: ",X_test.shape)
Y_train=to_categorical(Y_train,num_classes=10)

Y_test=to_categorical(Y_test,num_classes=10)



print(Y_train.shape)

from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(X_train,Y_train,test_size=0.33,random_state=0)
model=Sequential()



model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same",input_shape=(28,28,1)))

model.add(Activation("relu"))

model.add(MaxPooling2D())

model.add(Dropout(0.25))



model.add(Conv2D(32,(3,3),padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D())

model.add(Dropout(0.25))



model.add(Conv2D(64,(3,3),padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D())

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(units=512))

model.add(Activation("relu"))

model.add(Dropout(0.5))



model.add(Dense(512))

model.add(Activation("relu"))

model.add(Dropout(0.5))



model.add(Dense(10))

model.add(Activation("softmax"))





model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])



batch_size=32
datagenTrain=ImageDataGenerator(

                   shear_range=0.3,

                   horizontal_flip=True,

                   zoom_range=0.3)





datagenTrain.fit(x_train)



history=model.fit_generator(datagenTrain.flow(x_train,y_train,batch_size=batch_size),

                            epochs=10,validation_data=(x_val,y_val),

                            steps_per_epoch=x_train.shape[0]//batch_size)





plt.plot(history.history["loss"],label="Train Loss")

plt.plot(history.history["val_loss"],label="Validation Loss")

plt.legend()



plt.figure()

plt.plot(history.history["accuracy"],label="Train Accuracy")

plt.plot(history.history["val_accuracy"],label="Validation Accuracy")

plt.legend()