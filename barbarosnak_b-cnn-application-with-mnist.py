# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt







import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")

train.head()
test=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")



test.head()
print(train.iloc[15][0]) #this is for seeing which number we are picturing.

img=train.iloc[15][1:].to_numpy() #you can also use as_matrix(), but the last time that I wanted to use it I've got a problem. Because of that, I used to_numpy(). 

                                  #I didn't get rid of the label part yet. Fot just using pixels we started from the 2nd([1]) element of one column.

img=np.reshape(img,(28,28))

plt.imshow(img,cmap="gray")

plt.axis("off")

plt.show()
X_train=train.drop("label",axis=1)

X_train.head()
Y_train=train.label

Y_train.head()
plt.figure(figsize=(15,8))

sns.countplot(Y_train,palette="rocket")

plt.title("Count of Numbers")

plt.show()

X_train=X_train/255.0

X_train.shape
Y_train.shape
X_train=X_train.values.reshape(-1,28,28,1)

X_train.shape
from keras.utils.np_utils import to_categorical

Y_train=to_categorical(Y_train,num_classes=10)

Y_train.shape
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(X_train,Y_train,test_size=0.15,random_state=42)



print("x train shape: ",x_train.shape)

print("x val shape: ",x_val.shape)

print("y train shape: ",y_train.shape)

print("y val shape: ",y_val.shape)
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Activation

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(28,28,1)))

model.add(Activation("tanh"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(32,(3,3)))

model.add(Activation("tanh"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64,(3,3)))

model.add(Activation("tanh"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256,activation="tanh"))

model.add(Dropout(0.5))

model.add(Dense(10,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
datagen=ImageDataGenerator(rotation_range=0.5,zoom_range=0.5,horizontal_flip=True)

datagen.fit(x_train)
history=model.fit_generator(datagen.flow(x_train,y_train,batch_size=32),epochs=50,validation_data=(x_val,y_val), steps_per_epoch=x_train.shape[0] // 32)
plt.subplot(1,2,1)

plt.plot(history.history["val_loss"],color="r",label="validation loss")

plt.xlabel("number of ephocs")

plt.ylabel("val_loss")



plt.subplot(1,2,2)

plt.plot(history.history["val_accuracy"],color="r",label="validation accuracy")

plt.xlabel("number of ephocs")

plt.ylabel("val_accuracy")



plt.show()



plt.subplot(1,2,1)

plt.plot(history.history["loss"],color="r",label="loss")

plt.xlabel("number of ephocs")

plt.ylabel("loss")



plt.subplot(1,2,2)

plt.plot(history.history["accuracy"],color="r",label="accuracy")

plt.xlabel("number of ephocs")

plt.ylabel("accuracy")



plt.show()
Y_pred = model.predict(x_val)



Y_pred_classes = np.argmax(Y_pred,axis = 1) 



Y_true = np.argmax(y_val,axis = 1) 



confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 



f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
x_test=test.drop("label",axis=1)



x_test.head()
y_test=test.label



y_test.head()
x_test=x_test/255.0

x_test=x_test.values.reshape(-1,28,28,1)



x_test.shape
y_test=to_categorical(y_test,num_classes=10)



y_test.shape
Y_pred = model.predict(x_test)

 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 



Y_true = np.argmax(y_test,axis = 1) 



confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 



f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()