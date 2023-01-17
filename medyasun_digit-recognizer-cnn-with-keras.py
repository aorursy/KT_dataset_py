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
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

print(train.shape)

train.head()
test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(test.shape)

test.head()
Y_train=train["label"]

X_train=train.drop("label",axis=1)
plt.figure(figsize=(15,7))

sns.countplot(Y_train,palette="icefire")

plt.title("Number Of Digit Counts")

plt.show()
img=X_train.iloc[0].to_numpy()

img=img.reshape((28,28))

plt.imshow(img,cmap="gray")

plt.show()
img=X_train.iloc[3].to_numpy()

img=img.reshape((28,28))

plt.imshow(img,cmap="gray")

plt.title(X_train.iloc[3,0])

plt.axis("off")

plt.show()
X_train=X_train/255

test=test/255
X_train=X_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)

test.shape
from keras.utils.np_utils import to_categorical

Y_train=to_categorical(Y_train,num_classes=10)

Y_train
from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.1,random_state=2)
from sklearn.metrics import confusion_matrix

import itertools

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))
optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

epochs=200

batch_size=64
datagen= ImageDataGenerator(

        featurewise_center=False,

        samplewise_center=False,

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,

        zca_whitening=False,

        rotation_range=0.5,

        zoom_range=0.5,

        width_shift_range=0.5,

        height_shift_range=0.5,

        horizontal_flip=False,

        vertical_flip=False)

datagen.fit(X_train)
history=model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),epochs=epochs,

                                        validation_data=(X_val,Y_val),steps_per_epoch=X_train.shape[0] // batch_size)
plt.plot(history.history['val_loss'],color='b',label='validation_loss')

plt.title('Test Loss')

plt.xlabel('Number Of Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
Y_pred=model.predict(X_val)

Y_pred_classes=np.argmax(Y_pred,axis=1)

Y_true=np.argmax(Y_val,axis=1)

confusion_mtx=confusion_matrix(Y_true,Y_pred_classes)

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(confusion_mtx,annot=True,linewidths=0.01,cmap='Greens',linecolor='gray',fmt='.1f',ax=ax)

plt.xlabel('Predicted Label')

plt.ylabel('True Label')

plt.title('Confusion Matrix')

plt.show()
results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission_digit_recognizer.csv",index=False)