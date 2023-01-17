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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
data=pd.read_csv('../input/Kannada-MNIST/train.csv')


X_train= data.values[:6000, 1:]
y_train=data.values[:6000, 0]
n_classes=10
y_train = to_categorical(y_train, n_classes) 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

from tensorflow.keras.preprocessing.image import ImageDataGenerator # for data augmentation

datagen = ImageDataGenerator(zoom_range=0.1,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             rotation_range=50)
datagen.fit(X_train)
train_generator=datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
val_generator=datagen.flow(X_test,y_test, batch_size=64, shuffle=True)
model = Sequential()

model.add(Conv2D(32, kernel_size=(11,11), activation='relu', padding='same', input_shape=(28, 28, 1)))

model.add(MaxPooling2D(3,3))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(5,5), activation='relu', padding='same'))
model.add(MaxPooling2D(3,3))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(3,3))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
AlexNet=model.fit(x=X_train, y=y_train,epochs=30,batch_size=128, validation_data=(X_test,y_test), shuffle=True)

loss = AlexNet.history['loss']
val_loss = AlexNet.history['val_loss']

len(loss)

train_epoch=range(1,31)
plt.plot(train_epoch,loss, label='loss')
plt.plot(train_epoch,val_loss, label='validation_loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
accuracy  = AlexNet.history['accuracy']
val_accuracy = AlexNet.history['val_accuracy']



train_epoch=range(1,31)
plt.plot(train_epoch,accuracy , label='accuracy')
plt.plot(train_epoch,val_accuracy , label='val_accuracy')

plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend()

text=('best accuracy' ,max(val_accuracy),)

plt.text(1,0.9940,text,fontsize =12)

# Data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator # for data augmentation

datagen = ImageDataGenerator(zoom_range=0.1,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             rotation_range=50)
datagen.fit(X_train)
alex_data_aug=model.fit_generator(train_generator, steps_per_epoch = 150,epochs=30,
                         validation_data=val_generator, validation_steps=10,shuffle=True)
loss = alex_data_aug.history['loss']
val_loss = alex_data_aug.history['val_loss']

len(loss)

train_epoch=range(1,31)
plt.plot(train_epoch,loss, label='loss')
plt.plot(train_epoch,val_loss, label='validation_loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('AlexNet Data Augmentation Loss')
accuracy  = alex_data_aug.history['accuracy']
val_accuracy = alex_data_aug.history['val_accuracy']



train_epoch=range(1,31)
plt.plot(train_epoch,accuracy , label='accuracy')
plt.plot(train_epoch,val_accuracy , label='val_accuracy')

plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend()

text=('best accuracy' ,max(val_accuracy),)

plt.text(1,0.6,text,fontsize =12)
plt.title('AlexNet Data Augmentation accuracy')

model2 = Sequential()

model2.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model2.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model2.add(MaxPooling2D(2,2))
model2.add(BatchNormalization())


model2.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model2.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model2.add(MaxPooling2D(2,2))
model2.add(BatchNormalization())
model2.summary()

model2.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model2.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model2.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model2.add(MaxPooling2D(2,2))
model2.add(BatchNormalization())

model2.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model2.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model2.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model2.add(MaxPooling2D(2,2))
model2.add(BatchNormalization())

model2.summary()
 
    #Should duplicate 3 times, but only 2. max pooling make it negative dimensions. 







model2.add(Flatten())
model2.add(Dense(4096, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(4096, activation='relu'))
model2.add(Dropout(0.25))

model2.add(Dense(10, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
VGG=model2.fit(x=X_train, y=y_train,epochs=30,batch_size=128, validation_data=(X_test,y_test), shuffle=True)

accuracy  = VGG.history['accuracy']
val_accuracy = VGG.history['val_accuracy']



train_epoch=range(1,31)
plt.plot(train_epoch,accuracy , label='accuracy')
plt.plot(train_epoch,val_accuracy , label='val_accuracy')

plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend()

text=('VGG best accuracy' ,max(val_accuracy),)

plt.text(1,0.9940,text,fontsize =12)
test_data=pd.read_csv('../input/Kannada-MNIST/test.csv')

test_data.values[0:, 1:].reshape(test_data.shape[0],28,28,1)
test_result=model2.predict(test_data.values[0:, 1:].reshape(test_data.shape[0],28,28,1))
print(test_result.shape)
test_res=np.argmax(test_result, axis=1, out=None)
test_res.shape
test_data.shape

sub=pd.DataFrame()
sub['id']=list(test_data.values[0:,0])


sub['label']=test_res
sub.to_csv("submission.csv", index=False)



