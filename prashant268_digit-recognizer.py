# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense 

from keras.layers import Dropout
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train.head()
X=train.iloc[:,1:].values

X
y=train.iloc[:,0].values

y
X.shape
X=X.reshape(42000,28,28,1)

X.shape
y.shape
plt.imshow(X[2][:,:,0],cmap='gray')
from keras.utils.np_utils import to_categorical

y=to_categorical(y)
y
y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
classifier=Sequential()
classifier.add(Convolution2D(32,(3,3),padding='same',input_shape=(28,28,1),activation='relu'))

classifier.add(Convolution2D(32,(3,3),padding='same',activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))



classifier.add(Convolution2D(64,(3,3),padding='same',activation='relu'))

classifier.add(Convolution2D(64,(3,3),padding='same',activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))



classifier.add(Convolution2D(128,(3,3),padding='same',activation='relu'))

classifier.add(Convolution2D(128,(3,3),padding='same',activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))



classifier.add(Flatten())



classifier.add(Dense(output_dim=512,activation='relu'))

classifier.add(Dropout(0.5))



classifier.add(Dense(output_dim=256,activation='relu'))

classifier.add(Dropout(0.5))



classifier.add(Dense(output_dim=10,activation='softmax'))
from keras.optimizers import Adam

opt=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

classifier.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(width_shift_range=0.1,

                                   height_shift_range = 0.1,

                                   rotation_range=30,

                                   rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   fill_mode='nearest'

                                  )



test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow(X_train,

                                  y_train,

                                  batch_size=32)

                                  



test_set = test_datagen.flow(X_test,

                             y_test,

                             batch_size=32)
X_train.shape
history=classifier.fit_generator(training_set,

                         steps_per_epoch=X_train.shape[0]/32,

                         epochs=50,

                         validation_data=test_set)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs=range(len(acc))
plt.plot(epochs,acc,label='Trainin_acc',color='blue')

plt.plot(epochs,val_acc,label='Validation_acc',color='red')

plt.legend()

plt.title("Training and Validation Accuracy")
plt.plot(epochs,loss,label='Training_loss',color='blue')

plt.plot(epochs,val_loss,label='Validation_loss',color='red')

plt.legend()

plt.title("Training and Validation loss")
final_model =Sequential()



final_model.add(Convolution2D(32,(3,3),padding='same',input_shape=(28,28,1),activation='relu'))

final_model.add(Convolution2D(32,(3,3),padding='same',activation='relu'))

final_model.add(MaxPooling2D(pool_size=(2,2)))



final_model.add(Convolution2D(64,(3,3),padding='same',activation='relu'))

final_model.add(Convolution2D(64,(3,3),padding='same',activation='relu'))

final_model.add(MaxPooling2D(pool_size=(2,2)))



final_model.add(Convolution2D(128,(3,3),padding='same',activation='relu'))

final_model.add(Convolution2D(128,(3,3),padding='same',activation='relu'))

final_model.add(MaxPooling2D(pool_size=(2,2)))



final_model.add(Flatten())



final_model.add(Dense(output_dim=512,activation='relu'))

final_model.add(Dropout(0.5))



final_model.add(Dense(output_dim=256,activation='relu'))

final_model.add(Dropout(0.5))



final_model.add(Dense(output_dim=10,activation='softmax'))
final_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
final_train_set = train_datagen.flow(X,

                                  y,

                                  batch_size=32)
final_model.fit_generator(final_train_set,

                         steps_per_epoch=X.shape[0]/32,

                         epochs=50)
test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test.head()
test.shape
test_array=test.values
test_array=test_array/255
test_array.max()
test_array=test_array.reshape((28000,28,28,1))
test_array.shape
prediction=final_model.predict(test_array)
prediction
prediction.shape
prediction=np.argmax(prediction,axis=1)
prediction
pd.DataFrame({'ImageId':pd.Series(range(1,28001)),'Label':prediction}).set_index('ImageId').to_csv('DigitSubmission.csv')