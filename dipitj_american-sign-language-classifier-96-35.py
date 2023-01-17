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
import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Dropout,BatchNormalization,Flatten
from keras.optimizers import Adam
train_data = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')
test_data = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')
print(train_data.shape, test_data.shape)
train_data.columns
label_frq=pd.value_counts(train_data['label'],ascending=True).reset_index(level=0)
pd.DataFrame(label_frq)
from PIL import Image
sign = Image.open('/kaggle/input/sign-language-mnist/amer_sign2.png')
sign
label_map={0:'A', 1:"B", 2:"C", 3:"D",4:"E",5:"F",6:"G",7:"H", 8:"I", 9:"J",10:"K", 11:"L",12:"M",
          13:"N", 14:"O",15:"P",16:"Q",17:"R",18:"S",19:"T",20:"U",21:"V",22:"X",23:"Y"}
print(label_map)
X_train = np.array(train_data.iloc[:,1:785])
y_train = np.array(train_data.iloc[:,0])
X_test = np.array(test_data.iloc[:,1:785])
y_test = np.array(test_data.iloc[:,0])
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
X_train = np.array(X_train).reshape(-1,28,28,1)
X_test = np.array(X_test).reshape(-1,28,28,1)
print(X_train.shape)
print(X_test.shape)
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 20,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

X_test = X_test/255
from sklearn.preprocessing import LabelBinarizer
le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train
model=Sequential()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPooling2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPooling2D((2,2),2,padding='same'))
          
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(24, activation="softmax"))
print(model.summary())
model.compile(loss="categorical_crossentropy",
             optimizer=Adam(lr=0.001),
             metrics=['accuracy'])
history = model.fit(train_datagen.flow(X_train,y_train,batch_size=200),
                 epochs = 30,
                  validation_data=(X_test,y_test))
model.save("ASL_epochs_30.h5")
scores = model.evaluate(X_test,y_test)
print("Test Accuracy: %.3f Test Loss: %.3f"%(scores[1]*100,scores[0]))
y_pred_30 = model.predict(X_test)
labels_30 = np.argmax(y_pred_30, axis=1)
labels_30
new_model=Sequential()
new_model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
new_model.add(Conv2D(128,kernel_size=(3,3),
                 strides=1,padding='same',activation='relu'))
new_model.add(MaxPooling2D(pool_size=(3,3),strides=2,padding='same'))

new_model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
new_model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
new_model.add(MaxPooling2D((2,2),2,padding='same'))

new_model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
new_model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
new_model.add(MaxPooling2D((2,2),2,padding='same'))
          
new_model.add(Flatten())
new_model.add(Dense(512, activation="relu"))
new_model.add(Dropout(0.25))
new_model.add(Dense(24, activation="softmax"))
print(new_model.summary())
new_model.compile(loss="categorical_crossentropy",
             optimizer=Adam(lr=0.0001),
             metrics=['accuracy'])

history_2 = new_model.fit(train_datagen.flow(X_train,y_train,batch_size=200),
                 epochs = 35,
                  validation_data=(X_test,y_test))
new_model.save('ASL_epoch_35.h5')
scores = new_model.evaluate(X_test,y_test)
print("Test Accuracy: %.3f Test Loss: %.3f"%(scores[1]*100,scores[0]))
y_pred_35 = new_model.predict(X_test)
labels_35 = np.argmax(y_pred_35, axis=1)
labels_35
new_model_2 = Sequential()
new_model_2.add(Conv2D(64,(5,5), padding="same", activation="relu",
                      strides = 1,input_shape=(28,28,1)))
new_model_2.add(Conv2D(64,(3,3), strides = 1,padding="same",activation="relu"))
new_model_2.add(BatchNormalization())
new_model_2.add(MaxPooling2D(pool_size=(2,2),strides=2))
new_model_2.add(Dropout(0.2))


new_model_2.add(Conv2D(32,(3,3), padding="same", activation="relu",
                      strides = 1))
new_model_2.add(Conv2D(32,(3,3), strides = 1,padding="same",activation="relu"))
new_model_2.add(BatchNormalization())
new_model_2.add(MaxPooling2D(pool_size=(2,2),strides=2))
new_model_2.add(Dropout(0.2))

new_model_2.add(Conv2D(32,(3,3), padding="same", activation="relu",
                      strides = 1))
new_model_2.add(Conv2D(32,(3,3), strides = 1,padding="same",activation="relu"))
new_model_2.add(BatchNormalization())
new_model_2.add(MaxPooling2D(pool_size=(2,2),strides=2))
new_model_2.add(Dropout(0.2))

new_model_2.add(Conv2D(32,(3,3), padding="same", activation="relu",
                      strides = 1))
new_model_2.add(Conv2D(32,(3,3), strides = 1,padding="same",activation="relu"))
new_model_2.add(BatchNormalization())
new_model_2.add(MaxPooling2D(pool_size=(2,2),strides=2))
new_model.add(Dropout(0.2))

new_model_2.add(Flatten())
new_model_2.add(Dense(512,activation="relu"))
new_model_2.add(Dropout(0.25))
new_model_2.add(Dense(24,activation="softmax"))
print(new_model_2.summary())
new_model_2.compile(loss="categorical_crossentropy",
                   optimizer=Adam(lr=0.0001),
                   metrics=['accuracy'])
history_3 = new_model_2.fit(train_datagen.flow(X_train,y_train,batch_size=200),
                           epochs=50,
                           validation_data = (X_test,y_test))
new_model_2.save('ASL_epoch_50.h5')
scores=new_model_2.evaluate(X_test,y_test)
print("Test Accuracy: %.3f Test Loss: %.3f"%(scores[1]*100,scores[0]))
y_pred_50 = new_model_2.predict(X_test)
labels_50 = np.argmax(y_pred_50,axis=1)
labels_50
Test_Preds=pd.DataFrame()
Test_Preds['30_epoch']=labels_30
Test_Preds['35_epoch']=labels_35
Test_Preds['50_epochs']=labels_50
Test_Preds
