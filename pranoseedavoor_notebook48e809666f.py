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
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Activation,Dropout,Conv2D,MaxPool2D,Flatten

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt 
train_data=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

SAMPLE=pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

SAMPLE.head()


y=train_data.pop("label")

x=train_data

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

##scalers to be used

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(X_train)

x_train=scaler.transform(X_train)

x_test=scaler.transform(X_test)

test_scaled=scaler.fit_transform(test_data)

enc = OneHotEncoder(handle_unknown='ignore')

from tensorflow.keras.utils import to_categorical

#unique_label=np.sort(y_train.unique())

#print(unique_label)



y_train=to_categorical(y_train,10)

y_test=to_categorical(y_test,10)

print(x_train.shape)

x_train=x_train.reshape(39900,28,28,1)

x_test=x_test.reshape(2100,28,28,1)

#plt.imshow(x_train[1,:,:,0])

shape=x_train[1,:,:,:].shape

print(test_scaled.max())
model=Sequential()

#model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))

#model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(filters=256,kernel_size=(4,4),activation='relu',input_shape=shape))       

model.add(Dropout(.3))

          

model.add(Conv2D(filters=128,kernel_size=(2,2),strides=(2,2),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

          

      

          

model.add(Conv2D(filters=64,kernel_size=(2,2),strides=(1,1),activation='relu'))

model.add(MaxPool2D(pool_size=(1,1)))





model.add(Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

          

          

model.add(Flatten())



model.add(Dense(256,activation='relu'))

model.add(Dropout(.3))



model.add(Dense(10,activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=["accuracy"])



callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

model.fit(x=x_train,y=y_train,epochs=50,batch_size=256,validation_data=(x_test, y_test),callbacks=[callback])
losses = pd.DataFrame(model.history.history)

losses.head()

plt.plot(losses[["loss","val_loss"]])

len(test_data)
j=0

y=16

x=16

plt.figure(figsize=(12,10))

for i in range (0,100):

    plt.subplot(y, x, j+1)

    plt.imshow(np.array(test_data.loc[i]).reshape(28,28))

    plt.title(str(j))

    j = j + 1

plt.show()
eval_val=model.predict(np.array(test_data.loc[1]).reshape(1,784))


test_final=np.array(test_scaled).reshape(28000,28,28,1)

#test_data.shape
test_data.min()



predictions = model.predict_classes(test_final, verbose=0)
df = pd.DataFrame(predictions) 
submission_file = pd.DataFrame({'ImageId': range(1,len(predictions)+1) ,'Label':predictions }) 
submission_file.head(10)
submission_file.to_csv("submission_1.csv",index=False)