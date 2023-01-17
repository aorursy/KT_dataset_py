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


import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
X = dataset.iloc[:,1:].values

y = dataset.iloc[:,0].values

for i in range(0,10):

    img = X[i,:]

    img = np.reshape(img,(28,28))

    plt.subplot(4,4,i+1)

    plt.axis('off')

    plt.title(y[i])

    plt.imshow(img,cmap='gray',interpolation=None)

    
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split( X, y,test_size=0.20,random_state=42)
print("train samples:",X_train.shape[0])

print("validation samples:",X_val.shape[0])
X_train = X_train.astype('float32')

X_val = X_val.astype('float32')

X_train /= 255

X_val /= 255
img_shape = 28

X_train = np.reshape(X_train,(X_train.shape[0], img_shape, img_shape, 1))

X_val = np.reshape(X_val,(X_val.shape[0], img_shape, img_shape, 1))

input_shape = (img_shape, img_shape, 1)
from tensorflow import keras

import tensorflow

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten,BatchNormalization

from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
batch_size = 256

num_classes = 10

epochs = 50

y_train = keras.utils.to_categorical(y_train,num_classes)

y_val = keras.utils.to_categorical(y_val,num_classes)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.25))

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer="rmsprop",

              metrics=['accuracy'])
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
history = model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          callbacks=[callback],

          validation_data=(X_val, y_val))

loss,accuracy = model.evaluate(X_val, y_val,batch_size=256, verbose=1)

print("accuracy:",accuracy)

print("loss:",loss) ##99.65  batch norm ekledim epochu dusur 0.02686 



val_loss = history.history['val_loss']

val_acc = history.history['val_accuracy']

train_loss = history.history['loss']

train_acc = history.history['accuracy']
plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(train_acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(train_loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([0,1.0])

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
sample_sub = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")
test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

X_test = test.iloc[:,1:].values

test_ID = test.iloc[:,0].values



X_test = X_test.astype('float32')

X_test /=255

X_test = np.reshape(X_test,(X_test.shape[0], img_shape, img_shape, 1))


predicts = model.predict(X_test,batch_size=256)
predicts_d = pd.DataFrame(predicts)
predicts_d
number_pred =[]

for i in range(predicts_d.shape[0]):

     probs = predicts_d.values[i]

     for index,number in enumerate(probs):

               max_ = probs.max()

               if probs[index] == max_:

                   number_pred.append(index)

                   

               

   

    
sub_dict = {"id":test_ID,"label":number_pred}

sub_dt = pd.DataFrame(sub_dict)
sub_csv = sub_dt.to_csv('my-submission23.csv',index=False)