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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import random

sns.set()
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
test.head()
print(f'shape of train dataset : {train.shape}')

print(f'shape of test dataset : {test.shape}')
training_array = np.array(train,dtype = 'float32')

testing_array = np.array(test,dtype = 'float32')
i = random.randint(0,training_array.shape[0])

plt.figure()

plt.imshow(training_array[i,1:].reshape(28,28))

plt.grid(False)

plt.show()

print(f'The image is for : {int(training_array[i,0])}')
w_grid = 15

l_gird = 15



fig ,axes = plt.subplots(l_gird,w_grid,figsize=(15,15))

axes = axes.ravel()



n_training = len(training_array)



for i in np.arange(0,l_gird*w_grid):

    

    index = np.random.randint(0,n_training)

    axes[i].imshow(training_array[index,1:].reshape(28,28))

    axes[i].set_title(int(training_array[index,0]),fontsize=8)

    axes[i].axis('off')

    

plt.subplots_adjust(hspace=0.4)
X_train = training_array[:,1:]/255

y_train = training_array[:,0]
test = testing_array/255
print(f'The shape of X_train : {X_train.shape}')

print(f'The shape of y_train : {y_train.shape}')

print(f'The shape of test dataset : {test.shape}')
X_train = X_train.reshape(X_train.shape[0],28,28,1)

test = test.reshape(test.shape[0],28,28,1)
print(f'The shape of X_train : {X_train.shape}')

print(f'The shape of y_train : {y_train.shape}')

print(f'The shape of test dataset : {test.shape}')
from sklearn.model_selection import train_test_split
X_train , X_validate ,y_train,y_validate = train_test_split(X_train,y_train,test_size = 0.1,random_state = 2)
print(f'The Shape of X_train : {X_train.shape}')

print(f'The Shape of X_validate : {X_validate.shape}')

print(f'The Shape of y_train : {y_train.shape}')

print(f'The Shape of y_validate : {y_validate.shape}')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPool2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping
print(f'The shape of y_train : {y_train.shape}')

print(f'The shape of y_validate : {y_validate.shape}')
y_train = to_categorical(y_train,num_classes=10)

y_validate = to_categorical(y_validate,num_classes=10)
print(f'The shape of y_train : {y_train.shape}')

print(f'The shape of y_validate : {y_validate.shape}')
model = Sequential()



model.add(Conv2D(filters=32,

                kernel_size=(3,3),

                padding='Same',

                 activation='relu',

                input_shape=(28,28,1)))



model.add(Conv2D(filters=32,

                kernel_size=(3,3),

                padding='Same',

                activation='relu'))



model.add(MaxPool2D(pool_size=(2,2)))



model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))



model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))



model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(256,

               activation='relu'))



model.add(Dropout(0.5))



model.add(Dense(10,

               activation='softmax'))
model.summary()
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
early_stopping =EarlyStopping(monitor='val_loss',patience=3,verbose=1)
model.fit(X_train,

         y_train,

         epochs=50,

          verbose=1,

         validation_data=(X_validate,y_validate),

         callbacks=[early_stopping])
model_history = pd.DataFrame(model.history.history)
model_history
model_history[['accuracy','val_accuracy']].plot()
model_history[['loss','val_loss']].plot()
predict_class = model.predict_classes(test)
predict_class
predict_class.shape
plt.imshow(test[0].reshape(28,28))
evaluate = model.evaluate(X_validate,y_validate)

print(f'The final Loss is : {evaluate[0]} \n The final Accuracy is : {evaluate[1]}')
y_hat = model.predict(X_validate)

y_pred = np.argmax(y_hat, axis=1)

y_true = np.argmax(y_validate, axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)
plt.figure(figsize=(10,10))

sns.heatmap(cm,annot=True,fmt='d',cmap='viridis')

plt.show()
W_grid = 6

L_grid = 6



fig,axes = plt.subplots(L_grid,W_grid,figsize=(15,15))



axes = axes.ravel()



for i in np.arange(0,L_grid*W_grid):

    axes[i].imshow(X_validate[i].reshape(28,28))

    axes[i].set_title(f'Predicted Class : {int(y_pred[i])} \n True Class : {y_true[i]}')

    axes[i].axis('off')

    

plt.subplots_adjust(wspace=0.4)
from sklearn.metrics import classification_report



num_classes = 10

target_names = ['Class {}'.format(i) for i in range(num_classes)]



print(classification_report(y_true,y_pred,target_names=target_names))
result = model.predict(test)
results = np.argmax(result,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)