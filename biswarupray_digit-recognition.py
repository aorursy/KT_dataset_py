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
from sklearn.model_selection import train_test_split

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

%matplotlib inline
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
X=train.drop(labels=["label"],axis=1)

y=train["label"]
X.isnull().any().describe()
test.isnull().any().describe()
X = X / 255.0

test = test / 255.0
X = X.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
y=to_categorical(y,num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=2)
# printing the number of samples in X_train, X_test, y_train, y_test

print("Initial shape of dimensions of X_train", str(X_train.shape))



print("Number of samples in our training data: "+ str(len(X_train)))

print("Number of labels in out training data: "+ str(len(y_train)))

print("Number of samples in our test data: "+ str(len(X_test)))

print("Number of labels in out test data: "+ str(len(y_test)))

print()

print("Dimensions of x_train:" + str(X_train[0].shape))

print("Labels in x_train:" + str(y_train.shape))

print()

print("Dimensions of X_test:" + str(X_test[0].shape))

print("Labels in X_test:" + str(y_test.shape))
g = plt.imshow(X_train[1][:,:,0])
num_classes = y_test.shape[1]

num_pixels = X_train.shape[1] * X_train.shape[2]

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.optimizers import SGD



#applying adam optimizer to reduce losses

optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

dropout=0.2



model = Sequential()

model.add(Conv2D(32, (3,3),activation='relu',input_shape=(28,28,1)))#adding convolutional layer

model.add(Conv2D(64, (3,3),activation='relu',input_shape=(28,28,1)))#adding convolutional layer        



model.add(MaxPooling2D(pool_size=(2, 2)))#adding pooling layer

model.add(Dropout(dropout))#regularizing using dropout with 20%

model.add(Flatten()) #flattening the input to apply Dense()

model.add(Dense(128, activation='relu'))#making a network with output of 128 dimensionality



model.add(Dropout(dropout))#regularizing using dropout with 20%

model.add(Dense(num_classes, activation='softmax'))#making a network with output of y_test.shape[1] dimensionality

model.compile(loss = 'categorical_crossentropy',optimizer = optimizer,metrics = ['accuracy'])



print(model.summary())

                 

                 

#training the model

batch_size = 66

epochs = 10



history = model.fit(X_train,y_train,batch_size = batch_size,epochs = epochs,verbose = 1,validation_data = (X_test, y_test))

                                                        

score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
%matplotlib inline

import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
%matplotlib inline

import matplotlib.pyplot as plt



history_dict = history.history



acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')

line2 = plt.plot(epochs, acc_values, label='Training Accuracy')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Accuracy')

plt.grid(True)

plt.legend()

plt.show()
#saving the model

model.save("/mnist_simple_cnn_10_epochs.h5")

print("Model save")
#loading model

from keras.models import load_model

classifier = load_model("/mnist_simple_cnn_10_epochs.h5")

print("Model loaded")
results = model.predict(test)





results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)