# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import math

import matplotlib.pyplot as plt

%matplotlib inline



import time #to measure execution time
#importing the data as pandas DataFrames

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data.tail()
test_data.tail() #there is no label in the test dataset
#now, let's convert the data to numpy arrays

#we could use the pandas as_matrix() method, but we received a warning

#FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.

#x_test = test_data.as_matrix()

#x_train = train_data.drop(['label'], axis=1).as_matrix()

#y_train = train_data['label'].as_matrix()



x_test = test_data.values

x_train = train_data.drop(['label'], axis=1).values

y_train = train_data['label'].values



x_test.shape, x_train.shape, y_train.shape

# We split the known data into train and validation sets

#let's choose 20% for validation set

from sklearn.model_selection import train_test_split

random_seed = 23

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.14, random_state=random_seed)
whos
#we scale the values using a linear interpolation, assuming that the values of the pixel intensities are initially from 0 to 255

xmin = 0.

xmax = 255. #the dot is to make them non-integer numbers, so that the result of the interpolation is also non-integer

ymin = 0

ymax = 1



x_train = (x_train-xmin)*(ymax-ymin)/(xmax-xmin)

x_test = (x_test-xmin)*(ymax-ymin)/(xmax-xmin)



x_train.shape, x_test.shape
x_train[10] #notice the values are now scaled from 0 to 1
plt.imshow(x_train[100].reshape([28,28]),'gray')
x_train=x_train.reshape(x_train.shape[0], 28, 28,1)

x_val=x_val.reshape(x_val.shape[0], 28, 28,1)

x_test=x_test.reshape(x_test.shape[0], 28, 28,1)



x_train.shape, x_val.shape, x_test.shape
from keras.models import Sequential #we import the class Sequential: Sequential Layer ANN Models

from keras.layers import Dense, BatchNormalization, Dropout

from keras.optimizers import SGD #there are more optimizers available too

from keras.regularizers import l2 #we use weight regularization

from keras.layers import Conv2D, MaxPool2D, Flatten



lr = 0.010#0.01

bs = 256 #batch size

nb = math.ceil(len(x_train)/bs) # batch number



model = Sequential([

    Conv2D(128, 3, activation='relu', padding='same', input_shape=(28,28,1)),

    MaxPool2D(),

    Flatten(),

    Dense(512, activation='relu', input_shape=(784,)),#we could have not added the input_shape information. it could have been inferred by keras

    #BatchNormalization(),#

    Dropout(0.25),

    Dense(10, activation='softmax', kernel_regularizer=l2(0.002))

          #because it's a classifier, we use softmax. 10 neurons at the end, because there are 10 classes (digits) to classify

])



#categorical_crossentropy: needs that we explicitly do the one hot encoding. if it's sparse, it does it internally

#we can get several metrics during the model training

#we use SGD with Nesterov momentum update

model.compile(SGD(lr, momentum=0.9, nesterov=True), loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.summary() #shows us the layer details
log = model.fit(x_train, y_train, batch_size=bs, epochs=23, validation_data=[x_val, y_val]) #model training
def show_results(model, log, cycling=False):

    loss, acc = model.evaluate(x_val, y_val, batch_size=512, verbose=False)

    print(f'Loss     = {loss:.4f}')

    print(f'Accuracy = {acc:.4f}')

    

    val_loss = log.history['val_loss']

    val_acc = log.history['val_acc']

    if cycling:

        val_loss += [loss]

        val_acc += [acc]

        

    fig, axes = plt.subplots(1, 2, figsize=(14,4))

    ax1, ax2 = axes

    ax1.plot(log.history['loss'], label='train')

    ax1.plot(val_loss, label='test')

    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss')

    ax2.plot(log.history['acc'], label='train')

    ax2.plot(val_acc, label='test')

    ax2.set_xlabel('epoch'); ax2.set_ylabel('accuracy')

    for ax in axes: ax.legend()
show_results(model, log)
# predict results

y_test = model.predict(x_test)



# select the indix with the maximum probability

y_test = np.argmax(y_test,axis = 1)



y_test = pd.Series(y_test,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_test],axis = 1)



submission.to_csv("predictions.csv",index=False)