#this is my second basic project on kaggle 
#formally going to use convolution neural network 
#basically this project can help me a lot for understanding CNN and ANN 


#simple importing the library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for data visualization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing the simple library for creating neural network 
#Using keras 
#using tensorflow backend 
#importing sequential library 

from keras.models import Sequential 
from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator

#after running this cell the backend of tensorflow will be activated


#importing the training dataset 
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()

#importing the test dataset 

test= pd.read_csv("../input/test.csv")
print(test.shape)
test.head()

#converting all the values of training and testing dataset into floating values
X_train = (train.iloc[:,1:].values).astype('float32') # features excluding label of images  
y_train = train.iloc[:,0].values.astype('int32') #labels of images 

X_test = test.values.astype('float32') 


#checking the value of training and testing dataset
X_train

y_train


#Convert training datset to (num_images, img_rows, img_cols) format
X_train = X_train.reshape(X_train.shape[0], 28, 28)


# data visualizing of the images 
for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);


#expand one more dimension in array of x_train as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape	

# same doing with testing dataset
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test.shape
# Feature Standardization
# It is important preprocessing step. 
#It is used to centre the data around zero mean and unit variance.
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def standardize(x): 
    return (x-mean_px)/std_px


# One Hot encoding of labels.
# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension.

# For example, 3 would be [0,0,0,1,0,0,0,0,0,0].

from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes

# plotting the first 10 0 & 1 after one hot encoding
plt.title(y_train[9])
plt.plot(y_train[9])
plt.xticks(range(10));
#knowing that when creating neural networks 
#it's standard practice to create a 'random seed' so that you can get producible results in your models
#it is designing phase of neural network architecture 
seed = 43
np.random.seed(seed)


from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
model= Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
print("input shape ",model.input_shape)
print("output shape ",model.output_shape)
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),
 loss='categorical_crossentropy',
 metrics=['accuracy'])
from keras.preprocessing import image
gen = image.ImageDataGenerator()
from sklearn.model_selection import train_test_split
X = X_train
y = y_train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches=gen.flow(X_val, y_val, batch_size=64)
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3, 
                    validation_data=val_batches, validation_steps=val_batches.n)
history_dict = history.history
history_dict.keys()
import matplotlib.pyplot as plt
%matplotlib inline
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1) #considering all epoch one by one one

# "bo" is for "blue dot"
plt.plot(epochs, loss_values, 'bo')
# b+ is for "blue crosses"
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()
plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()
def get_fc_model():
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
        ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
fc = get_fc_model()
fc.optimizer.lr=0.01
#simple using fit_generator method 
history=fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                    validation_data=val_batches, validation_steps=val_batches.n)
#def get_cnn_model():
#    model = Sequential([
 #       Lambda(standardize, input_shape=(28,28,1)),
  #      Convolution2D(32,(3,3), activation='relu'),
   #     Convolution2D(32,(3,3), activation='relu'),
    #    MaxPooling2D(),
     #   Convolution2D(64,(3,3), activation='relu'),
      #  Convolution2D(64,(3,3), activation='relu'),
       # MaxPooling2D(),
#        Flatten(),
 #       Dense(512, activation='relu'),
  #      Dense(10, activation='softmax')
   #     ])
 #   model.compile(Adam(), loss='categorical_crossentropy',
#                  metrics=['accuracy'])
 #   return model
#model= get_cnn_model()
#model.optimizer.lr=0.01
#history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs= 1, 
 #                   validation_data=val_batches, validation_steps=val_batches.n)
#Sorry very time taking as we have added upto 2 sdense layer
fc.optimizer.lr=0.01
gen = image.ImageDataGenerator()
batches = gen.flow(X, y, batch_size=64)
history=fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)
#creating a submission dataset for kaggle submission 
predictions = fc.predict_classes(X_test, verbose=0)

subm =pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
subm.to_csv("DR.csv", index=False, header=True)
