# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense , Dropout , Flatten
from keras.layers import Conv2D , MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#print("Train Data:{}\nTest Data:{}".format(train.head, test.head))
print("Train size:{}\nTest size:{}".format(train.shape, test.shape))

train_X = train.drop('label',1).values
train_Y = train['label'].values
test_X = test.values
print("Train shape:{}".format(train_X.shape))
#print(train_X[0])
#print(train_Y[0])
print("Test shape:{}".format(test_X.shape))
#print(test_X[0])
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X.reshape(train_X.shape[0],28,28)
test_X = test_X.reshape(test_X.shape[0],28,28)
for j,i in enumerate(range(15,20)):
    plt.subplot(1,5,j+1)
    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title(train_Y[i]);
train_X = train_X.reshape(train_X.shape[0],28,28,1)
test_X = test_X.reshape(test_X.shape[0],28,28,1)


# Rescaling
train_X = train_X / 255
test_X = test_X / 255
# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape
batch_size = 64
epochs = 20
num_classes = 10

model = Sequential()

model.add(Conv2D(32,kernel_size = (3,3),activation = 'linear',input_shape = (28,28,1),padding = 'same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2),padding='same')) # same padding means the output image size will be same as the input imageso we do padding
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))  
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics = ['accuracy'])
train_model = model.fit(train_X,
                 train_label,
                 batch_size=batch_size,
                 epochs = 20,
                 verbose = 1,
                 validation_data = (valid_X,valid_label))
accuracy = train_model.history['acc']
print(accuracy)
val_accuracy = train_model.history['val_acc']
print(val_accuracy)
loss = train_model.history['loss']
print(loss)
val_loss = train_model.history['val_loss']
print(val_loss)
epochs = range(len(accuracy))
print(epochs)
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
pred_Y = model.predict(test_X)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(pred_Y,axis = 1) 
Y_true = np.argmax(valid_label,axis = 1)
