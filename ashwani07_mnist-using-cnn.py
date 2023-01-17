# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#load other modules
import time
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
#Load train and test dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#Printing the size of the dataframe
print ("Training dataset has %i observations and %i variables" %(train.shape[0], train.shape[1]))
print ("Testing dataset has %i observations and %i variables" %(test.shape[0], test.shape[1]))
# Seperating target and image pixels
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels
test_x = test.values.astype('float32')
# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
#Reshaping train to 28 * 28 pixel size with 1 color (grey scale)
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape
#Reshaping test to 28 * 28 pixel size with 1 color (grey scale)
test_x = test_x.reshape(test_x.shape[0], 28, 28,1)
test_x.shape
# Normalize the data,
# Colors are from 0 to 255. 
X_train = X_train / 255.0
test_x = test_x / 255.0
#Label encoding
y_train = to_categorical(y_train, num_classes = nClasses)
y_train.shape
np.random.seed(0)
# Split the train and the validation set for the fitting
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
input_shape = X_train.shape[1:]
def createModel():
    model = Sequential()
    # The first two layers with 32 filters of filter size 3x3
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

#    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
    
    return model

model = createModel()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
starttime = time.time()
cnn = model.fit(X_train, y_train, epochs=20, batch_size=100, verbose=1, validation_data = (X_test, y_test))
print ("Time elapsed in training: ", time.time()- starttime, "s")
#PLot
plt.figure(figsize=[8,6])
plt.plot(cnn.history['loss'],'r',linewidth=3.0)
plt.plot(cnn.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

plt.figure(figsize=[8,6])
plt.plot(cnn.history['acc'],'r',linewidth=3.0)
plt.plot(cnn.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
# predict results
results = model.predict(test_x)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("mnist_cnn.csv",index=False)
