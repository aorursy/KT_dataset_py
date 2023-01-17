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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)
x_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
# check the digitis in train set
import matplotlib.pyplot as plt

def display(dataset):
    plt.figure(figsize = (10,5))
    for num in range(0,10):
        plt.subplot(2,5,num+1)
        grid_data = dataset.iloc[num].as_matrix().reshape(28,28)
        plt.imshow(grid_data, interpolation = "none", cmap = "Greys")
        
display(x_train)
from keras.utils import np_utils

# rescale [0,255] --> [0,1]
x_train = x_train.astype('float32')/255
x_test = test.astype('float32')/255

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)

# one-hot encode the labels
y_train = np_utils.to_categorical(y_train, 10)

from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

n = 5
model = [0]*n
# build model
for i in range(n):
    model[i] = Sequential()
    model[i].add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
    model[i].add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
    model[i].add(MaxPool2D(pool_size=(2,2)))
    model[i].add(Dropout(0.25))

    model[i].add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
    model[i].add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
    model[i].add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model[i].add(Dropout(0.25))

    model[i].add(Flatten())
    model[i].add(Dense(256, activation = "relu"))
    model[i].add(Dropout(0.5))
    model[i].add(Dense(10, activation = "softmax"))
    
    # compile the model
    model[i].compile(loss='categorical_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])
# data augmentation
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(        
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)   

from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import train_test_split

### TODO: 设置训练模型的epochs的数量
history = [0]*n
epochs = 30
batch_size = 80

# DECREASE LEARNING RATE EACH EPOCH
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# train the model
for j in range(n):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1,random_state= 42)
    history[j] = model[j].fit_generator(datagen.flow(x_train,y_train, batch_size= batch_size),
                                        epochs = epochs, steps_per_epoch = x_train.shape[0] // batch_size,  
                                        validation_data = (x_val,y_val), callbacks=[learning_rate_reduction], verbose=1)
# model evaluation
for j in range(n):    
    print("CNN {:2d}: Epochs={}, Train accuracy={:.5f}, Validation accuracy={:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
# ensemble predictions
preds = np.zeros((x_test.shape[0], 10))
for i in range(n):
    preds += model[i].predict(x_test)
preds = np.argmax(preds, axis = 1)   
# save result
result = pd.DataFrame({'ImageId':range(1,len(preds)+1), 'Label':preds})
result.to_csv('result.csv', index = False)
# display errors

def display_errors(x_val):
    # get error indexes    
    y_true = np.argmax(y_val, axis = 1)
    error_idx = [i for i in range(len(y_true)) if preds[i]!=y_true[i]]
    print(len(error_idx))            
    # display imgs
    fig = plt.figure(figsize=(10,10)) 
    n = 1
    for i in error_idx:
        img = x_val[i][:,:,0]        
        ax = fig.add_subplot(4, 5, n, xticks=[], yticks=[])
        ax.imshow(img,cmap = "Greys")       
        ax.set_title('True: {}\nPred: {}'.format(y_true[i], preds[i]))
        n += 1
        if n > 20:
            break
            
display_errors(x_val)
