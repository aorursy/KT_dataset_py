import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data() #load cifar 10 dataset from keras

# let's see the dimen of our training data!
print(x_train.shape)
print(y_train.shape)
#Looks like x_train has 50000 entries of dimen 32*32*3 and y_train has 50000 entries of dimen 1
# let's see the dimen of our testing data!
print(x_test.shape)
print(y_test.shape)
# As expected, similar to the shape of testing data
# how about we try to see the contents. Let's look at the first element of x_train
x_train[0]
# It is simply an array of numbers - Note that these numbers denote the pixel values(0-255)
y_train
# We know that all the images are labelled over 10 categories. 
#So, the y_train is a number between 0 to 10 where each number depicts one category.
# time to re-scale so that all the pixel values lie within 0 to 1
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
# Let's see how it looks after re-scale
x_train[0]
no_of_classes = len(np.unique(y_train))
no_of_classes
import keras
# here, we are transforming y_train and y_test to be an array of size 10.
# The value of y_train/y_test as we saw earlier was a number from 0 to 9 each depicting one category.
# The value of y is represented by 1 in the corresponding array position and others are set to 0. 
# So, each row has only one item whose value will be 1 which depicts the category.
y_train = keras.utils.to_categorical(y_train,no_of_classes)
y_test = keras.utils.to_categorical(y_test,no_of_classes)
y_test
# we are going to divide our training set into 2 sets - train and validation.
x_train,x_valid = x_train[5000:],x_train[:5000]
y_train,y_valid = y_train[5000:],y_train[:5000]
print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
#let's visualize the first 50 images of training set
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(25,5))
for i in range(50):
    ax = fig.add_subplot(5,10,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
    
# Time to create our model ! Simple use of convolutional and max pooling layers.
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=16, kernel_size = 2, padding = 'same',activation = 'relu',input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size = 2, padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size = 2, padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10,activation='softmax'))

model.summary()
#Compile the model
model.compile(optimizer = 'rmsprop', loss ='categorical_crossentropy',metrics=['accuracy'])
print('compiled!')
# start training
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath = 'best_model.h5',save_best_only = True,verbose=1)

history = model.fit(x_train,y_train,batch_size=32, epochs = 100,
          validation_data=(x_valid,y_valid),
          callbacks=[checkpoint],
          verbose=2, shuffle=True)
#Let's check the accuracy score of the best model on our test set
model.load_weights('best_model.h5')
score = model.evaluate(x_test,y_test,verbose=0)
score[1]
# Not bad ! we have an accuracy score of 68% on our test set.
#Lets try to visualize the accuracy and loss over the epochs.
plt.figure(1)  
   
 # summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
   
 # summarize history for loss  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()  