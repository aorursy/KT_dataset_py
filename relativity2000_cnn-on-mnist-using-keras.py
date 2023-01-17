#importing required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
#acquiring required data
train = pd.read_csv("../input/train.csv")
print(train.shape)
test= pd.read_csv("../input/test.csv")
print(test.shape)
#storing data into variables
x_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
x_test = test.values.astype('float32')
# Example of a picture
index = 6
plt.imshow(x_train[index].reshape((28,28)))
print ("y = " + str(y_train[index]))
# Exploring the dataset 
m_train = x_train.shape[0]
num_px = x_train.shape[1]
m_test = x_test.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px)+")")
print ("train_x_orig shape: " + str(x_train.shape))
print ("train_y shape: " + str(y_train.shape))
print ("test_x_orig shape: " + str(x_test.shape))
#print ("test_y shape: " + str(y_test.shape))
# Reshape the training and test examples 
x_train, x_test = x_train / 255.0, x_test / 255.0
train_x = x_train.reshape((m_train,28,28,1)) # The "-1" makes reshape flatten the remaining dimensions
test_x= x_test.reshape((m_test,28,28,1))

input_shape=(28,28,1)
train_x=train_x.astype('float32')
test_x=test_x.astype('float32')

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
#implementing the model of the layers
model=Sequential()
model.add(Conv2D(6,kernel_size=(1,1),input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(16,kernel_size=(5,5),input_shape=(14,14,6)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Flatten())
model.add(Dense(120,activation=tf.nn.relu))
model.add(Dense(84,activation=tf.nn.relu))
model.add(Dense(10,activation=tf.nn.softmax))
#compiling
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#training
model.fit(x=train_x,y=y_train,epochs=30)
#printed example from the test database
image_index=231
plt.imshow(x_test[image_index].reshape(28,28),cmap='Greys')
pred=model.predict(x_test[image_index].reshape(1, 28,28, 1))
print("Predicted No is: ",pred.argmax())
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
Pred = model.predict(X_test)
Pred.shape
y_pred = Pred.argmax(axis=1)
ImageID = np.arange(len(y_pred))+1
Out = pd.DataFrame([ImageID,y_pred]).T
Out.rename(columns = {0:'ImageId', 1:'Label'})
#Out
Out.to_csv('MNIST_Prediction.csv', header =  ['ImageId', 'Label' ], index = None) 
