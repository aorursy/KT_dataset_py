#import tensorflow
import numpy as np
import tensorflow as tf
#loading dataset
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data('../input/mnist-digit-recognition/mnist (1).npz')
#random example from trainning set
import matplotlib.pyplot as plt
%matplotlib inline
image_index=1111
print(y_train[image_index])
plt.imshow(x_train[image_index])
#shape of train set
x_train.shape
#reshape to 4-d 
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test  = x_test.reshape(x_test.shape[0],28,28,1)
input_shape =(28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train/=255
x_test/=255
 
# shape and size of sets    
print('x_train_shape :', x_train.shape)
print('x_test_shape :', x_test.shape)
print('images in train set:',x_train.shape[0])
print('images in test set:',x_test.shape[0])

#building model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , Dense , Dropout , MaxPooling2D , Flatten

model= Sequential()
model.add( Conv2D(28 ,kernel_size=(3,3) ,input_shape = input_shape))
model.add( MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128 , activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10 ,activation= tf.nn.sigmoid))
model.summary()
#optimizing
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=10)
model.evaluate(x_test,y_test)
image_index=67
plt.imshow(x_test[image_index].reshape(28,28),cmap='Greys')
pred =model.predict(x_test[image_index].reshape(1,28,28,1))
print('The no. is:',pred.argmax())

