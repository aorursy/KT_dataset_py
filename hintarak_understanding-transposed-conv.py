import numpy as np

from keras.models import Sequential

from keras.layers import Conv2DTranspose
x = np.arange(4).reshape(2,2)+1

print(x)
model = Sequential()

model.add(Conv2DTranspose(1,1,strides=2, input_shape=(2,2,1)))

model.summary()
print('Kernel size is 1')

print('x : \n',x)



weights = [np.ones((1,1,1,1)).astype('float32'), np.array([0.]).astype('float32')]

model.set_weights(weights)



yhat = model.predict(x.reshape(1,2,2,1))

print('\n',yhat.reshape(4,4))
x = np.arange(4).reshape(2,2)+1

print('Kernel size is 2')

print('x : \n',x)



model = Sequential()

model.add(Conv2DTranspose(1,2,strides=2, input_shape=(2,2,1)))



weights = [np.ones((2,2,1,1)).astype('float32'), np.array([0.]).astype('float32')]

model.set_weights(weights)



yhat = model.predict(x.reshape(1,2,2,1))

print('\n',yhat.reshape(4,4))
x = np.arange(4).reshape(2,2)+1

print('Kernel size is 3')

print('x : \n',x)



model = Sequential()

model.add(Conv2DTranspose(1,3,strides=2, input_shape=(2,2,1)))



weights = [np.ones((3,3,1,1)).astype('float32'), np.array([0.]).astype('float32')]

model.set_weights(weights)



yhat = model.predict(x.reshape(1,2,2,1))

print('\n',yhat.reshape(5,5))
x = np.arange(4).reshape(2,2)+1

print('Kernel size is 3 with same padding')

print('x : \n',x)



model = Sequential()

model.add(Conv2DTranspose(1,3,strides=2, padding='same', input_shape=(2,2,1)))



weights = [np.ones((3,3,1,1)).astype('float32'), np.array([0.]).astype('float32')]

model.set_weights(weights)



yhat = model.predict(x.reshape(1,2,2,1))

print('\n',yhat.reshape(4,4))