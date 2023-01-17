!pip install mnist 

import numpy as np

import mnist

from tensorflow import keras
# The first time you run this might be a bit slow, since the

# mnist package has to download and cache the data.

train_images = mnist.train_images()

train_labels = mnist.train_labels()



print(train_images.shape) # (60000, 28, 28)

print(train_labels.shape) # (60000,)
import numpy as np

import mnist



train_images = mnist.train_images()

train_labels = mnist.train_labels()

test_images = mnist.test_images()

test_labels = mnist.test_labels()



# Normalize the images.

train_images = (train_images / 255) - 0.5

test_images = (test_images / 255) - 0.5



# Reshape the images.

train_images = np.expand_dims(train_images, axis=3)

test_images = np.expand_dims(test_images, axis=3)



print(train_images.shape) # (60000, 28, 28, 1)

print(test_images.shape)  # (10000, 28, 28, 1)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from tensorflow.keras.models import Sequential

num_filters = 8

filter_size = 3

pool_size = 2



model = Sequential([

  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),

  MaxPooling2D(pool_size=pool_size),

  Flatten(),

  Dense(10, activation='softmax'),

])
model.compile(

  'adam',

  loss='categorical_crossentropy',

  metrics=['accuracy'],

)
import mnist



train_labels = mnist.train_labels()

print(train_labels[0]) # 5
from tensorflow.keras.utils import to_categorical



model.fit(

  train_images,

  to_categorical(train_labels),

  epochs=3,

  validation_data=(test_images, to_categorical(test_labels)),

)
model.save_weights('cnn.h5')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten



num_filters = 8

filter_size = 3

pool_size = 2



# Build the model.

model = Sequential([

  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),

  MaxPooling2D(pool_size=pool_size),

  Flatten(),

  Dense(10, activation='softmax'),

])



# Load the model's saved weights.

model.load_weights('cnn.h5')
# Predict on the first 5 test images.

predictions = model.predict(test_images[:5])



# Print our model's predictions.

print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]



# Check our predictions against the ground truths.

print(test_labels[:5]) # [7, 2, 1, 0, 4]
import numpy as np

from matplotlib import pyplot as plt



from keras import backend as K

# from keras.models import Sequential

from keras.layers import Input, Dense, Dropout, Activation, ZeroPadding2D

from keras.layers import Flatten, Conv2D, MaxPooling2D, BatchNormalization

from keras.models import Model, load_model, model_from_json, model_from_yaml

from keras.utils import to_categorical



batch_sz = 128

n_classes = 10

n_epoch = 12

shapeX = (28,28,1)

classes = np.asarray([0,1,2,3,4,5,6,7,8,9],dtype = np.float32)
def MNIST_Model(input_shape = (28,28,1),classes = 10):

	X_input = Input(input_shape)



	# zero padding probably not required since the main digit is in the centre only

	# X = zeroPadding2D((1,1))(X_input)



	X = Conv2D(32,(3,3),strides = (1,1), name = 'conv0')(X_input)

	X = BatchNormalization(axis=3,name='bn0')(X)

	X = Activation('relu')(X)

	X = Conv2D(32,(3,3),strides = (1,1), name = 'conv1')(X)

	X = BatchNormalization(axis=3,name='bn1')(X)

	X = Activation('relu')(X)

	X = MaxPooling2D((2,2),strides = (2,2),name = 'MP1')(X)



	X = Conv2D(64,(3,3),strides = (1,1), name = 'conv2')(X)

	X = BatchNormalization(axis=3,name='bn2')(X)

	X = Activation('relu')(X)

	X = Conv2D(64,(3,3),strides = (1,1), name = 'conv3')(X)

	X = BatchNormalization(axis=3,name='bn3')(X)

	X = Activation('relu')(X)

	X = MaxPooling2D((2,2),strides = (2,2),name = 'MP2')(X)

	

	X = Dropout(0.2)(X)

	X = Flatten()(X)

	X = Dense(256,activation = 'relu',name= 'fc1')(X)

	X = Dropout(0.4)(X)

	X = Dense(n_classes,activation = 'softmax',name = 'fco')(X)



	model = Model(inputs = X_input,outputs = X, name = 'MNIST_Model')

	return model



modelMNIST = MNIST_Model(shapeX,n_classes)

print (modelMNIST.summary())



modelMNIST.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



modelMNIST.fit(train_images, to_categorical(train_labels), epochs = n_epoch, batch_size = batch_sz)



pred = modelMNIST.evaluate(test_images,to_categorical(test_labels))





print ("Loss = " + str(pred[0]))

print ("Test Accuracy = " + str(pred[1]))


modelMNIST.save_weights('cnn_1.h5')

# Predict on the first 5 test images.

predictions = modelMNIST.predict(test_images[:5])



# Print our model's predictions.

print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]



# Check our predictions against the ground truths.

print(test_labels[:5]) # [7, 2, 1, 0, 4]
# Predict on the first 5 test images.

predictions = modelMNIST.predict(test_images[:7])



# Print our model's predictions.

print(np.argmax(predictions, axis=1)) # [7 2 1 0 4 1 4]



# Check our predictions against the ground truths.

print(test_labels[:7]) # [7 2 1 0 4 1 4]