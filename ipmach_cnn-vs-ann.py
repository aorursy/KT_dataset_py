from keras.datasets import mnist

from keras.models import Sequential

import numpy as np

from keras.utils import to_categorical 

from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D 

from matplotlib import pyplot as plt
def plot_training_score(history):

  print('Availible variables to plot: {}'.format(history.history.keys()))

  print(" Loss vs epochs")

  plt.figure(1)

  plt.plot( np.arange(len(history.history['loss'])),history.history['loss'])

  print(" Accuracy vs epochs")

  plt.figure(2)

  plt.plot( np.arange(len(history.history['acc'])), history.history['acc'])



def normalize_prototype(listMatrix):

  (a,_,_)=listMatrix.shape

  change = lambda t: 0 if t < 230 else 255

  vfunc = np.vectorize(change)

  for i in range(0,a):

    listMatrix[i] = vfunc(listMatrix[i])

  return listMatrix


#Loading Database

(x_train_entire, y_train_entire), (x_test, y_test) = mnist.load_data()



print('x_train_entire size: {}, y_train_entire size: {}'.format(len(x_train_entire), 

                                                                 len(y_train_entire)))

print('x_train size: {}, y_train size: {}\n'.format(len(x_test), len(y_test)))



# Show the format of one randomly chosen image and label (123)

image = x_train_entire[123]

label = y_train_entire[123]

print('Image shape: {}'.format(image.shape))

print('Label: {}'.format(label))



# Split x_train_entire and y_train_entire into training and validation

x_train    = x_train_entire[ :50000]

x_validate = x_train_entire[50001: ]

y_train    = y_train_entire[ :50000]

y_validate = y_train_entire[50001: ]



#new = normalize_prototype(x_train)

x_train = normalize_prototype(x_train)

x_test = normalize_prototype(x_test)

x_validate = normalize_prototype(x_validate)



# Flaten images for ANN

x_train_flatten = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test_flatten = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_validate_flatten = x_validate.reshape((len(x_validate), np.prod(x_test.shape[1:])))



#Reshaping images for CNN

(a,b,c)= x_train.shape

x_train_conv = np.reshape(x_train, (a,b,c,1))

(a,_,_) = x_validate.shape

x_validate_conv = np.reshape(x_validate, (a,b,c,1))

(a,_,_) = x_test.shape

x_test_conv = np.reshape(x_test, (a,b,c,1))



# Convert in one-hot encoding



y_train_one_hot = to_categorical(y_train)

y_test_one_hot = to_categorical(y_test)

y_validate_one_hot= to_categorical(y_validate)
def ann(sizeInput):

  model = Sequential()  # Initalize a new model

  model.add(Dense(512,activation='relu', bias_initializer='zeros', use_bias= True, input_dim = sizeInput))

  model.add(Dense(10,activation='softmax'))

  model.compile(loss='categorical_crossentropy',

              optimizer='sgd',

              metrics=['accuracy']) 

  return model



def cnn(img_width, img_height):

  model = Sequential()  # Initalize a new model

  model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = (28,28,1)))

  model.add(MaxPooling2D((2,2)))

  model.add(Conv2D(64,(3,3), activation = 'relu'))

  model.add(MaxPooling2D((2,2)))

  model.add(Flatten())

  model.add(Dense(64, activation= 'relu'))

  model.add(Dense(10,activation = 'softmax'))

  model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy']) 

  return model



#Get size of the data

width, height = image.shape

sizeImage = width * height



#Initializing models

model_ann = ann(sizeImage)

model_cnn = cnn(width, height)



print("ANN model")

model_ann.summary()

print("CNN model")

model_cnn.summary()

#Training ANN

history = model_ann.fit(x_train_flatten, y_train_one_hot, epochs=30, batch_size=512, validation_data=(x_validate_flatten, y_validate_one_hot))

#Testing ANN

score,acc = model_ann.evaluate(x_test_flatten, y_test_one_hot,batch_size=128)

#plot_training_score(history)

print('score: ' + str(score))

print('acc: ' + str(acc))

#Training CNN

history = model_cnn.fit(x_train_conv, y_train_one_hot, epochs=15, batch_size=128, validation_data=(x_validate_conv, y_validate_one_hot))

#Testing CNN

score,acc = model_cnn.evaluate(x_test_conv, y_test_one_hot,batch_size=128) 

#plot_training_score(history)

print('score: ' + str(score))

print('acc: ' + str(acc))