#Load libraries

import numpy as np

import keras

from keras import backend as K

from keras.models import Sequential, Model

from keras.layers import Activation

from keras.layers.core import Flatten, Dense

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import *

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

import itertools

from keras.preprocessing.image import ImageDataGenerator

from keras import applications

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras import optimizers

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

%matplotlib inline
datagen = ImageDataGenerator(rescale=1./255)

train_path = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/'



train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(115,115), 

                                                         classes=['Parasitized','Uninfected'], class_mode='categorical', 

                                                         batch_size=27560)



imgs, label = next(train_batches)

print(label.shape)

print(imgs.shape)
# We need to separate the data into train and test arrays 

X_train, X_test, y_train, y_test = train_test_split(imgs,label,test_size=0.3,random_state=42)


datagen = ImageDataGenerator(

        rotation_range=10,  # Rotate images randomly 

        zoom_range = 0.1, # Zoom images randomly

        width_shift_range=0.1,  # Horizontally shift images in random order

        height_shift_range=0.1  # Vertically shift images in random order

)



datagen.fit(X_train)
for i in range (0,5):

    image = imgs[i]

    plt.imshow((image * 255).astype(np.uint8))

    if (label[i][0]==1):

      print("Parasitized")

    if (label[i][1]==1):

        print("Uninfected")

    plt.show()
# Confusion Matrix

def show_confusion_matrix(history,model, x_test, y_test):

  

  # summarize history for accuracy

  plt.plot(history.history['acc'])

  plt.plot(history.history['val_acc'])

  plt.title('model accuracy')

  plt.ylabel('accuracy')

  plt.xlabel('epoch')

  plt.legend(['train', 'test'],loc='upper left')

  plt.show()



  pre_cls=model.predict_classes(x_test)   



  #Decode from one hot encoded to original for confusion matrix

  decoded_label = np.zeros(x_test.shape[0],)

  for i in range(y_test.shape[0]):

    decoded_label[i] = np.argmax(y_test[i])

  #print(decoded_label.shape)



  cm1 = confusion_matrix(decoded_label,pre_cls)

  print('Confusion Matrix : \n')

  print(cm1)
# model1 freeze all layers

# VGG16 pre-trained model

image_w, image_h = 115, 115

model1 = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (image_w, image_h, 3))

model1.summary()



# Freezing all the layers

for layer in model1.layers[:]:

    layer.trainable = False

    

# Trainable layers   

print("Trainable Layers:")

for i, layer in enumerate(model1.layers):

    print(i, layer.name, layer.trainable)

    

# Adding custom layers to create a new model 

new_model = Sequential([

    model1,

    Flatten(name='flatten'),

    Dense(256, activation='relu', name='new_fc1'),

    Dropout(0.5),

    Dense(2, activation='softmax', name='new_predictions')

])

new_model.summary()



# Compiling the model - SGD Optimizer

new_model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])



#Fitting the model on the train data and labels.

print("Batch Size: 10, Epoch: 5, Optimizer: SGD")

new_model.fit(imgs, label, batch_size=10, epochs=5, verbose=1, validation_split=0.30, shuffle=True)





# Compiling the model - Adam Optimizer

new_model.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.0001), metrics=["accuracy"])



#Fitting the model on the train data and labels.

print("Batch Size: 10, Epoch: 5, Optimizer: Adam")

new_model.fit(imgs, label, batch_size=10, epochs=5, verbose=1, validation_split=0.30, shuffle=True)

#Model2 unfreeze last layer

# VGG16 pre-trained model without fully connected layers and with different input dimensions

image_w, image_h = 115, 115

model2 = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (image_w, image_h, 3))

#model2.summary()



# Freezing the layers except the last one

for i, layer in enumerate(model2.layers):

  layer.trainable = False

  if (i >=15 ):

    layer.trainable = True



for i, layer in enumerate(model2.layers):

    print(i, layer.name, layer.trainable)

    

# Adding custom layers to create a new model 

new_model2 = Sequential([

    model2,

    Flatten(name='flatten'),

    Dense(512, activation='relu', name='new_fc1'),

    Dropout(0.5),

    Dense(2, activation='softmax', name='new_predictions')

])
# Compiling the model - SGD Optimizer

new_model2.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.00001, momentum=0.9), metrics=["accuracy"])



#Fitting the model on the train data and labels.

print("Batch Size: 10, Epoch: 5, Optimizer: SGD")

new_model2.fit(imgs, label, batch_size=10, epochs=5, verbose=1, validation_split=0.30, shuffle=True)
#Model2 unfreeze last layer

# VGG16 pre-trained model without fully connected layers and with different input dimensions

image_w, image_h = 115, 115

model2 = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (image_w, image_h, 3))

#model2.summary()



# Freezing the layers except the last one

for i, layer in enumerate(model2.layers):

  layer.trainable = True

  

for i, layer in enumerate(model2.layers):

    print(i, layer.name, layer.trainable)

    

# Adding custom layers to create a new model 

new_model2 = Sequential([

    model2,

    Flatten(name='flatten'),

    Dense(512, activation='relu', name='new_fc1'),

    Dropout(0.5),

    Dense(2, activation='softmax', name='new_predictions')

])

    
# Compiling the model - Adam Optimizer

new_model2.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.00001), metrics=["accuracy"])



#Fitting the model on the train data and labels.

print("Batch Size: 10, Epoch: 5, Optimizer: Adam")

new_model2.fit(imgs, label, batch_size=10, epochs=5, verbose=1, validation_split=0.30, shuffle=True)
image_w, image_h = 115, 115

model = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(image_w, image_h, 3))



model.summary()    

    # Adding custom layers to create a new model 

new_model2 = Sequential([

    model,

    Flatten(name='flatten'),

    Dense(512, activation='relu', name='new_fc1'),

    Dropout(0.5),

    Dense(2, activation='softmax', name='new_predictions')

])

#new_model2.summary()
# Compiling the model - Adam Optimizer

new_model2.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.00001), metrics=["accuracy"])

es = keras.callbacks.EarlyStopping(monitor='val_acc', verbose=1, mode='max', patience=2)

#Fitting the model on the train data and labels.

batch_size = 100

history = new_model2.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=10, validation_data = (X_test, y_test), verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=[es])



print("Batch Size: 20, Epoch: 5, Optimizer: Adam")



show_confusion_matrix(history,new_model2, X_test, y_test)
image_w, image_h = 115, 115

model = keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(image_w, image_h, 3))



for i, layer in enumerate(model.layers):

    print(i, layer.name, layer.trainable)

model.summary()    

    # Adding custom layers to create a new model 

new_model2 = Sequential([

    model,

    Flatten(name='flatten'),

    Dense(512, activation='relu', name='new_fc1'),

    Dropout(0.5),

    Dense(2, activation='softmax', name='new_predictions')

])

#new_model2.summary()
# Compiling the model - Adam Optimizer

new_model2.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.00001), metrics=["accuracy"])

es = keras.callbacks.EarlyStopping(monitor='val_acc', verbose=1, mode='max', patience=2)
#Fitting the model on the train data and labels.

batch_size = 100

history = new_model2.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=10, validation_data = (X_test, y_test), verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=[es])

print("Batch Size: 20, Epoch: 5, Optimizer: Adam")



show_confusion_matrix(history,new_model2, X_test, y_test)