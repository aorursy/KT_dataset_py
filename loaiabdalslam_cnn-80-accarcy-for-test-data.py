import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Activation


model = Sequential([
    
    #Create Convolution2d Layer with activation Relu
    Conv2D(32,(3,3),input_shape=(64,64,3)),
    Activation('relu'),
    #Create MaxsPooling2D Function 
    MaxPooling2D(pool_size=(2,2)),
    
    #Create Convolution 2d Layer again 
    Conv2D(32,(3,3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    #Flatten layer !
    Flatten(),
    
 
    #Full connection
    Dense(units=128),
    Activation('relu'),
    
    
    Dense(units=1),
    Activation('sigmoid'),
       
])


# For a binary classification problem
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../input/training_set/training_set/',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../input/test_set/test_set/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


model.summary()
history=model.fit_generator(training_set,
                   samples_per_epoch=8000,
                   nb_epoch=50,
                   validation_data=test_set,
                   nb_val_samples=50)
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#get result with 80 %
print('acc',sum(acc)/len(acc)*100,'%','val_acc',sum(val_acc)/len(val_acc)*100,'%')

