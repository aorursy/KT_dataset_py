import numpy as np  # Data manipulation
import pandas as pd # Dataframe manipulation 
import matplotlib.pyplot as plt # Plotting the data and the results
import matplotlib.image as mpimg # For displaying imagees
%matplotlib inline


from keras import models
from keras import layers
import keras.preprocessing  as kp
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import optimizers
train_datagen = ImageDataGenerator( # Data Augumentation for test data
rescale=1./255,
rotation_range=30,
shear_range=0.3,
zoom_range=0.3
)

test_datagen = ImageDataGenerator(rescale=1./255)
train_gen=train_datagen.flow_from_directory('../input/gender-recognition-200k-images-celeba/Dataset/Train',
                                            target_size=(150,150),
                                            batch_size=300,
                                            class_mode='binary')
valid_gen=test_datagen.flow_from_directory('../input/gender-recognition-200k-images-celeba/Dataset/Validation',
                                           target_size=(150,150),
                                           batch_size=300,
                                           class_mode='binary')
kernel_s=(3,3) # The size of kernel
model=models.Sequential()
model.add(layers.Conv2D(32,kernel_s,activation='relu',input_shape=(150,150,3),
                        kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,kernel_s,activation='relu'))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Conv2D(64,kernel_s,activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,kernel_s,activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,kernel_s,activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(train_gen,steps_per_epoch=70,epochs=30,
                  validation_data=valid_gen,validation_steps=50)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'ro', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'ro', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()
test_datagen1 = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen1.flow_from_directory(
'../input/gender-recognition-200k-images-celeba/Dataset/Test',
target_size=(150,150),
batch_size=64,
class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
print('test_loss:',test_loss)
test_datagen2 = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen2.flow_from_directory(
'../input/gender-classification-dataset/Training',
target_size=(150,150),
batch_size=64,
class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
print('test_loss:',test_loss)
test_datagen3 = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen1.flow_from_directory(
'../input/gender-classification-dataset/Validation',
target_size=(150,150),
batch_size=64,
class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
print('test_loss:',test_loss)