# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import glob

from keras.preprocessing import image

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Flatten, Dense, Dropout, Activation , Concatenate, Input , BatchNormalization

from keras.preprocessing.image import ImageDataGenerator ,img_to_array, load_img

from keras.callbacks import EarlyStopping

from keras.optimizers import SGD

from keras.utils import plot_model

from keras import Model
for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray/val'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
test_dir="/kaggle/input/chest-xray-pneumonia/chest_xray/test"

train_dir="/kaggle/input/chest-xray-pneumonia/chest_xray/train"

val_dir="/kaggle/input/chest-xray-pneumonia/chest_xray/val"



train_dir_noraml = train_dir + '/NORMAL'

train_dir_pneumonia = train_dir + '/PNEUMONIA'



test_dir_noraml  = test_dir + '/NORMAL'

test_dir_pneumonia  = test_dir + '/PNEUMONIA'



val_dir_noraml  = val_dir + '/NORMAL'

val_dir_pneumonia  = val_dir + '/PNEUMONIA'
print('number of normal training images - ',len(os.listdir(train_dir_noraml)))

print('number of pneumonia training images - ',len(os.listdir(train_dir_pneumonia)))

print('----------------------------------------------------------------------')

print('number of normal testing  images - ',len(os.listdir(test_dir_noraml)))

print('number of pneumonia testing  images - ',len(os.listdir(test_dir_pneumonia)))

print('----------------------------------------------------------------------')

print('number of normal validation  images - ',len(os.listdir(val_dir_noraml)))

print('number of pneumonia validation  images - ',len(os.listdir(val_dir_pneumonia)))
data_generator = ImageDataGenerator(rescale= 1./255 ,shear_range = 0.2,zoom_range = 0.2)
batch_size = 64

training_data = data_generator.flow_from_directory(directory = train_dir,

                                                   target_size = (150, 150),

                                                   class_mode='binary',

                                                   batch_size = batch_size)



testing_data = data_generator.flow_from_directory(directory = test_dir,

                                                  target_size = (150, 150),

                                                  class_mode='binary',

                                                  batch_size = batch_size)



test_generator = data_generator.flow_from_directory(directory = val_dir,

                                                  target_size = (150, 150),

                                                  class_mode='binary',

                                                  batch_size = 6)
set(training_data.classes)
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=2, patience=8)
input_model = Input(training_data.image_shape)





model1 = Conv2D(32,(7,7), activation='relu')(input_model)

model1 = Conv2D(32,(7,7), activation='relu', padding='same')(model1)

model1 = Conv2D(32,(6,6), activation='relu', padding='same')(model1)

model1 = BatchNormalization()(model1)

model1 = MaxPooling2D((2,2))(model1)

model1 = Conv2D(64,(6,6), activation='relu' ,padding='same')(model1)

model1 = Conv2D(64,(5,5), activation='relu' ,padding='same')(model1)

model1 = BatchNormalization()(model1)

model1 = AveragePooling2D((2, 2))(model1)

model1 = Conv2D(128,(5,5), activation='relu' ,padding='same')(model1)

model1 = Conv2D(128,(5,5), activation='relu' ,padding='same')(model1)

model1 = Conv2D(128,(5,5), activation='relu' ,padding='same')(model1)

model1 = BatchNormalization()(model1)

model1 = AveragePooling2D((2, 2))(model1)

model1 = Conv2D(512,(4,4), activation='relu' ,padding='same')(model1)

model1 = Conv2D(512,(4,4), activation='relu' ,padding='same')(model1)

model1 = Conv2D(512,(4,4), activation='relu' ,padding='same')(model1)

model1 = BatchNormalization()(model1)

model1 = MaxPooling2D((2, 2))(model1)

model1 = Conv2D(512,(3,3), activation='relu' ,padding='valid')(model1)

model1 = Conv2D(512,(3,3), activation='relu' ,padding='valid')(model1)

model1 = Conv2D(512,(3,3), activation='relu' ,padding='valid')(model1)

model1 = BatchNormalization()(model1)

model1 = AveragePooling2D((2, 2))(model1)

model1 = Flatten()(model1)

#########################################################                          

model2 = Conv2D(32,(4,4), activation='relu')(input_model)  

model2 = Conv2D(32,(4,4), activation='relu', padding='same')(model2)

model2 = Conv2D(32,(4,4), activation='relu', padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = MaxPooling2D((3, 3))(model2)

model2 = Conv2D(64,(3,3), activation='relu', padding='same')(model2) 

model2 = Conv2D(64,(3,3), activation='relu', padding='same')(model2)

model2 = Conv2D(64,(3,3), activation='relu', padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Conv2D(128,(3,3), activation='relu', padding='same')(model2)

model2 = Conv2D(128,(3,3), activation='relu', padding='same')(model2) 

model2 = Conv2D(128,(2,2), activation='relu' ,padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Conv2D(256,(2,2), activation='relu' ,padding='same')(model2)

model2 = Conv2D(256,(2,2), activation='relu' ,padding='same')(model2)

model2 = Conv2D(256,(2,2), activation='relu' ,padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Conv2D(512,(1,1), activation='relu' ,padding='same')(model2)

model2 = Conv2D(512,(1,1), activation='relu' ,padding='valid')(model2)

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Conv2D(1024,(1,1), activation='relu' ,padding='valid')(model2)

model2 = Conv2D(1024,(1,1), activation='relu' ,padding='valid')(model2)

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Flatten()(model2)

########################################################

merged = Concatenate()([model1, model2])

merged = Dense(units = 512, activation = 'relu')(merged)

merged = BatchNormalization()(merged)

merged = Dropout(rate = 0.2)(merged)

merged = Dense(units = 64, activation = 'relu')(merged)

merged = Dense(units = 32, activation = 'relu')(merged)

merged = Dense(units = 16, activation = 'relu')(merged)

merged = Dense(units = 8, activation = 'relu')(merged)

merged = Dense(units = 4, activation = 'relu')(merged)

merged = Dense(units = 2, activation = 'relu')(merged)

output = Dense(activation = 'sigmoid', units = 1)(merged)

#output = Dense(units = len(set(training_data.classes)), activation = 'softmax')(merged)



model = Model(inputs= [input_model], outputs=[output])
sgd = SGD(lr=0.01, momentum=0.9)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True)
history =  model.fit_generator(training_data,epochs = 30,

                               steps_per_epoch = len(training_data),

                               validation_data = testing_data ,

                               validation_steps = len(testing_data),

                               callbacks=[es],

                               verbose=1)
model.save_weights("weights.h5")
val_loss = history.history['val_loss']

loss = history.history['loss']



plt.plot(val_loss)

plt.plot(loss)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Val error','Train error'], loc='upper right')

plt.savefig('plot_error.png')

plt.show()
val_accuracy = history.history['val_accuracy']

accuracy = history.history['accuracy']



plt.plot(val_accuracy)

plt.plot(accuracy)

plt.xlabel('Epochs')

plt.ylabel('accuracy')

plt.legend(['Val accuracy','Train accuracy'], loc='lower right')

plt.savefig( 'plot_accuracy.png')

plt.show()
# evaluate the model

scores = model.evaluate_generator(test_generator)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
test_generator = data_generator.flow_from_directory(directory = val_dir,

                                                  target_size = (150, 150),

                                                  class_mode= None,

                                                  batch_size = 8)
pred = model.predict_generator(test_generator)

pred = pred.reshape(1,16)

predicted_class_indices= np.round_(pred)

labels = (test_generator.class_indices)

print(predicted_class_indices)

print (labels)
pred