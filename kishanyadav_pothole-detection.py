# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
pwd
data_path = "/kaggle/input/pothole-and-plain-rode-images/My Dataset/"
# Check images
img = cv2.imread(data_path+"train"+'/'+"Pothole"+"/"+"1.jpg")
# pothole 
plt.imshow(img)
img.shape
# Data agumentation on train and test

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   rotation_range=15,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
# create dataset train
training_set = train_datagen.flow_from_directory(data_path + 'train',
                                                 target_size = (300, 300),
                                                 batch_size = 16,
                                                 class_mode = 'categorical',
                                                 shuffle=True)
# Create test data set
test_set = test_datagen.flow_from_directory(data_path + 'test',
                                            target_size = (300, 300),
                                            batch_size = 16,
                                            class_mode = 'categorical',
                                            shuffle = False)
# Model creation with changes

model = VGG16(input_shape=(300,300,3),include_top=False)

for layer in model.layers:
    layer.trainable = False

newModel = model.output
newModel = AveragePooling2D()(newModel)
newModel = Flatten()(newModel)
newModel = Dense(128, activation="relu")(newModel)
newModel = Dropout(0.5)(newModel)
newModel = Dense(2, activation='softmax')(newModel)

model = Model(inputs=model.input, outputs=newModel)
model.summary()
opt=Adam(learning_rate=0.0001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit_generator(training_set,
                              validation_data=test_set,
                              epochs=10)    
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(len(acc))
plt.plot(epochs,acc,label='Trainin_acc',color='blue')
plt.plot(epochs,val_acc,label='Validation_acc',color='red')
plt.legend()
plt.title("Training and Validation Accuracy")
plt.plot(epochs,loss,label='Training_loss',color='blue')
plt.plot(epochs,val_loss,label='Validation_loss',color='red')
plt.legend()
plt.title("Training and Validation loss")
class_dict = {0:'Plain',
              1:'Pothole'}
# New Data for testing 

file_path =  '/kaggle//input/test-pothole/plain3.png'
test_image = cv2.imread(file_path)
test_image = cv2.resize(test_image, (300,300),interpolation=cv2.INTER_CUBIC)
plt.imshow(test_image)
test_image = np.expand_dims(test_image,axis=0)
probs = model.predict(test_image)
pred_class = np.argmax(probs)

pred_class = class_dict[pred_class]

print('prediction class: ',pred_class)
# New Data for testing 

file_path =  '/kaggle//input/test-pothole/pothole3.jfif'
test_image = cv2.imread(file_path)
test_image = cv2.resize(test_image, (300,300),interpolation=cv2.INTER_CUBIC)
plt.imshow(test_image)
test_image = np.expand_dims(test_image,axis=0)
probs = model.predict(test_image)
pred_class = np.argmax(probs)

pred_class = class_dict[pred_class]

print('prediction class: ',pred_class)