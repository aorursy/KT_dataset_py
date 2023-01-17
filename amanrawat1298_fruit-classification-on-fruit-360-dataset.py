#importing libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os


import keras
from keras.preprocessing import image
from keras import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.models import load_model


# Train and test directory 
train_directory = '/kaggle/input/fruits/fruits-360/Training'
test_directory = '/kaggle/input/fruits/fruits-360/Test'
#opening a sample image
path = "/kaggle/input/fruits/fruits-360/Training/Apple Golden 1/15_100.jpg"
img = image.load_img(path)
img_array = image.img_to_array(img)/255.0
plt.imshow(img_array)
print(img_array.shape)
train_gen = image.ImageDataGenerator(
    rescale = 1.0/255,              # rescaling image from 0-255 to 0-1
    rotation_range = 40,            # 40 degree of random rotation
#     zoom_range = 0.3,             # not using zoom as zooming might make  classification difficult
    horizontal_flip = True,
    vertical_flip=True
)
val_gen = image.ImageDataGenerator(rescale=1.0/255)  #just rescaling on test data

# batch size : the images from directory will be provided to model in the batches of 32
batch_size = 32
train_generator = train_gen.flow_from_directory(
    train_directory,                 # training data directory
    target_size = (100, 100),        # target size of the images which will be fed into the model
    batch_size = batch_size,         # no. of images send for 1 epoch to the model
    class_mode = "categorical",      # as there are 131 classes 
)

val_generator = val_gen.flow_from_directory(
    test_directory,                  # test data directory
    target_size = (100, 100),
    batch_size = batch_size,
    class_mode = "categorical",
)
X,y = train_generator.next()
print(len(X), len(y), X.shape)  # See the generator is sending images and labels in the specified batch_size ie 32
# some images of our augmented dat
for i in np.arange(5):
    plt.imshow(X[i])
    plt.show()
#my model architecture: 

# 5 CNN layers  with pooling  , with relu activation
# and finally flattened vector is sent to fully connected dense layer ie the last layer 
# with 131  neurons and with softmax activation function


model = Sequential()
model.add(Conv2D(filters = 16,kernel_size=(2,2), input_shape=(100, 100, 3), activation= 'relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))   #used padding ='same' so that edges wont be cut so wont lose any features from there

model.add(Conv2D(filters = 32, kernel_size=(2,2) ,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64, kernel_size=(2,2),activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128, kernel_size=(2,2),activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 256, kernel_size=(2,2) ,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))


model.add(Flatten())   # to flatten the matrix to a 1D vector
model.add(Dropout(0.4)) # To add regularization
model.add(Dense(131, activation='softmax'))
model.summary()
#Compiling models with
# loss as categorical crossentropy as their are 130 classes
# Adam as optimizer
# metric as Accuracy
# model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Training the model using fit_generator otherwise it will give a huge load to RAM if all the images are converted into data matrix at the same time.
hist  = model.fit_generator(
    train_generator,                # training generator
    epochs = 7,
    validation_data= val_generator, # validation generator
    verbose=2,
)
model.save("model_fin.h5")
print("Model saved at", os.getcwd())
#Plot for accuracy 
plt.plot(hist.history['accuracy'])  
plt.plot(hist.history['val_accuracy'])  
plt.title('MODEL_Accuracy')  
plt.ylabel('ACCURACY')  
plt.xlabel('EPOCHS')  
plt.legend(['train_acc', 'test_acc'], loc='lower right')  
plt.figure()   
  

# Plot for loss   
plt.plot(hist.history['loss'])  
plt.plot(hist.history['val_loss'])  
plt.title('MODEL loss')  
plt.ylabel('LOSS')  
plt.xlabel('EPOCHS')  
plt.legend(['train_loss', 'test_loss'], loc='upper right')  
plt.show()

# taking a batch of 32 images from generator
X_test,Y_test=val_generator.next()
y_pred = model.predict(X_test)     # Predicted labels 

pred_labels = y_pred.argmax(axis=1)   # output is a sparse vector having one at the most close predicted label index                           
true_labels = Y_test.argmax(axis=1)   # axis =1 means argmax is calculated along with the row ,(columns collapsed)

print( np.sum(pred_labels==true_labels),"/",len(pred_labels), " fraction of corrected predictions over total predictions")
# Checking our model on test images 
# Title printed in green color if correctly predicted and in red if it is predicted wrong
for i in np.arange(len(X_test)):
    plt.figure(figsize=(2,2))
    plt.title("Predicted label : " + str(pred_labels[i]), color=("green" if pred_labels[i] == true_labels[i] else "red")) 
    plt.imshow(X_test[i])
    plt.show()
    
print( np.sum(pred_labels==true_labels),"/",len(pred_labels)," correctly predicted")
#loaded saved model 
loaded_model = load_model("model_fin.h5")
loaded_model.summary()
# Evaluating it on test set
loaded_model.evaluate(X_test, Y_test) # getting same predictions , thus successfully saved and loaded
model.evaluate(X_test, Y_test)
# Thank you .
