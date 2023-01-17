#imports

import os,random

from shutil import copyfile,copytree,rmtree

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

import numpy as np



#visualization

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



print(tf.__version__)

tf.test.is_gpu_available()
#lets set the train & validation path

image_dir = "/kaggle/input/the-simpsons-characters-dataset/simpsons_dataset/"

test_dir = "/kaggle/input/the-simpsons-characters-dataset/kaggle_simpson_testset/"
#I noticed that simpson_dataset is already inside simpsons folder which should be removed

#Image should be copied to working to have delete permission

copytree(image_dir,"/kaggle/working/simpsons")
train_dir = "/kaggle/working/simpsons/"

rmtree('/kaggle/working/simpsons/simpsons_dataset/')
#Lets go with 32x32 pixels for faster training

IMG_SHAPE = (32,32,3)

num_classes = 42
#Now lets augement our image data

datagen_train = ImageDataGenerator(rescale=1./255,

                                   rotation_range=30,

                                   width_shift_range=0.3,

                                   height_shift_range=0.3,

                                   horizontal_flip=True,fill_mode='nearest')

datagen_test = ImageDataGenerator(rescale=1./255)
#We are converting all our training data to size of 32x32 pixels for reducing training time 

#Test set are allowed to have 224x224 for easy visualization and we can convert to 32x32 during predictions

train_generator = datagen_train.flow_from_directory(train_dir,target_size=(32,32))

test_generator = datagen_test.flow_from_directory(test_dir,target_size=(224,224))

class_names = {v:k for k,v in train_generator.class_indices.items()}
#Now lets take a sample from the train set

X_train,y_train = next(train_generator)

for i in range(5):

    plt.imshow(X_train[i])

    plt.show()

    print("Label:",class_names[np.argmax(y_train[i])])
#model building

model = tf.keras.models.Sequential()



#Conv 1

model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',input_shape = IMG_SHAPE,activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

#Conv 2

model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Dropout(0.2))

#Conv 3

model.add(tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

#Conv 4

model.add(tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Dropout(0.2))

#Conv 5

model.add(tf.keras.layers.Conv2D(256,(3,3),padding='same',activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

#Conv 6

model.add(tf.keras.layers.Conv2D(256,(3,3),padding='same',activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256,activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.5))



model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))



plot_model(model,show_shapes=True)
#lets compile the model

model.compile(loss='categorical_crossentropy',

             metrics=['acc'],

             optimizer=tf.keras.optimizers.Adam(lr = 0.001))
#We are not going for model checkpoints/earlystopping since squeezing out the best model is not the scope of this kernel

#So we haven't made any train/validation splits here as well

#lets train the model

epochs = 30

batch_size = 32

model.fit_generator(train_generator,epochs=epochs)
#Lets save the model fro later use

model.save('simpsons_model.h5')
#Now model building is completed.

#So lets see how good our model preforms on the test data

img,label = next(test_generator)

for i in range(5):

    #reshaping the image as per the model input shape

    pred_img = cv2.resize(img[i],(32,32))

    res = model.predict(np.expand_dims(pred_img,axis=0))

    plt.imshow(img[i])

    plt.show()

    print("Predicted :",class_names[np.argmax(res)])            

#Save the model

export_dir = 'simpson_saved_model'

tf.saved_model.save(model,export_dir)
#Now lets choose the optimzation strategy

optimization = tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
#Generate the tflite model

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)

converter.optimizations = [optimization]

tflite_model = converter.convert()



#Now lets save the TFlite model

with open('simpsons.tflite','wb') as f:

    f.write(tflite_model)

#Time to test the TFlite mode

interpreter = tf.lite.Interpreter(model_content = tflite_model)

interpreter.allocate_tensors()



input_index = interpreter.get_input_details()[0]["index"]

output_index = interpreter.get_output_details()[0]["index"]
#let us take an image from the test set

rand_item = random.randint(0,len(img))

pred_img = cv2.resize(img[rand_item],(32,32))



interpreter.set_tensor(input_index, pred_img.reshape(-1,32,32,3))

interpreter.invoke()

res = interpreter.get_tensor(output_index)

print("Predicted :",class_names[np.argmax(res)])  

plt.imshow(img[rand_item])