import os 

import numpy as np

import pandas as pd 

import random 

import PIL 

from matplotlib import pyplot as plt



import tensorflow as tf

from tensorflow import keras 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
#Preparing the dataset



base_dir = '../input/5100-midterm-data/Facial_Recog_Data/DatalfwTwoPeople'

seta = 'George_W_Bush'

setb = 'Colin_Powell'



#Each directory has its own subdirectores: train, validate and test

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')

test_dir = os.path.join(base_dir, 'test')



# test_path = os.listdir(train_dir)

# print(test_path)



#Each of the three subdirectories has two directories: seta and setb which correspond to GWB or CP

#Here we setting the paths for the directories 

def prepare_data(base_dir, seta, setb):

    print(base_dir, seta, setb)

    

    #Here, path is path-like object representing a file system path and join

    #join() will concatenate various path componets with one directory separator

    seta_train_dir = os.path.join(train_dir, seta)

    setb_train_dir = os.path.join(train_dir, setb)

    

    seta_valid_dir = os.path.join(validation_dir, seta)

    setb_valid_dir = os.path.join(validation_dir, setb)

    

    seta_train_fnames = os.listdir(seta_train_dir)

    setb_train_fnames = os.listdir(setb_train_dir)

    

    return seta_train_dir,setb_train_dir,seta_valid_dir, setb_valid_dir, seta_train_fnames, setb_train_fnames

    

#assign variables to returned values from prepare_data(base_dir, seta, setb)

seta_train_dir, setb_train_dir, seta_valid_dir, setb_valid_dir, seta_train_fnames, setb_train_fnames = prepare_data(base_dir, seta, setb)



#Set test directories by joining the test_dir to include seta and setb 

seta_test_dir = os.path.join(test_dir,seta)

setb_test_dir = os.path.join(test_dir, setb)



#os.listdir() returns a list containing the names of the entries in the directory given by the path

test_fnames_seta = os.listdir(seta_test_dir)

test_fnames_setb = os.listdir(setb_test_dir)





#This tensorflow function shifts and rotates the image

datagen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')

            

#Here is the plotting so user can see the augmentation 

img_path=os.path.join(seta_train_dir, seta_train_fnames[3])

img=load_img(img_path, target_size=(150,150))

x=img_to_array(img)

x=x.reshape((1,)+ x.shape)



i=0

for batch in datagen.flow(x, batch_size=1):

 plt.figure(i)

 imgplot = plt.imshow(array_to_img(batch[0]))

 i +=1

 print(i)

 if i % 5 == 0:

   break



#Convolution NN



from tensorflow.keras import layers

from tensorflow.keras import Model



#Define size of the input

img_input = layers.Input(shape=(150,150,3)) #3 dim as it is RGB



#2D cov 64 filters 

x = layers.Conv2D(64,3,activation ='relu')(img_input) #64 filters, 3x3 filter size 

x = layers.MaxPooling2D(2)(x) #stride is 2



#2D Convolution Layer with 128 filters of dimension 3x3 and ReLU activation algorithms -- the most common in image recognition 

x = layers.Conv2D(128, 3, activation = 'relu')(x)

x = layers.MaxPooling2D(2)(x)



#2D Convolution Layer with 256 filters of dimension 3x3 and ReLU activation algorithms

x = layers.Conv2D(256, 3, activation = 'relu')(x)

x = layers.MaxPooling2D(2)(x)



#2D Convolution Layer with 512 filters of dimension 3x3 and ReLU activation algorithms

x=layers.Conv2D(512, 3, activation='relu')(x)

x=layers.MaxPooling2D(2)(x)



#2D Convolution Layer with 512 filters of dimension 3x3 and ReLU activation algorithms

x=layers.Conv2D(512, 3, activation='relu')(x)



#Flatten Layer 

x = layers.Flatten()(x)





#Fully Connected layers with ReLU activation funciton 

x = layers.Dense(4096, activation='relu')(x)

x = layers.Dense(4096, activation='relu')(x)

x = layers.Dense(1000, activation='relu')(x)





#Dropout layer for optimization - used to optimize accuracy 

x = layers.Dropout(0.5)(x)



#Fully connected layers and Sigmoid Activation Function algorithm 

output = layers.Dense(1, activation='sigmoid')(x)





model = Model(img_input, output)



model.summary()
#Using binary crossentropy as the loss function and Adam Optimizer as the optimizing function 

model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['acc'])

#Creating a Data generator to rescale all images by 1/255 and creating a train_generator and validation_generator with a binary classification 

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



#Flow training images in batches of 20 using train_datagen generator 

train_generator = train_datagen.flow_from_directory(train_dir,

                                                   target_size=(150,150),

                                                   batch_size= 20,

                                                   class_mode= 'binary')





#Create Validation generator with images from the validation directory

validation_generator = train_datagen.flow_from_directory(validation_dir,

                                                        target_size=(150,150),

                                                        batch_size=20,

                                                        class_mode= 'binary')

#Use the MatPlotLib to show the images. 



import matplotlib.image as mpimg



#4x4 grid 

nrows = 5

ncols = 5



pic_index = 0



#Set up matplotlib fig and fit to 5x5 pics 

fig = plt.gcf()

fig.set_size_inches(ncols*5, nrows*5)



pic_index += 10

next_seta_pix = [os.path.join(seta_train_dir, fname)

                 for fname in seta_train_fnames[pic_index-10:pic_index]]



next_setb_pix = [os.path.join(setb_train_dir, fname)

                 for fname in setb_train_fnames[pic_index-10:pic_index]]



for i, img_path in enumerate(next_seta_pix + next_setb_pix):

    #Set up subplot 

    sp = plt.subplot(nrows, ncols, i+1)

    sp.axis = ('off')

    

    img = mpimg.imread(img_path)

    plt.imshow(img)

    

plt.show()

                



#Train the model 

mymodel = model.fit_generator(

            train_generator, 

            steps_per_epoch=4,

            epochs=4,

            validation_data= validation_generator,

            validation_steps = 4,

            verbose=2)
#Accuracy results for each training and validation epoch 

acc = mymodel.history['acc']

val_acc = mymodel.history['val_acc']



#Loss results for each each 

loss = mymodel.history['loss']

val_loss = mymodel.history['val_loss']



epochs = range(len(acc))



#Plotting accuracy for each training and vaildation epoch 

plt.plot(epochs, acc)

plt.plot(epochs, val_acc)

plt.title('Training and validation accuracy')



plt.figure()





#Plot loss for each 

plt.plot(epochs, loss)

plt.plot(epochs, val_loss)

plt.title('Training and validation loss')
#Testing the model on a random image from seta

train_img = random.choice(seta_train_fnames)

train_img_path = os.path.join(seta_train_dir, train_img)

train_img = load_img(train_img_path, target_size= (150,150))

plt.imshow(train_img)

train_img = (np.expand_dims(train_img, 0))

print(train_img.shape)

model.predict(train_img)
#Testing the model on a random image from setb

train_img = random.choice(setb_train_fnames)

train_img_path = os.path.join(setb_train_dir, train_img)

train_img = load_img(train_img_path, target_size= (150,150))

plt.imshow(train_img)

train_img = (np.expand_dims(train_img, 0))

print(train_img.shape)

model.predict(train_img)