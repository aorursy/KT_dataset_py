# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data Plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os, shutil

print(os.listdir("../input"))



import keras

from keras import layers

from keras import models

from keras import optimizers

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16



# Any results you write to the current directory are saved as output.
original_dataset_directory = "../input/train/train"



base_dir = "../cats_and_dogs_small"

os.mkdir(base_dir)
# Directories for the training, validation and test splits

train_dir = os.path.join(base_dir, "train")

os.mkdir(train_dir)



validation_dir = os.path.join(base_dir, "validation")

os.mkdir(validation_dir)



test_dir = os.path.join(base_dir, "test")

os.mkdir(test_dir)
# Directory with training, valiation and test pictures (for both Cats and Dogs)

train_cats_dir = os.path.join(train_dir, 'cats')

os.mkdir(train_cats_dir)



train_dogs_dir = os.path.join(train_dir, 'dogs')

os.mkdir(train_dogs_dir)



validation_cats_dir = os.path.join(validation_dir, 'cats')

os.mkdir(validation_cats_dir)



validation_dogs_dir = os.path.join(validation_dir, 'dogs')

os.mkdir(validation_dogs_dir)



test_cats_dir = os.path.join(test_dir, 'cats')

os.mkdir(test_cats_dir)



test_dogs_dir = os.path.join(test_dir, 'dogs')

os.mkdir(test_dogs_dir)
# Taking a sample from the original data set. Copying the first 1000 cat images to train_cats_dir

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]



for fnames in fnames:

    src = os.path.join(original_dataset_directory,fnames)

    dst = os.path.join(train_cats_dir,fnames)

    shutil.copyfile(src,dst)
#Copying the next 500 images to validation_cats_dir

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]



for fnames in fnames:

    src = os.path.join(original_dataset_directory,fnames)

    dst = os.path.join(validation_cats_dir,fnames)

    shutil.copyfile(src,dst)
#Copying the next 500 images to test_cats_dir

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]



for fnames in fnames:

    src = os.path.join(original_dataset_directory,fnames)

    dst = os.path.join(test_cats_dir,fnames)

    shutil.copyfile(src,dst)
# Taking a sample from the original data set. Copying the first 1000 dog images to train_dogs_dir

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]



for fnames in fnames:

    src = os.path.join(original_dataset_directory,fnames)

    dst = os.path.join(train_dogs_dir,fnames)

    shutil.copyfile(src,dst)
#Copying the next 500 images to validation_dogs_dir

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]



for fnames in fnames:

    src = os.path.join(original_dataset_directory,fnames)

    dst = os.path.join(validation_dogs_dir,fnames)

    shutil.copyfile(src,dst)
#Copying the next 500 images to test_dogs_dir

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]



for fnames in fnames:

    src = os.path.join(original_dataset_directory,fnames)

    dst = os.path.join(test_dogs_dir,fnames)

    shutil.copyfile(src,dst)
# As a sanity check lets count how many pictures are there in each data set

print("Total training cat images", len(os.listdir("../cats_and_dogs_small/train/cats")))

print("Total training dog images", len(os.listdir("../cats_and_dogs_small/train/dogs")))

print("Total validation cat images", len(os.listdir("../cats_and_dogs_small/validation/cats")))

print("Total validation dog images", len(os.listdir("../cats_and_dogs_small/validation/dogs")))

print("Total test cat images", len(os.listdir("../cats_and_dogs_small/test/cats")))

print("Total test dog images", len(os.listdir("../cats_and_dogs_small/test/dogs")))
# Creating the convnet model

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation = 'relu',

                       input_shape = (150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss = 'binary_crossentropy',

             optimizer = optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])
# Data Preprocessing



train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory(

train_dir,

target_size = (150,150),

batch_size = 20,

class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory(

validation_dir,

target_size = (150,150),

batch_size = 20,

class_mode = 'binary')
# Lets look at the output of one of these generators



for data_batch, lables_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('lables batch shape;', lables_batch.shape)

    break
# Model Training begins here

# Since we are using Data Generators, rather than static data sets, we would be using fit_generator to fit the data rather than traditional fit function



history = model.fit_generator(

train_generator,

steps_per_epoch=100,

epochs = 30,

validation_data=validation_generator,

validation_steps = 50)
# Saving the model post training



model.save('cats_and_dogs_small_1.h5')
# Lets plot the loss and accuracy of the model over the training and validation data during training



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'bo', label = 'Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()
# Lets take an example and see how does the image augmentation works



fnames = [os.path.join(train_cats_dir, fnames) for fnames in os.listdir(train_cats_dir)]



datagen = ImageDataGenerator(

rotation_range = 40,

width_shift_range=0.2,

height_shift_range=0.2,

shear_range=0.2,

horizontal_flip=True,

fill_mode='nearest')



img_path = fnames[3]

img = image.load_img(img_path, target_size = (150,150))



x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)



i = 0



for batch in datagen.flow(x, batch_size = 1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break



plt.show()
# Re creating the model with added drop out layer



model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation = 'relu',

                       input_shape = (150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy',

             optimizer = optimizers.RMSprop(lr=1e-4),

             metrics = ['acc'])
# Lets train the network using data augmentation and dropout



train_datagen = ImageDataGenerator(

rescale = 1./255,

rotation_range=40,

width_shift_range=0.2,

height_shift_range=0.2,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale = 1./255) # Please note that the validation data set is not being augmented. Its just being rescaled



# Passing the training directory to Image Data Generator constructor

train_generator=train_datagen.flow_from_directory(

train_dir,

target_size = (150,150),

batch_size=32,

class_mode='binary')



# Passing the test directory to its corresponding image pre processing constructor, just like the training data set

validation_generator = test_datagen.flow_from_directory(

validation_dir,

target_size=(150,150),

batch_size=32,

class_mode='binary')



# Fitting the model

history = model.fit_generator(

train_generator,

steps_per_epoch=100,

epochs=100,

validation_data = validation_generator,

validation_steps = 50)
# Saving the model post training



model.save('cats_and_dogs_small_2.h5')
# Plotting the results of the new network



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'bo', label = 'Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()
# We will be using VGG16 network for feature extraction

# Lets instantiate the VGG16 model



conv_base = VGG16(weights = 'imagenet',

                 include_top = False,

                 input_shape=(150,150,3))



# weights: It specifies the wieughts checkpoint from which to initialze the model

# include_top: Specifies whether we need to inlcude the "top" densely connected layer or not

# input_shape: Specifies the input shape which is required
# Instantiating an ImageDataGenerator intance for recaling the images

datagen = ImageDataGenerator(rescale = 1./255)

batch_size = 20



def extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 4, 4, 512))

    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_directory(

    directory,

    target_size=(150, 150),

    batch_size=batch_size,

    class_mode='binary')

    i = 0

    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count:

            break

    return features, labels



# Calling the extract features function on the training, validation and test directory



train_features, train_labels = extract_features(train_dir, 2000)

validation_features, validation_labels = extract_features(validation_dir, 1000)

test_features, test_labels = extract_features(test_dir, 1000)
# Next we need to flatten the train_features, test_features and validation features so that they can be transferred to fully connected layer



train_features = np.reshape(train_features,(2000, 4*4*512))

validation_features = np.reshape(validation_features,(1000, 4*4*512))

test_features = np.reshape(test_features,(1000, 4*4*512))
# Defining and training the densely connected classifier



model = models.Sequential()

model.add(layers.Dense(256,activation = 'relu', input_dim = 4*4*512))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1,activation = 'sigmoid'))



model.compile(optimizer=optimizers.RMSprop(lr=2e-5),

loss='binary_crossentropy',

metrics=['acc'])





history = model.fit(train_features, train_labels,

                   epochs = 30,

                   batch_size = 20,

                   validation_data = (validation_features, validation_labels))
# Plotting the results of the new network



# Plotting the results of the new network



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'bo', label = 'Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()
from keras import models

from keras import layers



model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))



model.summary()
print("The number of trainable weights before freezing the convolutional layer = ",len(model.trainable_weights))
# Freezing the conv base

conv_base.trainable = False
print("The number of trainable weights post freezing the convolutional layer", len(model.trainable_weights))
# We can start training the model with the data augmentation which we used previously



train_datagen = ImageDataGenerator(

rescale = 1./255,

rotation_range=40,

height_shift_range=0.2,

width_shift_range=0.2,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True,

fill_mode='nearest')



test_datagen = ImageDataGenerator(

rescale=1./255)



train_generator = train_datagen.flow_from_directory(

train_dir,

target_size = (150,150),

batch_size=20,

class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory(

validation_dir,

target_size=(150,150),

batch_size=20,

class_mode='binary')



model.compile(

loss='binary_crossentropy',

optimizer=optimizers.RMSprop(lr=2e-5),

metrics=['acc'])



history = model.fit_generator(

train_generator,

steps_per_epoch=100,

epochs=30,

validation_data=validation_generator,

validation_steps=50)


# Plotting the results of the new network



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'bo', label = 'Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()
# Freezing all layers upto a specific one 



conv_base.trainable = True

set_trainable = False



for layer in conv_base.layers:

    if layer.name == 'block5_conv1':

        set_trainable = True

    if set_trainable:

        layer.trainable == True

    else:

        layer.trainable == False
# Recompiling the model

model.compile(

loss='binary_crossentropy',

optimizer=optimizers.RMSprop(lr=1e-5),

metrics=['acc'])



history = model.fit_generator(

train_generator,

epochs=100,

steps_per_epoch=100,

validation_data=validation_generator,

validation_steps=50)
# Writing a smoothening function to smoothen the output plots



def smooth_curve(points, factor = 0.8):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous_point = smoothed_points[-1]

            smoothed_points.append(previous_point*factor + point*(1-factor))

        else:

            smoothed_points.append(point)

    return smoothed_points
plt.plot(epochs,

smooth_curve(acc), 'bo', label='Smoothed training acc')

plt.plot(epochs,

smooth_curve(val_acc), 'b', label='Smoothed validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs,

smooth_curve(loss), 'bo', label='Smoothed training loss')

plt.plot(epochs,

smooth_curve(val_loss), 'b', label='Smoothed validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
# Testing the model on Test data



test_generator = test_datagen.flow_from_directory(

test_dir,

target_size=(150, 150),

batch_size=20,

class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

print('test acc:', test_acc)