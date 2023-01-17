#global imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os, shutil
import numpy as np

import warnings
warnings.filterwarnings('ignore')
x = np.array([14,12,3,41,121])
print("Dimensions of the tensor is: {0}".format(x.ndim))
#getting the data first
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#Defining our network architecture

baseline_nn = models.Sequential()
baseline_nn.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,))) #relu = max(n,0)
baseline_nn.add(layers.Dense(10, activation='softmax')) #softmax will return a probability score of all the digits
baseline_nn.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
baseline_nn.summary()
#before training,we will create a single feature axes of an image, reshaping all the values in a 1D tensor
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
#One hot encode the train and test labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
baseline_nn.fit(train_images, train_labels, epochs=5, batch_size=128)
baseline_test_loss, baseline_test_acc = baseline_nn.evaluate(test_images, test_labels)
print('BASELINE TEST ACCURACY: {0} \nBASELINE TEST LOSS: {1}'.format(baseline_test_acc,baseline_test_loss))
#Initializing a convnet

basic_cnn = models.Sequential()
basic_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
basic_cnn.add(layers.MaxPooling2D((2, 2)))
basic_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
basic_cnn.add(layers.MaxPooling2D((2, 2)))
basic_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))

basic_cnn.summary()
#Now let's attach a convnet to a flatten layer to predict outputs
basic_cnn.add(layers.Flatten())
basic_cnn.add(layers.Dense(64, activation='relu'))
basic_cnn.add(layers.Dense(10, activation='softmax'))

basic_cnn.summary()
#To make the input clear, we gotta reformat the input data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
basic_cnn.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
basic_cnn.fit(train_images, train_labels, epochs=5, batch_size=64)
basic_cnn_test_loss, basic_cnn_test_acc = basic_cnn.evaluate(test_images, test_labels)
print('BASIC CNN TEST ACCURACY: {0} \nBASIC CNN TEST LOSS: {1}'.format(basic_cnn_test_acc,basic_cnn_test_loss))
basic_cnn_no_max_pool = models.Sequential()
basic_cnn_no_max_pool.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
basic_cnn_no_max_pool.add(layers.Conv2D(64, (3,3), activation='relu'))
basic_cnn_no_max_pool.add(layers.Conv2D(64, (3,3), activation='relu'))
basic_cnn_no_max_pool.add(layers.Flatten())
basic_cnn_no_max_pool.add(layers.Dense(512, activation='relu'))
basic_cnn_no_max_pool.add(layers.Dense(10, activation='softmax'))
basic_cnn_no_max_pool.summary()
#The path to the directory where the original dataset is stored(uncompressed)
original_dataset_dir = '../input/dogs_cats_original'

#The path to the directory where we will store our smaller dataset
base_dir = '../dogs_cats_small'
os.mkdir(base_dir)

#Making directories for training, validating and testing sub-directories
for name in ['train', 'validation', 'test']:
    exec("{0}_dir = os.path.join(base_dir,'{1}')".format(name, name))
    exec("os.mkdir({0}_dir)".format(name))

#mapper_dictionary
mapper ={
    "train": (0, 1000),
    "validation": (1000, 1500),
    "test": (1500, 2000)
}

label_mapper = {
    "cats": "cat",
    "dogs": "dog"
}
    
    
#Filling directories with cat and dog images
for subdirectory in ['train','validation','test']:
    subdirectory_range = mapper[subdirectory]
    for label in ['cats','dogs']:
        exec("{0}_{1}_dir = os.path.join({2}_dir,'{3}')".format(subdirectory, label, subdirectory, label))
        exec("os.mkdir({0}_{1}_dir)".format(subdirectory, label))
        
        fnames = ['{0}.{1}.jpg'.format(label_mapper[label],i) for i in range(subdirectory_range[0], subdirectory_range[1])]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            exec("dst = os.path.join({0}_{1}_dir, fname)".format(subdirectory,label))
            shutil.copy(src, dst)
cats_dogs_scratch = models.Sequential()
cats_dogs_scratch.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
cats_dogs_scratch.add(layers.MaxPooling2D((2, 2)))
cats_dogs_scratch.add(layers.Conv2D(64, (3, 3), activation='relu'))
cats_dogs_scratch.add(layers.MaxPooling2D((2, 2)))
cats_dogs_scratch.add(layers.Conv2D(128, (3, 3), activation='relu'))
cats_dogs_scratch.add(layers.MaxPooling2D((2, 2)))
cats_dogs_scratch.add(layers.Conv2D(128, (3, 3), activation='relu'))
cats_dogs_scratch.add(layers.MaxPooling2D((2, 2)))
cats_dogs_scratch.add(layers.Flatten())
cats_dogs_scratch.add(layers.Dense(512, activation='relu'))
cats_dogs_scratch.add(layers.Dense(1, activation='sigmoid')) #sigmoid layer as it is a binary classification problem

cats_dogs_scratch.summary()
cats_dogs_scratch.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])
#Rescaling images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    #This is the target directory 
    train_dir,
    #Resizing the images to 150x150
    target_size=(150, 150),
    batch_size=20,
    #Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
history = cats_dogs_scratch.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)
#Please save model after training unless you want it to be deleted
cats_dogs_scratch.save("cats_and_dogs_scratch.h5")
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
# This is module with image preprocessing utilities
from tensorflow.keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# We pick one image to "augment"
img_path = fnames[3]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()
cats_dog_data_aug = models.Sequential()
cats_dog_data_aug.add(layers.Conv2D(32, (3, 3), activation='relu',
                       input_shape=(150, 150, 3)))
cats_dog_data_aug.add(layers.MaxPooling2D((2, 2)))
cats_dog_data_aug.add(layers.Conv2D(64, (3, 3), activation='relu'))
cats_dog_data_aug.add(layers.MaxPooling2D((2, 2)))
cats_dog_data_aug.add(layers.Conv2D(128, (3, 3), activation='relu'))
cats_dog_data_aug.add(layers.MaxPooling2D((2, 2)))
cats_dog_data_aug.add(layers.Conv2D(128, (3, 3), activation='relu'))
cats_dog_data_aug.add(layers.MaxPooling2D((2, 2)))
cats_dog_data_aug.add(layers.Flatten())
cats_dog_data_aug.add(layers.Dropout(0.5))
cats_dog_data_aug.add(layers.Dense(512, activation='relu'))
cats_dog_data_aug.add(layers.Dense(1, activation='sigmoid'))

cats_dog_data_aug.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])
cats_dog_data_aug.summary()
#Training our network using the data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#NOTE: VALIDATION DATA MUST NOT BE AUGMENTED!!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

history = cats_dog_data_aug.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)
cats_dog_data_aug.save('cats_and_dogs_data_aug.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
