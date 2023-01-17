from tensorflow.keras import models

from tensorflow.keras import layers



# Sequential Model is a linear stack of layers

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))



print(model.summary())
# Load some useful libraries for machine learning

import numpy as np # Working with tensors and numerical computing

import matplotlib.pyplot as plt # Plotting charts and graphs

from tensorflow.keras import models # Creating ML deep learning models

from tensorflow.keras import layers  # Creating layers

from tensorflow.keras import optimizers # Customizing optimizers

import os # For working with the operating system e.g: creating dirs
base_dir = '/kaggle/input/intel-image-classification' # original directory of the data (read-only)

base_custom_dir = '/kaggle/working/custom' # custom directory where to copy the data too (all privileges) 



original_train_dir = os.path.join(base_dir, 'seg_train', 'seg_train')

original_test_dir = os.path.join(base_dir, 'seg_test', 'seg_test')

train_dir = os.path.join(base_custom_dir,  'seg_train', 'seg_train')

test_dir = os.path.join(base_custom_dir, 'seg_test', 'seg_test')

validation_dir = os.path.join(base_custom_dir, 'seg_val', 'seg_val' )
# A utility function to handle the moving of copying of the images.

import shutil



def copy_tree(src, dst, symlinks=False, ignore=None):

    for item in os.listdir(src):

        s = os.path.join(src, item)

        d = os.path.join(dst, item)

        if os.path.isdir(s):

            shutil.copytree(s, d, symlinks, ignore)

        else:

            shutil.copy2(s, d)
# Copy data from the original directory to the custom directory

# Since the original directory does not allow modification of the data,

# which is required for separating the validation data from the training data.

# Also the original training data is required for the final training of the model.

copy_tree(base_dir, base_custom_dir)
# Create a directory in the base custom directory for the validation set

os.makedirs(validation_dir) # Makes directories that may be missing in the path

print(os.listdir(base_custom_dir)) # Check if the files were moved successfully and the validation dir was made
# Separate training set from validation set

categories_dirs = os.listdir(train_dir) # list of folders in the custom trainin directory.

val_number = 300 # Last index of the image from each category

for category in categories_dirs:

    category_dir = os.path.join(train_dir, category) # Original category directory

    category_val_dir = os.path.join(validation_dir, category) # Validation direcoty

    if not os.path.exists(category_val_dir):

        os.mkdir(category_val_dir)

    print(f"Moving {category}")

    for i, filename in enumerate(os.listdir(category_dir)):

        file_path = os.path.join(category_dir, filename) # Original  file path

        file_val_path = os.path.join(category_val_dir, filename) # Validation file path

        

        if i > val_number:

            break;

        shutil.move(file_path, file_val_path)

    print(f"Finished Moving {category}")

        



print("Folders in the training directory.")

print(os.listdir(train_dir))

print("Folders in the testing directory.")

print(os.listdir(test_dir))

print("Folders in the validation directory")

print(os.listdir(validation_dir))



# Number of images in each validation category

for category in categories_dirs:

    print(category, len(os.listdir(os.path.join(validation_dir, category))))
# Display a sample of images from each category

fig = plt.figure(figsize=(10, 10))

fig.suptitle("Samples of images from the training data.")

number_of_samples = 2

sub_plot_number = 0

for category in categories_dirs:

    category_dir = os.path.join(train_dir, category)

    for i, filename in enumerate(os.listdir(category_dir)):

        file_path = os.path.join(category_dir, filename)



        if i >= number_of_samples:

            break

        else:

            plt.subplot(5, 5, sub_plot_number+1)

            plt.xticks([])

            plt.yticks([])

            image = plt.imread(file_path)

            plt.imshow(image, cmap=plt.cm.binary)

            plt.xlabel(category)

            sub_plot_number+=1

plt.show()
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Generate Image data from folders

# Since we have  a fairly large number of images it would not be wise to read them into memory,

# luckily keras comes with the ImageDataGenerator class that handles the actual loading of images

# it also handle normalization `rescale=1./2555` of images as well as,

# resizing images to the required size for the model `target_size=(150, 150)`

train_datagen = ImageDataGenerator(rescale=1./255) # Image generator for training data



test_datagen = ImageDataGenerator(rescale=1./255) # Image generator for test data



validation_datagen = ImageDataGenerator(rescale=1./255) # Image generator for validation data



# Generator for training data

train_datagen = train_datagen.flow_from_directory(

    train_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary',

    shuffle=True

)



print(train_datagen.class_indices)



# Generator for testing data

test_datagen = test_datagen.flow_from_directory(

    test_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary')



print(test_datagen.class_indices)



# Generator for validating data

validation_datagen = validation_datagen.flow_from_directory(

    validation_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary',

    shuffle=True

)



print(validation_datagen.class_indices)
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))

model.add(layers.Conv2D(128,(3, 3), activation="relu"))

model.add(layers.Flatten()) # Output from Conv2D is in the shape (height, width, depth) dense layers expect 1D array

model.add(layers.Dense(512, activation="relu"))

model.add(layers.Dense(6, activation="softmax")) # Outputs the probabilty of each of the classes/labels



print(model.summary())
model.compile(

    optimizer='Adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy'])
# steps_per_epoch = ceil(number_of_training_images / batch_size)

# validation_steps = ceil(number_of_validation_images / batch_size)

history = model.fit_generator(

    train_datagen,

    steps_per_epoch=383,

    epochs=35,

    validation_data=validation_datagen,

    validation_steps=57)
# Utility function to plot the results of training and validation performance

def plot_train_val_performance(history):

    history_dict = history.history

    accuracy = history_dict['accuracy']

    val_accuracy = history_dict['val_accuracy']

    loss = history_dict['loss']

    val_loss = history_dict['val_loss']



    epochs = range(1, len(accuracy) + 1)



    # Accuracy Performance

    plt.figure()

    plt.suptitle("Accuracy Performance")

    plt.plot(epochs, accuracy, 'b', label='Training Accuracy')

    plt.plot(epochs, val_accuracy, 'y', label='Validation Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()



    # Loss Performance

    plt.figure()

    plt.suptitle("Loss Performance")

    plt.plot(epochs, loss, 'b', label='Training Loss')

    plt.plot(epochs, val_loss, 'y', label='Validation Loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.show()

plot_train_val_performance(history)
# Data augmentation with Keras

datagen = ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')
# Example of data augmentation

import random

from tensorflow.keras.preprocessing import image



mountains_dir = os.path.join(train_dir, "mountain")



# generate a list of with the path to each image in the mountains directory

mountain_pic_paths = [os.path.join(mountains_dir, pic_name) for pic_name in os.listdir(mountains_dir)]



# select a random path of image from the sequence

random_mountain_path = random.choice(mountain_pic_paths)



# load the image

mountain_img = image.load_img(random_mountain_path, target_size=(150, 150))



# convert image to a numpy array

img_arr = image.img_to_array(mountain_img)



img_arr = img_arr.reshape((1,) + img_arr.shape)



i=0

for batch in datagen.flow(img_arr, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break

plt.show()
# Modify the train data generator to perform data augmentationdatagen = ImageDataGenerator(

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')

# The rest of the generator remain unchanged since its only the training data that needs augmentation

train_datagen = train_datagen.flow_from_directory(

    train_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary')


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation="relu"))

model.add(layers.Dense(6, activation="softmax"))



print(model.summary())
model.compile(

    optimizer = 'Adam',

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)
# steps_per_epoch = ceil(number_of_training_images / batch_size)

# validation_steps = ceil(number_of_validation_images / batch_size)

history = model.fit_generator(

    train_datagen,

    steps_per_epoch=383,

    epochs=35,

    validation_data=validation_datagen,

    validation_steps=57)
plot_train_val_performance(history)
# Load VGG16 model that comes bundled with keras

from tensorflow.keras.applications import VGG16



conv_base = VGG16(weights='imagenet',

                include_top=False,

                input_shape=(150, 150, 3))



print(conv_base.summary())
conv_base.trainable = False
model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dropout(0.7))

model.add(layers.Dense(256, activation="relu"))

model.add(layers.Dense(6, activation="softmax"))

print(model.summary())


model.compile(

    optimizer = 'Adam',

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)
# steps_per_epoch = ceil(number_of_training_images / batch_size)

# validation_steps = ceil(number_of_validation_images / batch_size)

history = model.fit_generator(

    train_datagen,

    steps_per_epoch=383,

    epochs=35,

    validation_data=validation_datagen,

    validation_steps=57)
plot_train_val_performance(history)
# unfreeze the layers in block5

from tensorflow.keras.applications import VGG16



conv_base = VGG16(

    weights='imagenet', 

    include_top=False, 

    input_shape=(150, 150, 3))



conv_base.trainable = True

print(conv_base.summary())



# Unfreeze the top layers

set_trainable = False

for layer in conv_base.layers:

    if layer == 'block5_conv1':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
# define our model

model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation="relu"))

model.add(layers.Dense(6, activation="softmax"))



print(model.summary())
model.compile(

    optimizer='Adam',

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)
# steps_per_epoch = ceil(number_of_training_images / batch_size)

# validation_steps = ceil(number_of_validation_images / batch_size)

history = model.fit_generator(

    train_datagen,

    steps_per_epoch=383,

    epochs=35,

    validation_data=validation_datagen,

    validation_steps=57)
plot_train_val_performance(history)
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Generate Image data from folders

# Since we have  a fairly large number of images it would not be wise to read them into memory,

# luckily keras comes with the ImageDataGenerator class that handles the actual loading of images

# it also handle normalization `rescale=1./2555` of images as well as,

# resizing images to the required size for the model `target_size=(150, 150)`

train_datagen = ImageDataGenerator(rescale=1./255) # Image generator for training data



test_datagen = ImageDataGenerator(rescale=1./255) # Image generator for test data



# Generator for training data

train_datagen = train_datagen.flow_from_directory(

    original_train_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary',

    shuffle=True

)



print(train_datagen.class_indices)



# Generator for testing data

test_datagen = test_datagen.flow_from_directory(

    original_test_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary')



print(test_datagen.class_indices)
model.compile(

    loss='sparse_categorical_crossentropy',

    optimizer='Adam',

    metrics=['accuracy']

)
# steps_per_epoch = ceil(number_of_training_images / batch_size)

# validation_steps = ceil(number_of_validation_images / batch_size)

history = model.fit_generator(

    train_datagen,

    steps_per_epoch=439,

    epochs=20

)
history_dict = history.history

accuracy = history_dict['accuracy']

loss = history_dict['loss']



epochs = range(1, len(accuracy) + 1)



# Accuracy Performance

plt.figure()

plt.suptitle("Accuracy Performance")

plt.plot(epochs, accuracy, 'b', label='Training Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



# Loss Performance

plt.figure()

plt.suptitle("Loss Performance")

plt.plot(epochs, loss, 'b', label='Training Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
results = model.evaluate_generator(test_datagen)

accuracy = results[1]

print(f"The model has an accuracy of {accuracy}")
model.save("intel_image_classification_V1.h5")