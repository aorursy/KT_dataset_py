#Import basic things

import numpy as np

import pandas as pd
import tensorflow as tf
tf.__version__
#Checking if GPU is available

device_names = tf.test.gpu_device_name()

device_names
#Mounting Google Drive (ONLY FOR GOOGLE COLAB)

'''

from google.colab import drive

#drive.mount('/content/drive')

'''
#Where to download data (a folder) (For Google Colab)

'''

%cd/content/drive/My Drive/Colab Notebooks/dog-cat

'''
#Upload caggle API token as this is to downlaod from caggle (as .json file)



# To donlaod the .json file, at Caggle-

#     ->Go to My Account (Not My Profile)

#     ->Go to API Section and click on Create New API Token

#     ->Download the token, which will be saved as a .json file (e.g. kaggle.json)





'''

from google.colab import files 

files.upload()

'''
'''!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!rm kaggle.json'''
#This API command is copied from the dataset page in Caggle (On Google Colab)

'''!kaggle datasets download -d biaiscience/dogs-vs-cats'''
#Unzip the zipped files (In Google Colab)

'''from zipfile import ZipFile

file_name="/content/drive/My Drive/Colab Notebooks/dog-cat/dogs-vs-cats.zip"

print("Starting...")

with ZipFile(file_name,'r') as zip:

  zip.extractall()

print("Done!")'''
#Making directoris for reserving data and copying classified data

#Saving the foder addresses to variables for further use

import os, shutil
#original_dataset_dir = "/content/drive/My Drive/Colab Notebooks/dog-cat/train/train" #(For Colab)

original_dataset_dir = "../input/dogs-vs-cats/train/train" #For Caggle

#base_dir = "/content/drive/My Drive/Colab Notebooks/dog-cat-small" #(For Colab)

base_dir = "/kaggle/working/dog-cat-small" #For Caggle

os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')

os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)
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
filenames = ['cat.{}.jpg'.format(i) for i in range(1000)]

filenames
#Copying files to train folder and validation folder

#In each of these folders, there are two folders to save cats and dog images



for filename in filenames:

  source = os.path.join(original_dataset_dir, filename)

  destination = os.path.join(train_cats_dir, filename)

  shutil.copyfile(source, destination)
filenames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]

filenames
for filename in filenames:

  source = os.path.join(original_dataset_dir, filename)

  destination = os.path.join(validation_cats_dir, filename)

  shutil.copyfile(source, destination)
filenames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]

for filename in filenames:

  source = os.path.join(original_dataset_dir, filename)

  destination = os.path.join(test_cats_dir, filename)

  shutil.copyfile(source, destination)
'''Unfortunately, I inserted wrong files to the dog directories.

So, I deleted them and recreated the files agian.'''



# train_dogs_dir = os.path.join(train_dir, 'dogs')

# os.mkdir(train_dogs_dir)

# validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# os.mkdir(validation_dogs_dir)

# test_dogs_dir = os.path.join(test_dir, 'dogs')

# os.mkdir(test_dogs_dir)
filenames = ["dog.{}.jpg".format(i) for i in range(0, 1000)]

for filename in filenames:

  source = os.path.join(original_dataset_dir, filename)

  destination = os.path.join(train_dogs_dir, filename)

  shutil.copyfile(source, destination)



filenames = ["dog.{}.jpg".format(i) for i in range(1000, 1500)]

for filename in filenames:

  source = os.path.join(original_dataset_dir, filename)

  destination = os.path.join(validation_dogs_dir, filename)

  shutil.copyfile(source, destination)



filenames = ["dog.{}.jpg".format(i) for i in range(1500, 2000)]

for filename in filenames:

  source = os.path.join(original_dataset_dir, filename)

  destination = os.path.join(test_dogs_dir, filename)

  shutil.copyfile(source, destination)
#Checking if the desired number of data inserted into expected folders



dic = dict()

dic["Set"] = ["Train image", "", "Validation image", "", "Tesr image", ""]

dic["Type"] = ["Cats", "Dogs", "Cats", "Dogs", "Cats", "Dogs"]

dic["Number"] =[len(os.listdir(train_cats_dir)),

                len(os.listdir(train_dogs_dir)),

                len(os.listdir(validation_cats_dir)),

                len(os.listdir(validation_dogs_dir)),

                len(os.listdir(test_cats_dir)),

                len(os.listdir(test_dogs_dir))]

dicframe = pd.DataFrame(dic)

dicframe
#Making a convolutional model



from keras import models, layers

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation="relu"))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation="relu"))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation="relu"))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation = "relu"))

model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()
from keras import optimizers

model.compile(

  loss = "binary_crossentropy",

  optimizer = optimizers.RMSprop(lr = 1e-4),

  metrics = ["acc"]

)
#Fetching train data and validation data and processing the data



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1.00 / 255.0)

test_datagen = ImageDataGenerator(rescale = 1.00 / 255.0)



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size = (150, 150),

    batch_size = 20,

    class_mode = "binary"

)
validation_generator = test_datagen.flow_from_directory(

    validation_dir,

    target_size = (150, 150),

    batch_size = 20,

    class_mode = "binary"

)
for data_batch, labels_batch in train_generator:

  print("Data batch shape: ", data_batch.shape)

  print("Labels batch shape: ", labels_batch.shape)

  break
#Training the model with train data and judging this training with validation data

history = model.fit(

    train_generator,

    steps_per_epoch = 100,

    epochs = 30,

    validation_data = validation_generator,

    validation_steps = 50   

)
#Saving the model

!pip install pyyaml h5py 

model.save("cats_dogs_small_01.h5")
#Train accuracy and validation accuracy vs epoch graph

import matplotlib.pyplot as plt

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

plt.show()
#Data augmentation example



datagen = ImageDataGenerator(

    rotation_range = 40,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    shear_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True,

    fill_mode = 'nearest'

)



from keras.preprocessing import image 



fnames = [os.path.join(train_cats_dir, filename)

          for filename in os.listdir(train_cats_dir)]



img_path = fnames[69]

img = image.load_img(img_path, target_size=(150,150))

img_as_array = image.img_to_array(img)

img_as_array = img_as_array.reshape((1,) + img_as_array.shape)



i = 0

for batch in datagen.flow(img_as_array, batch_size=1):

  plt.figure()

  imgplot = plt.imshow(image.array_to_img(batch[0]))

  i += 1

  if(i == 7):

    break
#Making a new model with dropout regularisation



model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(150, 150, 3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation="relu"))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation="relu"))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation="relu"))

model.add(layers.Dense(1, activation="sigmoid"))

model.compile(

  loss = 'binary_crossentropy', 

  optimizer=optimizers.RMSprop(learning_rate=1e-4),

  metrics = ["acc"]

)
model.summary()
#fetching train data and augmenting this and fetching validation data, also processing them



train_datagen = ImageDataGenerator(

    rescale = 1.0 / 255.0,

    rotation_range = 40,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    shear_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True

)



test_datagen = ImageDataGenerator(

    rescale = 1.0 / 255.0

)



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size = (150, 150),

    batch_size = 20,

    class_mode = "binary"

)



validation_generator = test_datagen.flow_from_directory(

    validation_dir,

    target_size = (150, 150),

    batch_size = 20,

    class_mode = "binary"

)







history = model.fit(

    train_generator,

    steps_per_epoch = 100,

    epochs = 100,

    validation_data = validation_generator,

    validation_steps = 50

)

#Saving the model

model.save('cats_and_dogs_small_2.h5')
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))

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



plt.show()



#See this time result is much better than the previous model
#Reference

#Deep Learning with Python, François Chollet

#Chapter 5 (Section 5.2)