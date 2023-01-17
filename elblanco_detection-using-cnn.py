# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf #tensorflow using for Image Processing
from pathlib import Path
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

epochs = 15
train_directory = '../input/10-monkey-species/training/training'
validation_directory = '../input/10-monkey-species/validation/validation'

label = pd.read_csv('../input/10-monkey-species/monkey_labels.txt')

label.columns = label.columns.str.strip()
type(label)
label['Latin Name'] = label['Latin Name'].str.replace("\t","")
label

# Any results you write to the current directory are saved as output.
CName = list(label["Common Name"])
labels = {}
for i in range(10):
    labels[i] = CName[i].strip()

print(labels)

plt.figure(figsize=(20, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    class_name = 'n' + str(i)
    plt.imshow(plt.imread('../input/10-monkey-species/validation/validation/' + class_name + '/' + class_name + '00.jpg'))
    plt.xlabel(class_name)
plt.figure(figsize=(20, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    class_name = 'n' + str(i)
    plt.imshow(plt.imread('../input/10-monkey-species/training/training/' + class_name + '/' + class_name + '023.jpg'))
    plt.xlabel(class_name)
training_data = Path(train_directory) 
validation_data = Path(validation_directory) 

train_df = []
for folder in os.listdir(training_data):
    # Define the path to the images
    imgs_path = training_data / folder
    
    # Get the list of all the images stored in that directory
    imgs = sorted(imgs_path.glob('*.jpg'))
    
    # Store each image path and corresponding label 
    for img_name in imgs:
        train_df.append((str(img_name), (str(folder)[1])))


train_df = pd.DataFrame(train_df, columns=['image', 'label'], index=None)
# shuffle the dataset 
train_df = train_df.sample(frac=1.).reset_index(drop=True)
#print(train_df)


validation_df = []
for folder in os.listdir(validation_data):
    # Define the path to the images
    imgs_path = validation_data / folder
    
    
    # Get the list of all the images stored in that directory
    imgs = sorted(imgs_path.glob('*.jpg'))
    
    # Store each image path and corresponding label 
    for img_name in imgs:
        validation_df.append((str(img_name), (str(folder)[1])))


validation_df = pd.DataFrame(validation_df, columns=['image', 'label'], index=None)
# shuffle the dataset 
validation_df = validation_df.sample(frac=1.).reset_index(drop=True)
#validation_df


print("Total number of Images in Training Set " + str(len(train_df)))

print("Total number of Images in Validation Set " + str(len(validation_df)))
from PIL import Image
for i in range(10):
    class_name = 'n' + str(i)
    path = '../input/10-monkey-species/training/training/' + class_name + '/' + class_name + '023.jpg'
    print(path)
    img = Image.open(path)
    width,height = img.size
    print("ht = " + str(height) + " width = " + str(width) )
print(train_df.label)
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1.0/255)

image_size = 224

train_generator = train_datagen.flow_from_directory(
    directory = train_directory,
    batch_size=20,
    target_size=(image_size, image_size),
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    directory=validation_directory,
    target_size=(image_size, image_size),
    shuffle=False,
    class_mode='categorical')

batch_size = 32
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size,image_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer = tf.optimizers.Adam(),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])
model.summary()
history = model.fit_generator(train_generator, steps_per_epoch = len(train_df)/batch_size, epochs=120, verbose=1, callbacks=None,
                        validation_data=validation_generator, validation_steps=len(validation_df)/batch_size)
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc ,label = "Training Data")
plt.plot  ( epochs, val_acc ,label = "Validation Data")
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.title ('Training and validation accuracy')
plt.legend(loc='best')
plt.show()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss ,label = "Training Data")
plt.plot  ( epochs, val_loss ,label = "Validation Data")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.title ('Training and validation loss'   )
plt.legend(loc = "best")
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
#ax1.plot(history.history['loss'], color='b', label="Training loss")
#ax1.plot(history.history['val_loss'], color='r', label="validation loss")
#ax1.set_xticks(np.arange(1, epochs, 1))


#ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
#ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

#legend = plt.legend(loc='best', shadow=True)
#plt.tight_layout()
#plt.show()'''
